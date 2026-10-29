#!/usr/bin/env python3
"""Bounded single-GPU production-trainer integration, with real tiny V/A/T modules.

Run with the existing OpenWAM Python environment (no downloads):
  CUDA_HOME="$CONDA_PREFIX" python -m scripts.validation.touchwam_trainer_smoke --output /tmp/touchwam-trainer-smoke

Launches three sequential one-rank torchrun/NCCL phases: uninterrupted training,
interruption after a complete checkpoint, and resume. Frozen T5/VAE preparation
is replaced by deterministic small latents; the actual model loss, optimizer,
Accelerator/ZeRO-2, trainer loop, scheduler and checkpoint code are exercised.
Never loads Alpha, formal datasets, or more than a tiny model. This is not a
multi-GPU or full-model capacity acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import socket
from types import MethodType
from unittest.mock import patch


def phase(args):
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator, DeepSpeedPlugin
    from omegaconf import OmegaConf

    from openwam.dataloader.mixture import MixtureDataset
    from openwam.tactile.registry import REGION_NAMES
    from openwam.tactile.regions import INTENSITY_CODEC, RGB_CODEC
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.checkpointing import find_latest_accel_state
    from openwam.train.utils.seeding import seed_everything
    from scripts.validation.touchwam_multisource_gpu import make_inputs, tiny_model
    import openwam.train.openwam_trainer as trainer_module

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This smoke test deliberately uses exactly one GPU process")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free; leaving existing GPU work untouched")
    torch.cuda.set_per_process_memory_fraction(args.max_allocated_mib * 1024**2 / total)
    torch.cuda.reset_peak_memory_stats()
    seed_everything(42)
    plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2,
                            gradient_clipping=1.0, offload_optimizer_device="none")
    # Same algorithm; bounded communication buffers suit a 440k-parameter model.
    plugin.deepspeed_config["zero_optimization"].update(
        reduce_bucket_size=1000000, allgather_bucket_size=1000000, ignore_unused_parameters=True)
    accelerator = Accelerator(deepspeed_plugin=plugin, gradient_accumulation_steps=2, mixed_precision="bf16")

    class Source(torch.utils.data.Dataset):
        action_dim = 80

        def __init__(self, mode, codec=None, encoder_id=0):
            self.training_modality = mode
            self.tactile_codec = codec
            self.encoder_id = encoder_id
            self.tactile_observation_shape = ((4, 6) if encoder_id == 2 else (32, 32)) if args.heterogeneous and codec else None

        def __len__(self):
            return 2

        def __getitem__(self, index):
            result = {"training_modality": self.training_modality}
            if "tactile" in self.training_modality:
                result["tactile_region_names"] = [REGION_NAMES[i] for i in (58, 59)]
                result["tactile_codec"] = self.tactile_codec
                result["smoke_encoder_id"] = self.encoder_id
                if self.tactile_observation_shape is not None:
                    result["tactile_observation_maps"] = torch.zeros(1, 2, 3 if self.encoder_id == 1 else 1,
                                                                     *self.tactile_observation_shape)
                    result["tactile_observation_shape"] = self.tactile_observation_shape
            return result

    modes = ["video_only", "video_action", "video_tactile", "video_action_tactile"]
    sources = [Source(mode, INTENSITY_CODEC if "tactile" in mode else None) for mode in modes]
    if args.rgb:
        sources.extend(Source(mode, RGB_CODEC, 1) for mode in modes if "tactile" in mode)
    if args.heterogeneous:
        sources.extend(Source(mode, INTENSITY_CODEC, 2) for mode in modes if "tactile" in mode)
    names = [source.training_modality + ("_rgb" if source.tactile_codec == RGB_CODEC else "")
             + ("_pressure" if source.encoder_id == 2 else "")
             for source in sources]
    total_steps = 2 * len(sources)
    dataset = MixtureDataset(sources, names=names, seed=42, strict_action_dim=True,
                             source_configs=[{"type": "synthetic_smoke"}] * len(sources))
    output = Path(args.output).resolve()
    resume = json.loads((output / "pilot.json").read_text())["run_dir"] if args.phase == "resume" else None
    cfg = OmegaConf.create({"training": {
        "num_epochs": 1, "max_steps": total_steps, "batch_size": 1, "gradient_accumulation_steps": 2,
        "dataset_num_workers": 0, "max_grad_norm": 1.0, "learning_rate": 1e-5,
        "weight_decay": 0.01, "offload_optimizer_device": "none", "zero_stage": 2,
        "lr_scheduler": "cosine", "warmup_ratio": 0.25, "lr_min_ratio": 0.01,
        "save_steps": 3, "save_full_states_for_resume": True, "keep_last_k_ckpts": -1,
        "resume_ckpt_path": resume, "output_path": str(output / args.phase)},
        "project": {"seed": 42, "wandb": None}, "model": {}})
    observation = None
    if args.heterogeneous:
        norm = dict(mean=[0., 0., 0.], std=[1., 1., 1.], pre_scale=2, pre_bias=-1)
        observation = dict(kind="ftp1_heterogeneous_v1", initialization="random", image_size=32,
            image_patch_size=8, image_width=32, image_front_layers=1, image_shared_layers=1,
            image_heads=4, matrix_width=16, freeze_pretrained=False,
            sensors=[dict(name="Sharpa", kind="image", input_representation="deform"),
                     dict(name="Optical", kind="image", input_representation="rgb"),
                     dict(name="Pressure", kind="matrix", input_shape=[4, 6])],
            source_to_encoder={"sharpa_real":["Sharpa"], "optical":["Optical"], "pressure":["Pressure"]},
            normalizations={"sharpa_real":{"Sharpa":norm}, "optical":{"Optical":norm}})
    model = tiny_model(torch.device("cpu"), torch.bfloat16, rgb_enabled=args.rgb, observation_encoder=observation)
    model.set_training_runtime(use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
                               max_timestep_boundary=1.0, min_timestep_boundary=0.0)

    seen_codecs = set()
    seen_encoders = set()

    def prepare_latents(self, samples):
        if len(samples) != 1:
            raise ValueError("Smoke input preparation expects one sample")
        codec = samples[0].get("tactile_codec")
        if codec is not None:
            seen_codecs.add(codec)
        inputs = make_inputs(samples[0]["training_modality"], self.device, self.dtype, tactile_codec=codec)
        if args.heterogeneous and codec is not None:
            idx = samples[0]["smoke_encoder_id"]
            seen_encoders.add(idx)
            shape = (4, 6) if idx == 2 else (32, 32)
            channels = 3 if idx == 1 else 1
            inputs.update(tactile_encoder_ids=torch.full((1, 2), idx, device=self.device, dtype=torch.long),
                tactile_sensor_ids=torch.full((1, 2), idx, device=self.device, dtype=torch.long),
                tactile_observation_maps=torch.rand(1, 4, 2, channels, *shape, device=self.device, dtype=self.dtype),
                tactile_observation_spatial_valid=torch.ones(1, 2, 1, *shape, device=self.device, dtype=torch.bool))
        return inputs

    model.prepare_inputs = MethodType(prepare_latents, model)
    model.save_assets_for_deployment = MethodType(lambda self, path, cfg: None, model)
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg, trainer.dataset, trainer.accelerator = cfg, dataset, accelerator
    trainer.architecture = model
    trainer._rank, trainer._run_seed, trainer._ckpt_source_dir = 0, 42, None
    trainer.lambda_video = trainer.lambda_action = 1.0
    real_save = trainer_module.save_full_state

    class PlannedInterruption(Exception):
        pass

    def checkpoint(*values, **kwargs):
        real_save(*values, **kwargs)
        if args.phase == "pilot" and values[2] == 4:
            raise PlannedInterruption()

    started = time.monotonic()
    try:
        with patch.object(trainer_module, "save_full_state", checkpoint):
            try:
                trainer.train()
            except PlannedInterruption:
                pass
        run = Path(resume) if resume else next((output / args.phase).iterdir())
        state_dir = find_latest_accel_state(str(run))
        if not state_dir:
            raise RuntimeError("Production trainer did not retain a complete state")
        meta = json.loads((Path(state_dir) / "trainer_state.json").read_text())

        def tree_hash(value):
            digest = hashlib.sha256()

            def visit(obj):
                if isinstance(obj, torch.Tensor):
                    tensor = obj.detach().cpu().contiguous()
                    digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
                elif isinstance(obj, dict):
                    for key in sorted(obj, key=str):
                        digest.update(str(key).encode())
                        visit(obj[key])
                elif isinstance(obj, (tuple, list)):
                    for item in obj:
                        visit(item)
                else:
                    digest.update(repr(obj).encode())
            visit(value)
            return digest.hexdigest()

        unwrapped = accelerator.unwrap_model(trainer.architecture)
        rows = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text().splitlines()]
        seen = sorted({name.removeprefix("samples_") for row in rows for name, value in row.items()
                       if name.startswith("samples_") and name.removeprefix("samples_") in modes and value})
        report = {"status": "passed", "phase": args.phase, "run_dir": str(run), "trainer_state": meta,
                  "world_size": dist.get_world_size(), "communication": dist.get_backend(),
                  "model_parameters": sum(p.numel() for p in unwrapped.parameters()),
                  "model_sha256": tree_hash(unwrapped.state_dict()),
                  "optimizer_sha256": tree_hash(trainer._modality_optimizer.state_dict()),
                  "scheduler_sha256": tree_hash(accelerator._schedulers[0].state_dict()),
                  "seen_modalities": seen, "seen_codecs_this_phase": sorted(seen_codecs),
                  "rgb_enabled": args.rgb, "elapsed_s": time.monotonic() - started,
                  "heterogeneous": args.heterogeneous, "seen_encoders_this_phase": sorted(seen_encoders),
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        if report["peak_allocated_mib"] > args.max_allocated_mib:
            raise RuntimeError("Exceeded the bounded smoke allocation")
        (output / f"{args.phase}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh empty output directory; earlier audit artifacts are preserved")
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MAX_JOBS="1", NCCL_DEBUG="WARN",
               TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
    env.setdefault("CUDA_HOME", str(Path(sys.executable).resolve().parents[1]))
    for name in ("baseline", "pilot", "resume"):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=1",
                   "--node_rank=0", "--master_addr=127.0.0.1", f"--master_port={port}",
                   "-m", "scripts.validation.touchwam_trainer_smoke", "--phase", name,
                   "--output", str(output), "--max-allocated-mib", str(args.max_allocated_mib)]
        if args.rgb:
            command.append("--rgb")
        if args.heterogeneous:
            command.append("--heterogeneous")
        with (output / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if child.wait(timeout=args.phase_timeout_seconds):
                    raise subprocess.CalledProcessError(child.returncode, command)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
    baseline, resumed = [json.loads((output / f"{name}.json").read_text()) for name in ("baseline", "resume")]
    for field in ("model_sha256", "optimizer_sha256", "scheduler_sha256"):
        if baseline[field] != resumed[field]:
            raise RuntimeError(f"Interrupted/resumed trainer differs from uninterrupted: {field}")
    expected_updates = 8 if args.heterogeneous else 6 if args.rgb else 4
    if len(baseline["seen_modalities"]) != 4 or baseline["trainer_state"]["opt_step"] != expected_updates:
        raise RuntimeError("Missing four-modality coverage or optimizer updates")
    if args.rgb and baseline["seen_codecs_this_phase"] != ["tactile_intensity_v1", "tactile_rgb_v1"]:
        raise RuntimeError("Both intensity and raw RGB sources must reach the production trainer")
    if args.heterogeneous and baseline["seen_encoders_this_phase"] != [0, 1, 2]:
        raise RuntimeError("Sharpa, RGB and matrix observation encoders must all reach the production trainer")
    report = {"status": "passed", "validation_only": True, "world_size": 1, "communication": "nccl",
              "full_model": False, "real_t5_vae": False, "real_datasets": False,
              "production_trainer_loop": True, "gradient_accumulation_steps": 2,
              "exact_model_optimizer_scheduler_resume": True, "phases": ["baseline", "pilot", "resume"],
              "peak_allocated_mib": max(json.loads((output / f"{name}.json").read_text())["peak_allocated_mib"]
                                        for name in ("baseline", "pilot", "resume")),
              "seen_modalities": baseline["seen_modalities"], "rgb_enabled": args.rgb,
              "heterogeneous": args.heterogeneous, "seen_encoders": baseline["seen_encoders_this_phase"],
              "seen_codecs": baseline["seen_codecs_this_phase"], "optimizer_updates": expected_updates}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("baseline", "pilot", "resume"))
    parser.add_argument("--rgb", action="store_true", help="Mix optical RGB and intensity sources with separate heads")
    parser.add_argument("--heterogeneous", action="store_true", help="Exercise tiny explicit-random image and native-matrix observation encoders")
    parser.add_argument("--max-allocated-mib", type=int, default=768)
    parser.add_argument("--phase-timeout-seconds", type=int, default=120)
    arguments = parser.parse_args()
    if arguments.heterogeneous:
        arguments.rgb = True
    if not 32 <= arguments.max_allocated_mib <= 1024:
        parser.error("The smoke allocation must stay between 32 and 1024 MiB")
    phase(arguments) if arguments.phase else run(arguments)
