#!/usr/bin/env python3
"""P2a numeric gate: the production trainer on a tiny TouchWAM, DeepSpeed ZeRO-2 vs FSDP2.

One torchrun phase per (backend, precision, world size) trains the same tiny real V/A/T
model on V / VA / VT / VAT batches (global modality batching, per-step seeds, so every
rank sees the same inputs under both backends) and saves the full parameters after every
optimizer step. ``--compare`` then checks, per step:

- FSDP2 parameter updates match DeepSpeed's (relative to the update size);
- streams inactive in a step keep their parameters exactly (ModalityAwareAdamW).

  torchrun --nproc_per_node=8 -m scripts.validation.touchwam_fsdp2_gate --backend fsdp2 --precision fp32 --out D
  python -m scripts.validation.touchwam_fsdp2_gate --compare D/deepspeed_fp32_w8.pt D/fsdp2_fp32_w8.pt
"""
import argparse
import os
from pathlib import Path
from types import MethodType
from unittest.mock import patch


def phase(args):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from torch.distributed.tensor import DTensor

    from openwam.dataloader.mixture import MixtureDataset
    from openwam.tactile.registry import REGION_NAMES
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.seeding import seed_everything
    from scripts.validation.touchwam_multisource_gpu import make_inputs, tiny_model
    import openwam.train.openwam_trainer as trainer_module

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    mixed = "no" if args.precision == "fp32" else "bf16"
    seed_everything(42)

    class Source(torch.utils.data.Dataset):
        action_dim = 80

        def __init__(self, mode):
            self.training_modality = mode
            self.tactile_codec = "tactile_intensity_v1" if "tactile" in mode else None
            self.tactile_observation_shape = None

        def __len__(self):
            return 64 * world

        def __getitem__(self, index):
            result = {"training_modality": self.training_modality}
            if self.tactile_codec is not None:
                result.update(tactile_region_names=[REGION_NAMES[i] for i in (58, 59)], tactile_codec=self.tactile_codec)
            return result

    modes = ["video_only", "video_action", "video_tactile", "video_action_tactile"]
    dataset = MixtureDataset([Source(m) for m in modes], names=modes, seed=42, strict_action_dim=True,
                             source_configs=[{"type": "synthetic_gate"}] * len(modes))
    out = Path(args.out)
    cfg = OmegaConf.create({"training": {
        "num_epochs": None, "max_steps": args.steps * args.grad_accum, "batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum, "dataset_num_workers": 0, "max_grad_norm": args.max_grad_norm,
        "learning_rate": 1e-3, "weight_decay": 0.01, "offload_optimizer_device": "none", "zero_stage": 2,
        "mixed_precision": mixed, "backend": args.backend, "fsdp2": {"replicate": args.replicate, "reshard_after_forward": not args.no_reshard,
                                                                 "forward_prefetch": args.prefetch,
                                                                 "local_optimizer": args.local_opt,
                                                                 "reduce_dtype": args.reduce_dtype},
        "optimizer_fused": args.fused,
        "lr_scheduler": "cosine", "warmup_ratio": 0.25, "lr_min_ratio": 0.01, "save_steps": args.save_steps,
        "save_full_states_for_resume": args.save_steps > 0, "resume_ckpt_path": args.resume_from,
        "keep_last_k_ckpts": -1, "output_path": str(out / f"run_{args.backend}_{args.precision}_w{world}")},
        "project": {"seed": 42, "wandb": None}, "model": {}})
    if args.backend == "deepspeed":
        from accelerate import Accelerator, DeepSpeedPlugin

        plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=args.grad_accum,
                                 gradient_clipping=args.max_grad_norm, offload_optimizer_device="none")
        plugin.deepspeed_config["zero_optimization"].update(
            reduce_bucket_size=1000000, allgather_bucket_size=1000000, ignore_unused_parameters=True)
        accelerator = Accelerator(deepspeed_plugin=plugin, gradient_accumulation_steps=args.grad_accum,
                                  mixed_precision=mixed)
    else:
        import openwam.train.fsdp2.accelerator as fsdp2_accelerator

        if args.backend == "eager":
            # Reference: the FSDP2 backend's loop with nothing sharded (single GPU only).
            class PlainRoot(torch.nn.Module):
                def __init__(self, arch):
                    super().__init__()
                    self.arch = arch

                def compute_loss(self, **kwargs):
                    return self.arch.compute_loss(**kwargs)

                def set_requires_all_reduce(self, value):
                    pass

            fsdp2_accelerator.parallelize_touchwam = lambda arch, mesh, **kw: (PlainRoot(arch), [])
        if args.backend == "eager" or os.environ.get("GATE_ALLOW_SHARD1"):
            fsdp2_accelerator.FSDP2Accelerator.min_shard = 1
        accelerator = fsdp2_accelerator.FSDP2Accelerator(cfg)
        if os.environ.get("GATE_DUMP_GRADS"):
            plain_clip = accelerator.clip_grad_norm_

            def clip(params, max_norm, *a):
                params = list(params)
                if not getattr(clip, "done", False):
                    clip.done = True
                    names = {id(p): n for n, p in model.named_parameters()}
                    grads = {names[id(p)]: (p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad).float().cpu()
                             for p in params if p.grad is not None}
                    if rank == 0:
                        torch.save(grads, os.environ["GATE_DUMP_GRADS"])
                return plain_clip(params, max_norm, *a)

            accelerator.clip_grad_norm_ = clip
        if args.sum_accum:
            # Reproduce OpenWAM's DeepSpeed path, which sums microbatch gradients: DeepSpeed 0.18
            # applies 1/G in an output hook installed by engine.forward, which the trainer bypasses.
            plain_backward = accelerator.backward
            accelerator.backward = lambda loss: plain_backward(loss * args.grad_accum)

    model = tiny_model(torch.device("cpu"), dtype)
    model.set_training_runtime(use_gradient_checkpointing=not args.no_gc, use_gradient_checkpointing_offload=False,
                               max_timestep_boundary=1.0, min_timestep_boundary=0.0)

    def prepare_latents(self, samples):
        inputs = make_inputs(samples[0]["training_modality"], self.device, self.dtype,
                             tactile_codec=samples[0].get("tactile_codec"))
        inputs["use_gradient_checkpointing"] = not args.no_gc
        return inputs

    model.prepare_inputs = MethodType(prepare_latents, model)
    model.save_assets_for_deployment = MethodType(lambda self, path, cfg: None, model)
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg, trainer.dataset, trainer.accelerator = cfg, dataset, accelerator
    trainer.architecture = model
    trainer._rank, trainer._run_seed, trainer._ckpt_source_dir = rank, 42, None
    trainer.lambda_video = trainer.lambda_action = 1.0

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    snapshots, modalities = [], []

    def full_params():
        # Rounded to the training dtype: DeepSpeed's module holds the bf16 copy, FSDP2 the fp32 master.
        arch = accelerator.unwrap_model(trainer.architecture)
        params = dict(arch.named_parameters())
        return {n: (params[n].full_tensor() if isinstance(params[n], DTensor) else params[n]).detach().to(dtype)
                .float().cpu() for n in names}

    real_reduce = trainer_module.reduce_step_metrics
    real_compute = OpenWAMTrainer.compute_loss

    def compute_loss(self, batch):
        modalities.append(batch[0]["training_modality"] if isinstance(batch, list) else batch["training_modality"])
        return real_compute(self, batch)

    def reduce(acc, losses, grad_norm):
        metrics = real_reduce(acc, losses, grad_norm)
        if accelerator.sync_gradients:
            window = sorted(set(modalities))
            modalities.clear()
            snapshots.append({"params": full_params(), "modality": "+".join(window), "loss": metrics["loss_total"],
                              "grad_norm": metrics["grad_norm"], "step": trainer._global_step + 1})
        return metrics

    initial = None

    real_build = OpenWAMTrainer.build_optimizer

    streams = {}

    def build_optimizer(self):
        # Streams read before prepare: DeepSpeed replaces the groups' params with flat partitions.
        nonlocal initial
        initial = full_params()
        opt = real_build(self)
        # Adam's first steps map |g| >> eps to a full +-lr update, so near-zero gradients whose sign
        # depends on reduction order would dominate the comparison; a larger eps keeps the update a
        # smooth function of the gradient (same on both backends).
        for group in opt.param_groups:
            group["eps"] = args.adam_eps
        by_id = {id(p): n for n, p in self.architecture.named_parameters()}
        for group in opt.param_groups:
            for p in group["params"]:
                streams[by_id[id(p)]] = group.get("touchwam_stream", "observation")
        return opt

    class PlannedInterruption(Exception):
        pass

    real_save = trainer_module.save_full_state

    def save_full_state(*values, **kwargs):
        real_save(*values, **kwargs)
        if values[2] == args.interrupt_at:
            raise PlannedInterruption()

    with patch.object(trainer_module, "reduce_step_metrics", reduce), \
            patch.object(trainer_module, "save_full_state", save_full_state), \
            patch.object(OpenWAMTrainer, "compute_loss", compute_loss), \
            patch.object(OpenWAMTrainer, "build_optimizer", build_optimizer):
        try:
            trainer.train()
        except PlannedInterruption:
            print(f"interrupted after the step-{args.interrupt_at} checkpoint", flush=True)
    export = None
    if args.save_steps and rank == 0 and snapshots:
        # The deploy safetensors written at the last step must hold the trained weights (training dtype).
        from safetensors.torch import load_file

        run = Path(args.resume_from) if args.resume_from else next((out / f"run_{args.backend}_{args.precision}_w{world}").iterdir())
        path = run / f"checkpoint_step_{snapshots[-1]['step']}.safetensors"
        saved = load_file(str(path))
        export = {"path": str(path), "missing": sorted(set(names) - set(saved)),
                  "max_abs_diff": max((saved[n].float() - snapshots[-1]["params"][n]).abs().max().item()
                                      for n in names if n in saved)}
        print("export check", export, flush=True)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"initial": initial, "steps": snapshots, "streams": streams, "world": world, "export": export,
                    "backend": args.backend, "precision": args.precision, "grad_accum": args.grad_accum,
                    "peak_mib": torch.cuda.max_memory_allocated() / 2**20},
                   out / f"{args.backend}_{args.precision}_w{world}_g{args.grad_accum}{args.tag_suffix}.pt")
        print(f"saved {len(snapshots)} steps, modalities {[s['modality'] for s in snapshots]}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


ACTIVE = {"video_only": {"video"}, "video_action": {"video", "action", "proprio"},
          "video_tactile": {"video", "tactile", "tactile_intensity"},
          "video_action_tactile": {"video", "action", "proprio", "tactile", "tactile_intensity"}}


def resume_compare(paths):
    import torch

    base, resumed = (torch.load(p, weights_only=False) for p in paths)
    by_step = {s["step"]: s for s in base["steps"]}
    ok = bool(resumed["steps"])
    for s in resumed["steps"]:
        b = by_step[s["step"]]
        worst = max((s["params"][n] - b["params"][n]).abs().max().item() for n in b["params"])
        # Two identical runs already differ by ~1e-7 (nondeterministic GPU kernels).
        ok &= worst <= 1e-6 and abs(s["loss"] - b["loss"]) <= 1e-6 * abs(b["loss"])
        print(f"step {s['step']:2d} {s['modality']:22s} loss {b['loss']:.6f}/{s['loss']:.6f} max|param diff| {worst:.2e}")
    for name, run in (("baseline", base), ("resumed", resumed)):
        export = run.get("export")
        print(f"{name} export: {export}")
        ok &= export is not None and not export["missing"] and export["max_abs_diff"] == 0.0
    print("RESUME GATE PASSED" if ok else "RESUME GATE FAILED")
    return ok


def compare(paths, tol, loss_only=False):
    import torch

    ref, other = (torch.load(p, weights_only=False) for p in paths)
    print(f"reference {paths[0]}  vs  {paths[1]}")
    worst = 0.0
    worst_loss = 0.0
    prev_r, prev_o = ref["initial"], other["initial"]
    for name in prev_r:
        assert torch.equal(prev_r[name], prev_o[name]), f"initial weights differ: {name}"
    ok = True
    for i, (sr, so) in enumerate(zip(ref["steps"], other["steps"])):
        assert sr["modality"] == so["modality"], (i, sr["modality"], so["modality"])
        active = set().union(*(ACTIVE[m] for m in sr["modality"].split("+")))
        step_worst, untouched_bad = 0.0, []
        for name, r in sr["params"].items():
            o = so["params"][name]
            # Relative L2 of the parameter difference to the reference's total update so far.
            update = (r - ref["initial"][name]).norm().item()
            diff = (r - o).norm().item()
            step_worst = max(step_worst, diff / update if update > 0 else diff)
            stream = ref["streams"][name]
            if stream not in active:
                for run, prev, cur in (("ref", prev_r, sr["params"]), ("other", prev_o, so["params"])):
                    if not torch.equal(prev[name], cur[name]):
                        untouched_bad.append(f"{run}:{name}")
        worst = max(worst, step_worst)
        loss_rel = abs(sr["loss"] - so["loss"]) / abs(sr["loss"])
        worst_loss = max(worst_loss, loss_rel)
        flag = "ok" if (loss_rel if loss_only else step_worst) <= tol and not untouched_bad else "FAIL"
        ok &= flag == "ok"
        print(f"step {i + 1:2d} {sr['modality']:22s} loss {sr['loss']:.6f}/{so['loss']:.6f} "
              f"gnorm {sr['grad_norm']:.4f}/{so['grad_norm']:.4f}  max_param |diff|/|update| {step_worst:.2e}  "
              f"inactive-changed {len(untouched_bad)} {flag}")
        if untouched_bad:
            print("   ", untouched_bad[:6])
        prev_r, prev_o = sr["params"], so["params"]
    print(f"worst parameter difference {worst:.2e}, worst relative loss difference {worst_loss:.2e} "
          f"({'loss' if loss_only else 'parameter'} tolerance {tol:g}); peak MiB {ref['peak_mib']:.0f}/{other['peak_mib']:.0f}")
    print("GATE PASSED" if ok else "GATE FAILED")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("deepspeed", "fsdp2", "eager"))
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--replicate", type=int, default=None)
    parser.add_argument("--adam-eps", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--no-gc", action="store_true", help="disable activation recompute")
    parser.add_argument("--prefetch", action="store_true", help="fsdp2: forward_prefetch")
    parser.add_argument("--fused", action="store_true", help="optimizer_fused")
    parser.add_argument("--local-opt", action="store_true", help="fsdp2: local_optimizer")
    parser.add_argument("--reduce-dtype", default=None, help="fsdp2: reduce_dtype (bf16 / fp32)")
    parser.add_argument("--no-reshard", action="store_true", help="fsdp2: reshard_after_forward=false")
    parser.add_argument("--sum-accum", action="store_true", help="fsdp2: sum microbatch gradients like the DeepSpeed path")
    parser.add_argument("--save-steps", type=int, default=0, help="checkpoint every N steps (weights + resumable state)")
    parser.add_argument("--interrupt-at", type=int, default=None, help="stop right after the checkpoint at this step")
    parser.add_argument("--resume-from", default=None, help="run directory to resume from")
    parser.add_argument("--tag-suffix", default="")
    parser.add_argument("--resume-compare", nargs=2, help="baseline.pt resumed.pt: resumed steps must equal the baseline's")
    parser.add_argument("--loss-only", action="store_true",
                        help="compare: judge by relative loss difference (bf16, where parameter rounding dominates)")
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--tol", type=float, default=1e-3)
    a = parser.parse_args()
    if a.compare:
        raise SystemExit(0 if compare(a.compare, a.tol, a.loss_only) else 1)
    if a.resume_compare:
        raise SystemExit(0 if resume_compare(a.resume_compare) else 1)
    phase(a)
