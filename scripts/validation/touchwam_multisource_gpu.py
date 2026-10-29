#!/usr/bin/env python3
"""Bounded real-Wan V/VA/VT/VAT validation, never a production training run.

Single GPU: python -m scripts.validation.touchwam_multisource_gpu --backend single
ZeRO-2: torchrun --standalone --nproc_per_node=1 -m scripts.validation.touchwam_multisource_gpu --backend zero2
Two GPUs: torchrun --standalone --nproc_per_node=2 -m scripts.validation.touchwam_multisource_gpu --backend ddp
One GPU, two test processes: add --shared-device (Gloo, not NCCL acceptance).
"""

import argparse
import copy
import datetime
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from openwam.model import build_architecture
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan21
from openwam.tactile.regions import INTENSITY_CODEC, RGB_CODEC
from openwam.train.utils.modality_optimizer import (
    ModalityAwareAdamW, build_modality_groups, validate_global_modality_layout,
)


def tiny_model(device, dtype, *, rgb_enabled=False, observation_encoder=None, sensor_names=None):
    def init_video(self, cfg):
        dit = WanModel(dim=96, in_dim=4, out_dim=4, ffn_dim=192, text_dim=32, freq_dim=32,
                       eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
                       has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True)
        self.video_backbone = Wan21(SimpleNamespace(dit=dit, scheduler=ActionScheduler()))

    cfg = dict(action_dim=80, state_dim=80, use_proprioception=True, dim=32, ffn_dim=64,
               num_heads=4, attn_head_dim=24, text_dim=32, video_dim=96, bridge_layers=[0, 1],
               attention_mask_mode="mutual", mot_checkpoint_mixed_attn=False, training_mode="joint_finetune",
               tactile=dict(width=16, heads=4, layers=2, bridge_layers=[0, 1], history_layers=1,
                            map_size=32, patch_size=16, history=4, future_frames=8, future_stride=3,
                            codec=INTENSITY_CODEC))
    if rgb_enabled:
        cfg["tactile"].update(rgb_enabled=True, rgb_map_size=128, rgb_patch_size=64)
    if observation_encoder:
        cfg["tactile"].update(observation_encoder=observation_encoder,
                             sensor_names=sensor_names or ["sharpa_real", "optical", "pressure"])
    with patch.object(BaseWAMArchitecture, "_init_video_backbone", init_video):
        model = build_architecture("touchwam_regions", cfg)
    model.set_dtype_device(dtype, device)
    model.init_training_schedulers()
    return model


def make_inputs(modality, device, dtype, *, absent="none", tactile_codec=None):
    # Small spatial video grid and real 80-slot actions; tactile time/grid matches production.
    inputs = dict(input_latents=torch.randn(1, 4, 3, 4, 4, device=device, dtype=dtype),
                  context=torch.randn(1, 4, 32, device=device, dtype=dtype),
                  context_mask=torch.ones(1, 4, dtype=torch.bool, device=device),
                  fuse_vae_embedding_in_latents=True, use_gradient_checkpointing=True)
    inputs["first_frame_latents"] = inputs["input_latents"][:, :, :1].clone()
    if "action" in modality:
        inputs.update(actions=torch.randn(1, 24, 80, device=device, dtype=dtype),
                      proprio=torch.randn(1, 80, device=device, dtype=dtype),
                      action_is_pad=torch.zeros(1, 24, 80, device=device, dtype=torch.bool))
        inputs["action_is_pad"][:, :, 20:] = True
    else:
        inputs["video_only_mask"] = torch.ones(1, device=device, dtype=torch.bool)
    if "tactile" in modality and absent != "layout":
        if tactile_codec not in (None, INTENSITY_CODEC, RGB_CODEC):
            raise ValueError(f"Unsupported smoke tactile codec: {tactile_codec}")
        channels, size = (3, 128) if tactile_codec == RGB_CODEC else (1, 32)
        inputs.update(tactile_maps=torch.rand(1, 4, 2, channels, size, size, device=device, dtype=dtype),
                      tactile_valid=torch.ones(1, 4, 2, device=device, dtype=torch.bool),
                      tactile_region_ids=torch.tensor([[58, 59]], device=device),
                      tactile_layout_present=torch.ones(1, 2, device=device, dtype=torch.bool),
                      future_tactile_maps=torch.rand(1, 8, 2, channels, size, size, device=device, dtype=dtype),
                      future_tactile_valid=torch.ones(1, 8, 2, device=device, dtype=torch.bool))
        if tactile_codec is not None:
            inputs["tactile_codec"] = tactile_codec
        if absent in ("labels", "queries"):
            inputs["future_tactile_valid"].zero_()
        if absent == "queries":
            inputs["tactile_valid"].zero_()
            inputs["tactile_query_mask"] = torch.zeros_like(inputs["future_tactile_valid"])
        if absent == "fewer_regions":
            for key in ("tactile_maps", "tactile_valid", "future_tactile_maps", "future_tactile_valid"):
                inputs[key] = inputs[key][:, :, :1]
            inputs["tactile_region_ids"] = inputs["tactile_region_ids"][:, :1]
            inputs["tactile_layout_present"] = inputs["tactile_layout_present"][:, :1]
    return inputs


def snapshot_groups(optimizer):
    return {g["touchwam_stream"]: [(p.detach().clone(), copy.deepcopy(optimizer.state.get(p, {})))
                                   for p in g["params"]] for g in optimizer.param_groups}


def check_inactive_groups(optimizer, snapshot, active):
    checked = []
    for group in optimizer.param_groups:
        stream = group["touchwam_stream"]
        if stream in active:
            continue
        for p, (old, state) in zip(group["params"], snapshot[stream]):
            if not torch.equal(p, old):
                raise RuntimeError(f"Inactive {stream} weights changed")
            current = optimizer.state.get(p, {})
            if current.keys() != state.keys() or any(
                not torch.equal(current[k], v) if isinstance(v, torch.Tensor) else current[k] != v
                for k, v in state.items()
            ):
                raise RuntimeError(f"Inactive {stream} optimizer state changed")
        checked.append(stream)
    return checked


class LossModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model.compute_loss(**inputs)["loss"]


def run(args):
    torch.set_num_threads(1)
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = 0 if args.shared_device else int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < args.minimum_free_mib * 1024**2:
            raise RuntimeError(f"GPU has only {free / 1024**2:.0f} MiB free; validation will not evict running models")
        torch.cuda.set_per_process_memory_fraction(args.max_allocated_mib * 1024**2 / total, device)
        torch.cuda.reset_peak_memory_stats(device)
    backend = "nccl" if device.type == "cuda" and not args.shared_device else "gloo"
    if args.backend != "single":
        if "RANK" in os.environ:
            dist.init_process_group(backend, timeout=datetime.timedelta(seconds=180))
        else:
            os.environ.setdefault("LOCAL_RANK", "0")
            marker = tempfile.NamedTemporaryFile(delete=False)
            marker.close()
            dist.init_process_group(backend, init_method=f"file://{marker.name}", rank=0, world_size=1,
                                    timeout=datetime.timedelta(seconds=180))
    torch.manual_seed(42)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = tiny_model(device, dtype)
    wrapper = LossModule(model)
    base_optimizer = optimizer = ModalityAwareAdamW(build_modality_groups(model), lr=1e-5)
    if args.backend == "ddp":
        wrapper = nn.parallel.DistributedDataParallel(wrapper, find_unused_parameters=True,
                  device_ids=[local_rank] if device.type == "cuda" else None)
    elif args.backend == "zero2":
        import deepspeed
        if args.shared_device:
            os.environ["LOCAL_RANK"] = "0"
        config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": args.accumulation_steps,
                  "train_batch_size": world * args.accumulation_steps, "bf16": {"enabled": dtype == torch.bfloat16},
                  "zero_optimization": {"stage": 2, "ignore_unused_parameters": True,
                                        "reduce_bucket_size": 1000000, "allgather_bucket_size": 1000000},
                  "steps_per_print": 1000, "wall_clock_breakdown": False, "zero_allow_untested_optimizer": True}
        wrapper, optimizer, _, _ = deepspeed.initialize(model=wrapper, optimizer=optimizer,
                                                        config=config, dist_init_required=False)
    cases = [("video_only", "none"), ("video_action", "none"), ("video_tactile", "labels"),
             ("video_action_tactile", "queries"), ("video_action_tactile", "layout"),
             ("video_tactile", "fewer_regions"), ("video_action_tactile", "none"),
             ("video_only", "none"), ("video_only", "none")]
    if args.aligned_layout:
        cases = [case for case in cases if case[1] != "layout"]
    if args.accumulation_steps == 2:
        cases.insert(1, ("video_action_tactile", "none"))
        cases.append(("video_only", "none"))
    measurements = []
    started = time.time()
    microstep, active, snapshot = 0, set(), None
    for modality, absent in cases:
        print(json.dumps({"rank": rank, "starting_modality": modality, "rank0_absence": absent}), flush=True)
        data = make_inputs(modality, device, dtype, absent=absent if rank == 0 and world > 1 else "none")
        if args.backend == "zero2":
            sample = {"training_modality": modality,
                      "tactile_region_names": list(range(data.get("tactile_region_ids", torch.empty(1, 0)).shape[1]))}
            try:
                validate_global_modality_layout([sample], device)
            except ValueError:
                if absent != "layout" or world == 1:
                    raise
                measurements.append({"modality": modality, "rank0_absence": absent,
                                     "result": "collectively_rejected_before_forward"})
                continue
        if microstep % args.accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            active, snapshot = set(), snapshot_groups(base_optimizer)
        base_optimizer.activate(modality)
        active.update(g["touchwam_stream"] for g in base_optimizer.param_groups if g["touchwam_active_in_window"])
        loss = wrapper.module.model.compute_loss(**data)["loss"] if (
            args.backend == "zero2" and args.trainer_call_style
        ) else wrapper(data)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss: {modality}")
        if args.backend == "zero2":
            wrapper.backward(loss)
            # Optimizer/all-gather updates are never differentiable. Explicitly
            # disable grad here also avoids Gloo's multi-view in-place guard.
            with torch.no_grad():
                wrapper.step()
        else:
            (loss / args.accumulation_steps).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"Nonfinite gradient: {modality}")
            if (microstep + 1) % args.accumulation_steps == 0:
                optimizer.step()
        microstep += 1
        checked = []
        if microstep % args.accumulation_steps == 0:
            checked = check_inactive_groups(base_optimizer, snapshot, active)
            base_optimizer.reset_activation()
        if world > 1:
            checksums = torch.stack([sum((p.detach().double().sum() for p in model.parameters())),
                                     sum((p.detach().double().square().sum() for p in model.parameters()))])
            peer_checksums = [torch.empty_like(checksums) for _ in range(world)]
            dist.all_gather(peer_checksums, checksums)
            if any(not torch.equal(checksums, peer) for peer in peer_checksums):
                raise RuntimeError(f"Parameters diverged across ranks after {modality}")
        measurements.append({"modality": modality, "rank0_absence": absent, "loss": float(loss.detach()),
                             "inactive_weights_and_optimizer_state_unchanged": checked})
    if microstep % args.accumulation_steps:
        raise ValueError("Validation cases must end on an optimizer boundary")
    if args.resume_roundtrip:
        if args.backend != "zero2":
            raise ValueError("Resume roundtrip currently validates the production ZeRO-2 engine only")
        before = snapshot_groups(base_optimizer)
        model_before = {name: p.detach().clone() for name, p in model.named_parameters()}
        checkpoint_root = str(Path(args.output).with_suffix(".checkpoint"))
        wrapper.save_checkpoint(checkpoint_root, tag="validation")
        with torch.no_grad():
            for g in base_optimizer.param_groups:
                for p in g["params"]:
                    p.add_(0.125)
            for p in model.parameters():
                p.add_(0.125)
        loaded, _ = wrapper.load_checkpoint(checkpoint_root, tag="validation", load_optimizer_states=True)
        if not loaded:
            raise RuntimeError("ZeRO-2 checkpoint did not load")
        check_inactive_groups(base_optimizer, before, set())
        if any(not torch.equal(p, model_before[name]) for name, p in model.named_parameters()):
            raise RuntimeError("ZeRO-2 resume did not restore model parameters exactly")
        if any(g["touchwam_active_in_window"] for g in base_optimizer.param_groups):
            raise RuntimeError("Resume restored stale stream activation at an optimizer boundary")
    report = dict(status="passed", validation_only=True, backend=args.backend, communication=backend,
                  shared_single_device=args.shared_device, trainer_compute_loss_call=args.trainer_call_style,
                  synchronized_parameters=True, world_size=world, rank=rank, device=str(device),
                  accumulation_steps=args.accumulation_steps, modality_aware_optimizer=True,
                  checkpoint_roundtrip_verified=args.resume_roundtrip,
                  dtype=str(dtype), torch_version=torch.__version__, elapsed_s=time.time() - started,
                  model_parameters=sum(p.numel() for p in model.parameters()), steps=measurements,
                  peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                  peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_name(out.stem + f".rank{rank}" + out.suffix).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("single", "ddp", "zero2"), default="single")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--shared-device", action="store_true")
    p.add_argument("--trainer-call-style", action="store_true",
                   help="ZeRO-2: exercise the production trainer's direct module.compute_loss call")
    p.add_argument("--aligned-layout", action="store_true",
                   help="Keep a tactile layout on all ranks for T modalities, as required by the production reader")
    p.add_argument("--accumulation-steps", type=int, choices=(1, 2), default=1)
    p.add_argument("--resume-roundtrip", action="store_true")
    p.add_argument("--max-allocated-mib", type=int, default=1024)
    p.add_argument("--minimum-free-mib", type=int, default=3072)
    p.add_argument("--output", default="/tmp/touchwam_gpu_validation.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
