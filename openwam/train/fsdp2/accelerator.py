"""``training.backend=fsdp2``: an Accelerator stand-in that trains with FSDP2 / HSDP.

Implements the slice of ``accelerate.Accelerator`` that ``OpenWAMTrainer`` uses, so the
trainer loop, losses, logging and step timing are shared with the DeepSpeed backend.
Semantics kept identical to the DeepSpeed ZeRO-2 path, so the two runs are comparable
step by step:

- gradients are averaged over all data-parallel ranks (FSDP2's default ``ReduceOp.AVG``),
  so the losses keep their DeepSpeed normalization;
- the DataLoader is sharded by ``accelerate.data_loader.prepare_data_loader`` with the
  arguments ``Accelerator.prepare`` passes (same per-rank samples, one-batch lookahead,
  device placement and ``skip_first_batches`` support);
- the LR scheduler advances ``num_processes`` times per optimizer step, like
  ``AcceleratedScheduler`` (``build_cosine_scheduler`` scales its horizon for that);
- ``backward`` divides the loss by the accumulation steps and gradients are clipped to
  the global norm before ``optimizer.step``.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from openwam.train.fsdp2.parallelize import parallelize_touchwam

logger = logging.getLogger(__name__)

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, None: None, "none": None}


class _SchedulerShim:
    """Advance the wrapped scheduler ``num_processes`` times per call (AcceleratedScheduler)."""

    def __init__(self, scheduler, num_processes: int):
        self.scheduler = scheduler
        self.num_processes = num_processes

    def step(self, *args, **kwargs):
        for _ in range(self.num_processes):
            self.scheduler.step(*args, **kwargs)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


class FSDP2Accelerator:
    deepspeed_engine_wrapped = None
    optimizer_step_was_skipped = False
    mixed_precision = "bf16"
    min_shard = 2

    def __init__(self, cfg):
        t = cfg.training
        f = dict(t.get("fsdp2") or {})
        unknown = set(f) - {"replicate", "param_dtype", "reduce_dtype", "reshard_after_forward"}
        if unknown:
            raise ValueError(f"Unknown training.fsdp2 keys: {sorted(unknown)}")
        if str(t.get("mixed_precision", "bf16")) not in ("bf16", "no"):
            raise ValueError("training.backend=fsdp2 supports mixed_precision bf16 or no")
        if int(t.get("zero_stage", 2)) != 2 or str(t.get("offload_optimizer_device", "none")) != "none":
            raise ValueError("training.backend=fsdp2 replaces ZeRO-2 without optimizer offload")
        if t.get("overlap_param_allgather", False):
            raise ValueError("training.overlap_param_allgather is DeepSpeed-only; disable it for fsdp2")
        self.mixed_precision = str(t.get("mixed_precision", "bf16"))
        default_param = "bf16" if self.mixed_precision == "bf16" else None
        self.param_dtype = _DTYPES[f.get("param_dtype") or default_param]
        self.reduce_dtype = _DTYPES[f.get("reduce_dtype") or ("fp32" if self.param_dtype is not None else None)]
        rafs = f.get("reshard_after_forward")
        self.reshard_after_forward = True if rafs is None else bool(rafs)
        self.gradient_accumulation_steps = int(t.gradient_accumulation_steps)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(self.device)
        if not dist.is_initialized():
            dist.init_process_group("nccl", device_id=self.device, timeout=timedelta(minutes=30))
        self.num_processes = dist.get_world_size()
        self.process_index = dist.get_rank()
        self.is_main_process = self.process_index == 0
        self.is_local_main_process = local_rank == 0
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
        replicate = f.get("replicate")
        replicate = self.num_processes // local_world if replicate in (None, "auto") else int(replicate)
        if replicate < 1 or self.num_processes % replicate:
            raise ValueError(f"training.fsdp2.replicate={replicate} must divide world size {self.num_processes}")
        self.replicate, self.shard = replicate, self.num_processes // replicate
        if self.shard < self.min_shard:
            # On a 1-GPU shard group the tiny-model gate gives a wrong gradient for part of the
            # Wan blocks' ``modulation`` (exact at 2 and 8); unexplained, and never a training layout.
            raise ValueError("training.backend=fsdp2 needs at least 2 GPUs per shard group")
        if replicate > 1:
            self.mesh = init_device_mesh("cuda", (replicate, self.shard), mesh_dim_names=("dp_replicate", "dp_shard"))
        else:
            self.mesh = init_device_mesh("cuda", (self.shard,), mesh_dim_names=("dp_shard",))

        from accelerate.state import GradientState

        self.gradient_state = GradientState()
        self.sync_gradients = True
        self._accum_step = 0
        self.root = None
        self._replicated = []
        self._replicated_checks = 0
        self._schedulers = []
        self._optimizer = None
        self.training_forward = None

    # ---- model ----

    def shard_model(self, arch) -> None:
        """Place ``arch`` on this GPU and shard it; call before building the optimizer."""
        arch.set_dtype_device(arch.dtype, self.device)
        arch.move_frozen_to_device(self.device)
        self.root, self._replicated = parallelize_touchwam(
            arch, self.mesh, param_dtype=self.param_dtype, reduce_dtype=self.reduce_dtype,
            reshard_after_forward=self.reshard_after_forward,
        )
        self.training_forward = self.root.compute_loss
        torch.cuda.empty_cache()
        if self.is_main_process:
            logger.info("FSDP2: %d replicate x %d shard; allocated after sharding %.2f GiB",
                        self.replicate, self.shard, torch.cuda.memory_allocated() / 2**30)

    def unwrap_model(self, model):
        return model

    def prepare(self, *objects):
        from accelerate.data_loader import prepare_data_loader

        if self.root is None:
            raise RuntimeError("FSDP2Accelerator.shard_model must run before prepare")
        out = []
        for obj in objects:
            if isinstance(obj, torch.utils.data.DataLoader):
                # The arguments Accelerator.prepare_data_loader passes under DeepSpeed.
                obj = prepare_data_loader(
                    obj, self.device, num_processes=self.num_processes, process_index=self.process_index,
                    split_batches=False, put_on_device=True, rng_types=["generator"], dispatch_batches=None,
                    even_batches=True, use_seedable_sampler=False, non_blocking=False,
                )
            elif isinstance(obj, torch.optim.lr_scheduler.LRScheduler):
                obj = _SchedulerShim(obj, self.num_processes)
                self._schedulers.append(obj)
            elif isinstance(obj, torch.optim.Optimizer):
                self._optimizer = obj
            out.append(obj)
        return tuple(out) if len(out) > 1 else out[0]

    # ---- checkpoints ----

    # save_weights must call get_state_dict on every rank (it gathers the shards).
    state_dict_needs_all_ranks = True

    def get_state_dict(self, model):
        """Full weights for the deploy safetensors, on rank 0 (empty elsewhere).

        Floating tensors come back in the model dtype, as DeepSpeed's bf16 module holds
        them; the fp32 master values stay in the resumable state.
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if not self.is_main_process:
            return {}
        return {k: v.to(model.dtype) if v.is_floating_point() else v for k, v in state.items()}

    def _training_state(self):
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_state, optim_state = get_state_dict(self.root.arch, self._optimizer)
        return {"model": model_state, "optimizer": optim_state}

    def _materialize_optimizer_state(self) -> None:
        """Create the optimizer's per-parameter state so DCP has a template to load into.

        DCP's own initialization steps on zero gradients, which ModalityAwareAdamW skips for
        groups not marked active; here every group takes one lr=0 step (parameters unchanged).
        """
        opt = self._optimizer
        if opt.state:
            return
        lrs = [group["lr"] for group in opt.param_groups]
        for group in opt.param_groups:
            group["lr"] = 0.0
            if "touchwam_active_in_window" in group:
                group["touchwam_active_in_window"] = True
            for param in group["params"]:
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)
        opt.step()
        for group, lr in zip(opt.param_groups, lrs):
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)

    def save_state(self, output_dir: str) -> None:
        """Resumable state: DCP shards of model + optimizer, scheduler, per-rank RNG states."""
        import random

        import numpy as np
        import torch.distributed.checkpoint as dcp

        dcp.save(self._training_state(), checkpoint_id=os.path.join(output_dir, "dcp"))
        if self.is_main_process:
            torch.save([s.state_dict() for s in self._schedulers], os.path.join(output_dir, "schedulers.pt"))
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state(self.device)}
        torch.save(rng, os.path.join(output_dir, f"random_states_{self.process_index}.pt"))
        self.wait_for_everyone()

    def load_state(self, input_dir: str) -> None:
        import random

        import numpy as np
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import set_state_dict

        self._materialize_optimizer_state()
        state = self._training_state()
        dcp.load(state, checkpoint_id=os.path.join(input_dir, "dcp"))
        set_state_dict(self.root.arch, self._optimizer, model_state_dict=state["model"],
                       optim_state_dict=state["optimizer"])
        for scheduler, saved in zip(self._schedulers, torch.load(os.path.join(input_dir, "schedulers.pt"),
                                                                 weights_only=False)):
            scheduler.load_state_dict(saved)
        rng = torch.load(os.path.join(input_dir, f"random_states_{self.process_index}.pt"), weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], self.device)
        self.wait_for_everyone()

    # ---- step ----

    @contextlib.contextmanager
    def accumulate(self, model=None):
        gs = self.gradient_state
        if gs.end_of_dataloader:
            self._accum_step = 0
            self.sync_gradients = True
        else:
            self._accum_step += 1
            self.sync_gradients = self._accum_step % self.gradient_accumulation_steps == 0
        # Gradients reduce every microbatch, HSDP's replicate all-reduce included: deferring it
        # (TorchTitan's default) keeps one flat partial buffer per unit across microbatches,
        # which torch 2.7 requires to cover the same parameters each time, and an accumulation
        # window can mix modalities (VT then VAT). Only the optimizer step waits for sync.
        yield

    def backward(self, loss) -> None:
        (loss / self.gradient_accumulation_steps).backward()
        if self.sync_gradients and self._replicated:
            self._average_replicated_grads()

    def _average_replicated_grads(self) -> None:
        params = self._replicated
        flat = torch.zeros(2 * len(params), device=self.device, dtype=torch.float32)
        for i, p in enumerate(params):
            if p.grad is not None:
                flat[i] = p.grad.float()
                flat[len(params) + i] = 1
        dist.all_reduce(flat)
        # Global modality batches give every rank the same None pattern; the collective
        # itself is unconditional so a mismatch cannot hang. Checked (a host sync) only
        # over the first steps.
        self._replicated_checks += 1
        if self._replicated_checks <= 20:
            counts = flat[len(params):].tolist()
            if any(c not in (0, self.num_processes) for c in counts):
                raise RuntimeError(f"Replicated scalar parameters got gradients on only some ranks: {counts}")
        for i, p in enumerate(params):
            p.grad = None if p.grad is None else (flat[i] / self.num_processes).to(p.dtype).reshape(p.shape)

    @torch.no_grad()
    def clip_grad_norm_(self, parameters, max_norm, norm_type=2.0):
        """Global gradient norm across shards, then clip (TorchTitan ``clip_grad_norm_``)."""
        by_mesh = defaultdict(list)
        for p in parameters:
            if p.grad is not None:
                by_mesh[p.grad.device_mesh if isinstance(p.grad, DTensor) else None].append(p)
        norms = []
        for params in by_mesh.values():
            norm = torch.nn.utils.get_total_norm([p.grad for p in params], norm_type)
            norms.append(norm.full_tensor() if isinstance(norm, DTensor) else norm)
        if not norms:
            return torch.zeros((), device=self.device)
        if math.isinf(norm_type):
            total = torch.stack(norms).amax()
        else:
            total = sum(n.float() ** norm_type for n in norms) ** (1.0 / norm_type)
        for params in by_mesh.values():
            torch.nn.utils.clip_grads_with_norm_(params, max_norm, total)
        return total

    # ---- collectives ----

    def wait_for_everyone(self) -> None:
        dist.barrier(device_ids=[self.device.index])

    def gather(self, tensor):
        out = torch.empty((self.num_processes * tensor.shape[0], *tensor.shape[1:]),
                          dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(out, tensor.contiguous())
        return out
