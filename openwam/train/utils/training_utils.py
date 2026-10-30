"""Stateless trainer helpers.

Pure compute / IO with no training state: config access, parameter reporting,
LR scheduling, wandb, cross-rank reduction, debug-CSV writing.
"""

import json
import logging
import math
import os

import torch

logger = logging.getLogger(__name__)


def align_stop_step(max_steps: int, batches_per_epoch: int, grad_accum: int) -> int:
    """Finish the requested microstep budget at the next actual update boundary.

    Epoch tails are update boundaries too, even when shorter than grad_accum.
    """
    if any(type(v) is not int or v < 1 for v in (max_steps, batches_per_epoch, grad_accum)):
        raise ValueError("max_steps, batches_per_epoch and gradient_accumulation_steps must be positive integers")
    offset = max_steps % batches_per_epoch
    if offset == 0 or offset % grad_accum == 0:
        return max_steps
    return max_steps + min(grad_accum - offset % grad_accum, batches_per_epoch - offset)


def optimizer_steps_through(microsteps: int, batches_per_epoch: int, grad_accum: int) -> int:
    """Count update boundaries including the short tail of every complete epoch."""
    epochs, remainder = divmod(microsteps, batches_per_epoch)
    return epochs * math.ceil(batches_per_epoch / grad_accum) + math.ceil(remainder / grad_accum)


def append_training_metrics(output_path, *, metrics, global_step, opt_step, epoch, lr, steps_per_sec):
    """Rank-zero persistent microbatch metrics, independent of debug/W&B."""
    row = {"step": int(global_step), "opt_step": int(opt_step), "epoch": int(epoch),
           "lr": float(lr), "steps_per_sec": float(steps_per_sec), **metrics}
    with open(os.path.join(output_path, "training_metrics.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class TrainingMetricsLog:
    """Buffer the existing per-source/per-modality JSONL without changing its schema."""

    def __init__(self, output_path, flush_every=1):
        self.path = os.path.join(output_path, "training_metrics.jsonl")
        self.flush_every = max(1, int(flush_every))
        self.rows = []

    def add(self, *, metrics, global_step, opt_step, epoch, lr, steps_per_sec):
        row = {"step": int(global_step), "opt_step": int(opt_step), "epoch": int(epoch),
               "lr": float(lr), "steps_per_sec": float(steps_per_sec), **metrics}
        self.rows.append(json.dumps(row, ensure_ascii=False) + "\n")
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self):
        if self.rows:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.writelines(self.rows)
            self.rows.clear()


def cfg_get(cfg, key: str, default=None):
    """Read ``key`` from a dict or attr-style config, with default."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def log_parameter_counts(architecture, *, is_main: bool) -> None:
    """Print per-backbone total/trainable param counts (rank-0 only).

    Uses print() not logger so it survives Hydra's default logging filter;
    the is_main gate keeps it correct regardless of the launcher's stdout
    suppression on non-main ranks.
    """
    if not is_main:
        return

    def _count(module):
        if module is None:
            return 0, 0
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    bb_counts = {name: _count(module) for name, module in architecture.backbones.items()}
    extra_counts = {}
    for name, module in architecture.named_children():
        if name in bb_counts:
            continue
        total, trainable = _count(module)
        if total:
            extra_counts[name] = (total, trainable)
    arch_total = sum(total for total, _ in bb_counts.values()) + sum(total for total, _ in extra_counts.values())
    arch_train = sum(train for _, train in bb_counts.values()) + sum(train for _, train in extra_counts.values())
    print("=" * 60)
    print("Parameter counts")
    for name, (total, trainable) in {**bb_counts, **extra_counts}.items():
        print(f"  {name:<15}: total={total / 1e6:7.1f}M  trainable={trainable / 1e6:7.1f}M")
    print(f"  Architecture  : total={arch_total / 1e6:7.1f}M  trainable={arch_train / 1e6:7.1f}M")
    print("=" * 60, flush=True)


def build_cosine_scheduler(optimizer, *, total_opt_steps: int, cfg, num_processes: int = 1):
    """Linear-warmup + cosine-anneal LR schedule.

    total_opt_steps is in per-process optimizer steps; prepare() wraps the scheduler in
    AcceleratedScheduler which advances it num_processes times per opt step, so scale the
    horizons by num_processes to cancel that out.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    t = cfg.training
    lr = float(t.learning_rate)
    warmup_ratio = float(getattr(t, "warmup_ratio", 0.05))
    lr_min_ratio = float(getattr(t, "lr_min_ratio", 0.01))
    n = max(int(num_processes), 1)
    warmup_steps = int(total_opt_steps * warmup_ratio)
    cosine_steps = max(total_opt_steps - warmup_steps, 1)
    warmup_iters = warmup_steps * n
    warmup_sched = LinearLR(optimizer, start_factor=1.0 / max(warmup_iters, 1), total_iters=warmup_iters)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps * n, eta_min=lr * lr_min_ratio)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_iters])
    logger.info(
        "LR scheduler: cosine | total_opt_steps=%d warmup=%d eta_min=%.2e",
        total_opt_steps,
        warmup_steps,
        lr * lr_min_ratio,
    )
    return scheduler


def init_wandb(cfg):
    """Init a wandb run from cfg.project.wandb. Returns the run or None."""
    wandb_cfg = cfg.project.get("wandb", None)
    if wandb_cfg is None:
        return None
    project = getattr(wandb_cfg, "project", None)
    if not project:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None

    run_name = getattr(wandb_cfg, "run_name", None)
    entity = getattr(wandb_cfg, "entity", None)
    from omegaconf import OmegaConf

    run = wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    logger.info("wandb initialized: %s/%s", project, run.name)
    return run


def reduce_step_metrics(accelerator, losses: dict, grad_norm) -> dict:
    """Reduce loss/grad_norm across ranks (mean); single-process fast path."""
    loss = losses["total"]

    def _f(v):
        return v.item() if isinstance(v, torch.Tensor) else float(v)

    extra_keys = [key for key in ("tactile", "tactile_contact", "tactile_distance", "tactile_deform", "tactile_intensity", "tactile_rgb") if key in losses]
    count_keys = sorted(k for k in losses if k.startswith("samples_") or k.startswith("tactile_labeled_"))
    if accelerator is not None and accelerator.num_processes > 1:
        local = torch.tensor(
            [
                loss.detach().float().item(),
                _f(losses["video"]),
                _f(losses["action"]),
                grad_norm.item(),
                *[_f(losses[key]) for key in extra_keys],
                *[_f(losses[key]) for key in count_keys],
            ],
            device=loss.device,
            dtype=torch.float32,
        ).reshape(1, -1)
        g = accelerator.gather(local).mean(dim=0)
        return {
            "loss_total": g[0].item(),
            "loss_video": g[1].item(),
            "loss_action": g[2].item(),
            "grad_norm": g[3].item(),
            **{f"loss_{key}": g[i + 4].item() for i, key in enumerate(extra_keys)},
            **{key: g[i + 4 + len(extra_keys)].item() * accelerator.num_processes for i, key in enumerate(count_keys)},
        }
    return {
        "loss_total": loss.detach().item(),
        "loss_video": _f(losses["video"]),
        "loss_action": _f(losses["action"]),
        "grad_norm": grad_norm.item(),
        **{f"loss_{key}": _f(losses[key]) for key in extra_keys},
        **{key: _f(losses[key]) for key in count_keys},
    }


def write_debug_loss_row(
    output_path,
    *,
    labels,
    metrics,
    global_step,
    opt_step,
    epoch,
    lr,
    steps_per_sec,
) -> None:
    """Append one row to debug_loss_history.csv (writes the header on first call).

    Loss columns follow ``labels`` = ``[(display_name, metrics_key)]``.
    """
    log = LossCsvLog(output_path, labels=labels, flush_every=1)
    log.add(metrics=metrics, global_step=global_step, opt_step=opt_step, epoch=epoch, lr=lr,
            steps_per_sec=steps_per_sec)


class LossCsvLog:
    """debug_loss_history.csv rows, written ``flush_every`` rows at a time.

    One open/append per flush instead of per step (on a shared filesystem each
    open/stat is a metadata round trip). Call :meth:`flush` before checkpoints and at
    the end so the file covers every finished step; a crash loses at most
    ``flush_every - 1`` rows.
    """

    def __init__(self, output_path, *, labels, flush_every: int = 1):
        self.path = os.path.join(output_path, "debug_loss_history.csv")
        self.labels = labels
        self.flush_every = max(1, int(flush_every))
        self._rows: list = []

    def add(self, *, metrics, global_step, opt_step, epoch, lr, steps_per_sec) -> None:
        label_vals = ",".join(f"{metrics.get(key, 0.0):.10g}" for _, key in self.labels)
        self._rows.append(
            f"{global_step},{opt_step},{epoch},{metrics['loss_total']:.10g},{metrics['loss_video']:.10g},{label_vals},"
            f"{metrics['grad_norm']:.10g},{lr:.10g},{steps_per_sec:.10g}\n"
        )
        if len(self._rows) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        write_header = not os.path.exists(self.path)
        label_cols = ",".join(f"loss_{name}" for name, _ in self.labels)
        with open(self.path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(f"step,opt_step,epoch,loss,loss_video,{label_cols},grad_norm,lr,steps_per_sec\n")
            f.writelines(self._rows)
        self._rows.clear()


def collate_samples_with_video_uint8(samples: list) -> list:
    """DataLoader collate (runs in the workers, ``training.video_uint8_in_workers``).

    Returns the sample list like ``collate_fn=list`` and adds ``video_uint8``, each
    sample's ``video`` frames stacked as a uint8 ``[T, H, W, C]`` tensor; the
    DataLoader's pin-memory thread pins it, so the training process skips the
    PIL -> numpy -> stack -> pin copies. Samples whose frames are not uniform RGB
    uint8 are left without it (the backbone then stacks the PIL frames itself).
    """
    from openwam.model.video_backbone.wan.preprocess import clip_to_uint8

    for sample in samples:
        video = sample.get("video") if isinstance(sample, dict) else None
        if isinstance(video, (list, tuple)):
            stacked = clip_to_uint8(video)
            if stacked is not None:
                sample["video_uint8"] = stacked
    return list(samples)
