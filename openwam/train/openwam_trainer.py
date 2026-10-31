"""OpenWAM joint video-action trainer (self-contained).

Call order — core skeleton only:

  __init__():  seed -> build_architecture -> freeze -> init_schedulers
               -> [latent setup] -> log param counts
  train():     build optimizer/dataloader/scheduler -> setup output dir
               -> accelerate prepare -> loop{ compute_loss -> backward/clip/step
               -> reduce metrics -> log step -> maybe save ckpt }
               -> finish_training (final weights/state, retention, close wandb)

Methods below are ordered by call sequence: __init__, train, then the
helpers in the order train() reaches them (sub-helpers follow their caller).

Stateless helpers live in ``openwam.train.utils`` (config / param report / LR /
wandb / metric reduction / debug-CSV / checkpoint mgmt / optimizer groups /
seeding).

Usage:
    trainer = OpenWAMTrainer(cfg, accelerator, dataset)
    trainer.train()
"""

import contextlib
import itertools
import logging
import math
import os

import torch
from omegaconf import DictConfig

from openwam.model.compile_options import eager_compiled_regions
from openwam.train.utils.checkpointing import (
    compute_resume_position,
    find_latest_accel_state,
    load_full_state,
    manage_checkpoints,
    save_config,
    save_full_state,
    save_normalization_stats,
    save_weights,
    validate_checkpoint_retention,
    verify_resume_normalization_stats,
)
from openwam.train.utils import deferred_allgather
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.seeding import per_step_seed, seed_process, wire_sampler_seed
from openwam.train.utils.step_timing import StepTimer
from openwam.train.utils.torch_profile import TorchProfile
from openwam.train.utils.training_utils import (
    align_stop_step,
    LossCsvLog,
    TrainingMetricsLog,
    build_cosine_scheduler,
    cfg_get,
    collate_samples_with_video_uint8,
    init_wandb,
    log_parameter_counts,
    optimizer_steps_through,
    reduce_step_metrics,
    write_debug_loss_row,
)

logger = logging.getLogger(__name__)


def _checkpoint_layers_spec(spec):
    """``training.gradient_checkpointing_layers`` as a plain Python value (ListConfig -> list)."""
    if spec is None or isinstance(spec, (int, str)):
        return spec
    return [int(i) for i in spec]


class OpenWAMTrainer:
    """Joint video-action trainer for OpenWAM. See module docstring for call order.

    Args:
        cfg: Hydra DictConfig with model, training, data, project sections.
        accelerator: HuggingFace Accelerator instance.
        dataset: Training dataset (used for action stats loading).
    """

    # (1) Constructor — seed -> build architecture -> freeze -> init schedulers -> param report.
    def __init__(self, cfg: DictConfig, accelerator=None, dataset=None):
        self.cfg = cfg
        self.dataset = dataset
        self.accelerator = accelerator

        # ---- Reproducible seed (FastWAM-style, yaml-driven) ----
        # Seed before build_architecture so DiT/ActionDiT weight init is
        # deterministic. seed_process uses the same RANK_OFFSET rank stride as
        # per_step_seed and the launcher's seed_everything, so a process's init
        # and per-step seeds share one rank window (cudnn left to the launcher).
        project_cfg = getattr(cfg, "project", None)
        yaml_seed = getattr(project_cfg, "seed", None) if project_cfg is not None else None
        self._rank = int(os.environ.get("RANK", 0))
        self._run_seed = int(yaml_seed) if yaml_seed is not None else None
        if self._run_seed is not None:
            seed_process(self._run_seed, rank=self._rank)
            if self._rank == 0:
                logger.info("Reproducible mode: cfg.project.seed=%d (rank=%d)", self._run_seed, self._rank)

        t = cfg.training
        m = cfg.model

        # Build architecture (creates video_backbone internally from config).
        from openwam.model import build_architecture, resolve_architecture_config

        # Finetune/resume: build from the self-contained checkpoint dir
        # (skeletons from its config.yaml component specs, weights from its
        # safetensors) so model.video_backbone.model_path need not exist on
        # this host. Weights land here, BEFORE the freeze below. The ckpt config
        # is the reconstruction BASE only — on finetune this run's cfg.model is
        # layered on top (see ckpt_model_loader.merge_ckpt_model_cfg).
        finetune_path = cfg_get(t, "finetune_ckpt_path", None) or None
        resume_path = cfg_get(t, "resume_ckpt_path", None) or None
        if finetune_path and resume_path:
            raise ValueError("finetune_ckpt_path and resume_ckpt_path are mutually exclusive; set at most one.")
        if any(cfg_get(t, key, False) for key in (
            "initialize_touch_from_openwam", "initialize_touch_from_legacy", "initialize_touch_registry"
        )) and not finetune_path:
            raise ValueError("TouchWAM initialization requires finetune_ckpt_path; disable initialization for resume")
        action_init = cfg_get(t, "action_backbone_init", "pretrained")
        if action_init not in ("pretrained", "random"):
            raise ValueError("training.action_backbone_init must be pretrained or random")
        if action_init == "random" and not (finetune_path or resume_path):
            raise ValueError("Random action initialization requires training.finetune_ckpt_path")
        self._ckpt_source_dir = finetune_path or resume_path
        if self._ckpt_source_dir is not None:
            from openwam.train.utils.ckpt_model_loader import (
                build_architecture_from_ckpt_dir,
                propagate_component_specs,
                warn_live_model_cfg_ignored,
            )

            # Finetune starts a NEW run: the ckpt config is only the
            # reconstruction base (it alone carries the component/tokenizer
            # specs), and this run's cfg.model overrides it — merged in place, so
            # the architecture built here and the config save_config() writes to
            # the new run dir are the same one. Resume continues ONE run whose
            # config.yaml is already on disk and is reused untouched, so there the
            # ckpt config stays authoritative and divergence is only reported.
            resolved_arch, self.architecture, ckpt_cfg = build_architecture_from_ckpt_dir(
                self._ckpt_source_dir,
                weights_required=finetune_path is not None,
                override_cfg=cfg if finetune_path is not None else None,
                reset_action_backbone=action_init == "random" and finetune_path is not None,
                action_init_seed=cfg_get(t, "action_init_seed", None),
            )
            if finetune_path is not None:
                # The merge replaced the cfg.model node, so `m` (bound above)
                # still points at the pre-merge one — rebind it before `freeze`
                # and the architecture logging below read from it.
                m = cfg.model
            else:
                warn_live_model_cfg_ignored(ckpt_cfg, cfg, self._ckpt_source_dir)
                propagate_component_specs(ckpt_cfg, cfg)
        else:
            resolved_arch = resolve_architecture_config(m)
            self.architecture = build_architecture(resolved_arch.registry_name, resolved_arch.params)
        logger.info(
            "Architecture: %s (framework=%s variant=%s)",
            resolved_arch.registry_name,
            resolved_arch.canonical.framework,
            resolved_arch.canonical.variant,
        )

        # Device placement: skip .to(device) when initialize_model_on_cpu and a real
        # training Accelerator is present — DeepSpeed's prepare() then handles the move.
        _init_on_cpu = bool(t.get("initialize_model_on_cpu", False))
        if not (_init_on_cpu and self.accelerator is not None):
            self.architecture.set_dtype_device(self.architecture.dtype, self.architecture.device)

        # --- Freeze: declared per-architecture in the model yaml (freeze:);
        # freeze_modules silently skips paths absent on a given architecture.
        freeze_list = list(getattr(m, "freeze", []))
        for name in self.architecture.freeze_modules(freeze_list):
            logger.info("Frozen: %s", name)

        # Initialize all schedulers (video + action) inside architecture
        self.architecture.init_training_schedulers(1000)

        # Loss weights from the training config
        self.lambda_video = float(t.lambda_video)
        self.lambda_action = float(t.lambda_action)

        # Push forward-time training flags onto the architecture so prepare_inputs
        # is self-contained.
        self.architecture.set_training_runtime(
            use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
            use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
            max_timestep_boundary=float(t.max_timestep_boundary),
            min_timestep_boundary=float(t.min_timestep_boundary),
            gradient_checkpointing_layers=_checkpoint_layers_spec(t.get("gradient_checkpointing_layers")),
            gradient_checkpointing_layers_video_only=_checkpoint_layers_spec(
                t.get("gradient_checkpointing_layers_video_only")
            ),
        )

        self.model = self  # self-reference some external callers expect
        # Replaced in train() from training.step_timing; disabled stages are no-ops.
        self.step_timer = StepTimer(enabled=False)

        is_main = self.accelerator is None or self.accelerator.is_main_process
        log_parameter_counts(self.architecture, is_main=is_main)

    # (2) Driver — build optimizer/dataloader/scheduler -> setup dir -> accelerate prepare
    #     -> (resume) -> epoch/step loop{compute_loss -> log_step -> save} -> finish_training.
    def train(self, num_epochs: int = None, max_steps: int = None):
        """Scope buffered logs and the DeepSpeed patch to this training run."""
        try:
            return self._train(num_epochs=num_epochs, max_steps=max_steps)
        finally:
            self._flush_logs()
            if getattr(self, "_owns_deferred_allgather", False):
                deferred_allgather.uninstall()
                self._owns_deferred_allgather = False

    def _train(self, num_epochs: int = None, max_steps: int = None):
        """Run the training loop (HuggingFace Accelerate distributed).

        Three entry modes (train.yaml finetune/resume fields): fresh, finetune warm-start
        (weights loaded at construction), or resume (load full state after prepare).
        """
        t = self.cfg.training
        num_epochs = cfg_get(t, "num_epochs", None) if num_epochs is None else num_epochs
        max_steps = getattr(t, "max_steps", None) if max_steps is None else max_steps
        batch_size = t.batch_size
        grad_accum = t.gradient_accumulation_steps
        if batch_size is None or grad_accum is None:
            raise ValueError("training.batch_size and gradient_accumulation_steps cannot be null")
        for name, value in (("num_epochs", num_epochs), ("max_steps", max_steps),
                            ("batch_size", batch_size), ("gradient_accumulation_steps", grad_accum)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"training.{name} must be a positive integer or null for an optional limit")
        if self.accelerator.gradient_accumulation_steps != grad_accum:
            raise ValueError("Accelerator gradient accumulation differs from training configuration")
        keep_last_k = getattr(t, "keep_last_k_ckpts", 3)
        validate_checkpoint_retention(keep_last_k)
        configured_save_steps = getattr(t, "save_steps", None)
        if configured_save_steps is not None and (type(configured_save_steps) is not int or configured_save_steps < 0):
            raise ValueError("training.save_steps must be a nonnegative integer or null")
        stop_after_steps = getattr(t, "stop_after_steps", None)
        if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
            raise ValueError("training.stop_after_steps must be a positive integer or null")

        debug = bool(getattr(t, "debug", False))
        if debug:
            max_steps = 20
            save_steps_override = 10
            logger.info("DEBUG mode: max_steps=20, save@10, constant LR")

        # num_epochs=null means step-only training: max_steps is the sole stop condition.
        if num_epochs is None and not max_steps:
            raise ValueError(
                "training.num_epochs and training.max_steps are both unset; "
                "set num_epochs, or set max_steps for step-only training."
            )

        # finetune/resume validation + architecture construction happen in
        # __init__ (self-contained ckpt-dir path); only resume needs a path here.
        resume_path = cfg_get(t, "resume_ckpt_path", None) or None

        # training.backend=fsdp2 shards the model in place; the optimizer must see the
        # sharded parameters, so this precedes build_optimizer.
        shard_model = getattr(self.accelerator, "shard_model", None)
        if shard_model is not None:
            shard_model(self.architecture)
        optimizer = self.build_optimizer()
        dataloader = self.build_dataloader(batch_size)
        max_grad_norm = float(t.max_grad_norm) if getattr(t, "max_grad_norm", None) else None

        n_proc = self.accelerator.num_processes
        if len(dataloader) == 0:
            raise ValueError("Training dataloader has no batches")
        batches_per_epoch = math.ceil(len(dataloader) / n_proc)
        if stop_after_steps is not None:
            stop_after_steps = align_stop_step(stop_after_steps, batches_per_epoch, grad_accum)
        requested_max_steps = max_steps
        if max_steps:
            max_steps = align_stop_step(max_steps, batches_per_epoch, grad_accum)
            if max_steps != requested_max_steps:
                logger.warning("max_steps=%d ends inside accumulation; completing the update at step %d",
                               requested_max_steps, max_steps)
        steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
        max_opt_steps = optimizer_steps_through(max_steps, batches_per_epoch, grad_accum) if max_steps else None
        if num_epochs is not None:
            # prepare() shards the dataloader ~1/n_proc; fold that in so total_opt_steps is the
            # per-process optimizer steps the loop actually runs (same unit as max_opt_steps).
            total_opt_steps = steps_per_epoch * num_epochs
            if max_opt_steps:
                total_opt_steps = min(total_opt_steps, max_opt_steps)
        else:
            total_opt_steps = max_opt_steps
        scheduler = self.build_lr_scheduler(optimizer, total_opt_steps, debug=debug)

        if debug:
            save_steps = save_steps_override
        else:
            save_steps = configured_save_steps
        save_full_states_for_resume = bool(getattr(t, "save_full_states_for_resume", False))

        # Finetune warm-start weights were already loaded at architecture
        # construction (__init__, self-contained ckpt-dir path). Step stays 0.

        output_path, resume_state_dir = self.setup_output_dir(debug, resume_path)
        optimizer, dataloader, scheduler = self.prepare_accelerate(optimizer, dataloader, scheduler)
        self._batches_per_epoch = len(dataloader)
        if self._batches_per_epoch != batches_per_epoch:
            raise ValueError("Prepared dataloader length differs from the scheduler/accumulation plan")
        self._last_checkpoint_step = None
        # Build the default process group's all_reduce connections now. DeepSpeed reduces
        # gradients on its own cloned world group, so otherwise their first use is the barrier
        # inside the first checkpoint save, and a stray TCP connection (internet scanners reach
        # NCCL's listening ports on the txy nodes) landing in that lazy setup aborts the job.
        self.accelerator.wait_for_everyone()

        if self._run_seed is not None:
            from openwam.dataloader.mixture import MixtureDataset

            if isinstance(self.dataset, MixtureDataset):
                logger.info(
                    "Skipping DataLoader sampler seed wiring for MixtureDataset; "
                    "its index-map shuffle is seeded by the dataset itself."
                )
            else:
                wire_sampler_seed(dataloader, int(self._run_seed), rank=self._rank)

        all_params = [p for group in optimizer.param_groups for p in group["params"]]

        is_main = self.accelerator is None or self.accelerator.is_main_process
        wandb_run = None if (debug or not is_main) else init_wandb(self.cfg)

        from tqdm import tqdm

        if num_epochs is not None:
            total_steps = len(dataloader) * num_epochs
            if max_steps:
                total_steps = min(total_steps, max_steps)
        else:
            total_steps = max_steps

        import time as _time

        opt_step = 0
        global_step = 0
        start_epoch = 0
        skip_first = 0
        checkpoint_pending = False

        # Resume: restore full state AFTER prepare, then map global_step -> (epoch, skip).
        if resume_state_dir is not None:
            global_step, opt_step, start_epoch, skip_first = self.resume_if_configured(
                resume_state_dir, dataloader, grad_accum
            )
            already_done = (stop_after_steps is not None and global_step >= stop_after_steps) or (num_epochs is not None and start_epoch >= num_epochs) or (
                max_steps and global_step >= max_steps
            )
            if already_done:
                logger.info("[resume] global_step=%d already complete; finishing.", global_step)
                # A no-op resume must not rewrite or prune existing checkpoints.
                if wandb_run is not None:
                    wandb_run.finish()
                return
            if skip_first > 0 and self._run_seed is None:
                logger.warning(
                    "[resume] mid-epoch resume (skip_first=%d) without project.seed: DataLoader "
                    "shuffle is non-reproducible, so the resumed epoch's batch order differs from "
                    "the original run — samples may be silently re-fed or skipped. Set project.seed "
                    "for faithful mid-epoch resume.",
                    skip_first,
                )

        self._setup_step_timer(batch_size * n_proc, output_path, is_main)
        timer = self.step_timer
        profile = TorchProfile.from_cfg(self.cfg.training.get("torch_profile"), output_path=output_path,
                                        rank=self.accelerator.process_index)
        # Stage ranges come from StepTimer, so profiling needs training.step_timing.enabled.
        timer.annotate = profile.enabled and timer.enabled

        _step_t0 = _time.monotonic()
        pbar = tqdm(total=total_steps, desc="Training", unit="step", initial=min(global_step, total_steps))

        assert self.accelerator is not None, "OpenWAMTrainer requires an Accelerator"

        # Per-step manual_seed makes timestep + noise sampling in compute_loss
        # reproducible across runs and ZeRO stages: same (rank, step) -> same RNG,
        # different ranks at the same step keep in-batch timestep diversity.
        # Gated on _run_seed so unseeded production runs stay fully stochastic.
        epochs = itertools.count(start_epoch) if num_epochs is None else range(start_epoch, num_epochs)
        for epoch in epochs:
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(epoch)
            # On the resumed epoch, skip the batches already consumed before the checkpoint.
            if epoch == start_epoch and skip_first > 0:
                from accelerate import skip_first_batches

                epoch_iter = skip_first_batches(dataloader, skip_first)
            else:
                epoch_iter = dataloader
            for batch in epoch_iter:
                timer.begin_step()
                self._global_step = global_step
                if self._run_seed is not None:
                    step_seed = per_step_seed(self._run_seed, rank=self._rank, step=global_step)
                    torch.manual_seed(step_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(step_seed)
                # The FLOP-counted step runs train-time compiled regions eagerly, forward and
                # backward alike (checkpoint recompute must match the forward's path).
                eager = eager_compiled_regions() if timer.counting_flops else contextlib.nullcontext()
                with self.accelerator.accumulate(self.architecture), eager:
                    with timer.stage("compute_loss"):
                        losses = self.compute_loss(batch)
                    loss = losses["total"]
                    with timer.stage("backward"):
                        self._backward_and_step(loss)

                    grad_norm = torch.tensor(0.0, device=loss.device)
                    if self.accelerator.sync_gradients:
                        with timer.stage("optimizer_wrapper"):
                            if max_grad_norm is not None:
                                grad_norm_val = self.accelerator.clip_grad_norm_(all_params, max_grad_norm)
                                grad_norm = torch.tensor(float(grad_norm_val), device=loss.device)
                            optimizer.step()
                            if getattr(self, "_modality_optimizer", None) is not None:
                                # Also clear activation after a ZeRO overflow-skipped step.
                                self._modality_optimizer.reset_activation()
                            if scheduler is not None:
                                scheduler.step()
                            optimizer.zero_grad()
                            if not self.accelerator.optimizer_step_was_skipped:
                                opt_step += 1

                global_step += 1

                with timer.stage("metrics_reduce"):
                    metrics = reduce_step_metrics(self.accelerator, losses, grad_norm)
                timer.end_step()
                timer.record_step(global_step, metrics)

                current_lr = optimizer.param_groups[0]["lr"]
                _now = _time.monotonic()
                steps_per_sec = 1.0 / max(_now - _step_t0, 1e-9)
                _step_t0 = _now

                with timer.stage("log_step"):
                    self.log_step(
                        metrics=metrics,
                        global_step=global_step,
                        opt_step=opt_step,
                        epoch=epoch,
                        lr=current_lr,
                        steps_per_sec=steps_per_sec,
                        batch_size=batch_size,
                        pbar=pbar,
                        wandb_run=wandb_run,
                        debug=debug,
                        output_path=output_path,
                    )
                timer.maybe_summarize(global_step)

                # save_steps: write the weights line (+ the resumable full state when
                # save_full_states_for_resume=true), then prune in lockstep.
                # Delay a due save until gradients have been applied. Gradients
                # themselves are not part of Accelerate's resumable state.
                if save_steps and global_step % save_steps == 0:
                    checkpoint_pending = True
                if (save_steps and checkpoint_pending and self.accelerator.sync_gradients):
                    self._flush_logs()
                    deferred_allgather.wait()
                    save_weights(self.accelerator, self.architecture, output_path, global_step, final=False)
                    if save_full_states_for_resume:
                        save_full_state(self.accelerator, output_path, global_step, opt_step, epoch,
                                        batches_per_epoch=len(dataloader), grad_accum=grad_accum)
                    if is_main:
                        manage_checkpoints(output_path, keep_last_k)
                    self._last_checkpoint_step = global_step
                    checkpoint_pending = False

                timer.finish_iteration()
                profile.step()

                if (max_steps and global_step >= max_steps) or (stop_after_steps and global_step >= stop_after_steps):
                    pbar.close()
                    profile.stop()
                    self.finish_training(output_path, global_step, save_steps, keep_last_k, is_main, wandb_run,
                                         opt_step=opt_step, epoch=epoch)
                    return

        pbar.close()
        profile.stop()
        self.finish_training(output_path, global_step, save_steps, keep_last_k, is_main, wandb_run,
                             opt_step=opt_step, epoch=epoch)

    # (3) Called by train() first — AdamW over the per-module (action/video) LR param groups.
    def build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW over the per-module (action/video) LR parameter groups."""
        t = self.cfg.training
        if getattr(self.architecture, "model_version", None) == "touchwam_regions_v1":
            from openwam.train.utils.modality_optimizer import ModalityAwareAdamW, build_modality_groups

            if str(getattr(t, "offload_optimizer_device", "none")) != "none":
                raise ValueError("TouchWAM requires offload_optimizer_device=none to preserve its modality-aware optimizer")
            params = build_modality_groups(
                self.architecture,
                action_lr=float(t.action_lr) if getattr(t, "action_lr", None) else None,
                video_lr=float(t.video_lr) if getattr(t, "video_lr", None) else None,
            )
            self._modality_optimizer = ModalityAwareAdamW(
                params, lr=float(t.learning_rate), weight_decay=float(t.weight_decay),
                betas=tuple(getattr(t, "adam_betas", [0.9, 0.95])),
            )
            return self._modality_optimizer
        params = build_trainable_parameters(
            self,
            action_lr=float(t.action_lr) if getattr(t, "action_lr", None) else None,
            video_lr=float(t.video_lr) if getattr(t, "video_lr", None) else None,
        )
        betas = tuple(getattr(t, "adam_betas", [0.9, 0.95]))
        return torch.optim.AdamW(params, lr=float(t.learning_rate), weight_decay=float(t.weight_decay), betas=betas)

    # (4) Called by train() — build the training DataLoader (seeded generator when reproducible).
    def build_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Build the training DataLoader.

        ``MixtureDataset`` already owns a shuffled virtual index map and
        reshuffles it in ``set_epoch``.  Do not wrap it in PyTorch's
        ``RandomSampler``: on a large mixture that would build a second full-size
        permutation, convert it to Python integers, and duplicate it in every
        rank. Other datasets keep the standard DataLoader shuffle.

        When ``cfg.project.seed`` is set, wires a per-rank ``generator`` and a
        ``worker_init_fn`` so dataset-side randomness is reproducible across
        runs while keeping per-epoch / per-worker variation.
        """
        from openwam.dataloader.mixture import MixtureDataset

        t = self.cfg.training
        shuffle = not isinstance(self.dataset, MixtureDataset)
        kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(t.dataset_num_workers),
            # video_uint8_in_workers: frames stacked to uint8 in the workers, pinned by
            # the DataLoader (bitwise-identical model input).
            collate_fn=collate_samples_with_video_uint8 if getattr(t, "video_uint8_in_workers", False) else list,
            pin_memory=True,
        )
        if not shuffle:
            logger.info(
                "DataLoader shuffle disabled for MixtureDataset; its shuffled index map "
                "is reshuffled by dataset.set_epoch(epoch)."
            )
            accelerator = getattr(self, "accelerator", None)
            world_size = accelerator.num_processes if accelerator is not None else 1
            observation_shapes = getattr(self.dataset, "source_tactile_observation_shapes", [None] * len(self.dataset.source_modalities))
            observation_batching = any(shape is not None for shape in observation_shapes)
            source_graphs = list(zip(self.dataset.source_modalities, self.dataset.source_tactile_codecs, observation_shapes))
            mixed_modalities = len(set(source_graphs)) > 1
            tactile_codecs = {codec for codec in self.dataset.source_tactile_codecs if codec is not None}
            codec_batching = "tactile_rgb_v1" in tactile_codecs or len(tactile_codecs) > 1
            self.dataset.training_layout = {
                "world_size": world_size, "microbatch": batch_size,
                "gradient_accumulation": int(getattr(t, "gradient_accumulation_steps", 1)),
                "batching": ("global_modality_codec_observation_v4" if observation_batching else
                             "global_modality_codec_v3" if codec_batching else
                             "global_modality_v2" if any("tactile" in m for m in self.dataset.source_modalities)
                             else "global_modality_v1") if mixed_modalities else "shuffled_windows",
            }
            if mixed_modalities:
                from openwam.dataloader.modality_sampler import GlobalModalitySampler

                kwargs["sampler"] = GlobalModalitySampler(self.dataset, batch_size, world_size)
                logger.info("Global modality batching: %d samples per global microbatch", batch_size * world_size)
        if self._run_seed is not None:
            from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator

            kwargs["generator"] = make_dataloader_generator(self._run_seed, rank=self._rank)
            kwargs["worker_init_fn"] = dataloader_worker_init_fn
        return torch.utils.data.DataLoader(self.dataset, **kwargs)

    # (5) Called by train() — linear-warmup + cosine schedule, or None (constant LR / debug).
    def build_lr_scheduler(self, optimizer, total_opt_steps: int, debug: bool = False):
        """Linear-warmup + cosine schedule, or None (constant LR / debug)."""
        t = self.cfg.training
        if debug:
            return None
        if getattr(t, "lr_scheduler", None) == "cosine":
            return build_cosine_scheduler(
                optimizer, total_opt_steps=total_opt_steps, cfg=self.cfg, num_processes=self.accelerator.num_processes
            )
        return None

    # (6) Called by train() — locate/create the run dir and resolve the resume state dir.
    def setup_output_dir(self, debug: bool, resume_path: str | None) -> tuple[str, str | None]:
        """Locate/create the run dir and resolve the resume state dir.

        With a usable resume state the run dir is REUSED (assets/config/norm already
        present); otherwise rank-0 creates a fresh timestamped dir and broadcasts it.
        All ranks resolve ``resume_state_dir`` independently (shared FS, deterministic),
        so a missing-state error raises on every rank without deadlocking the broadcast.
        Returns ``(output_path, resume_state_dir)``.
        """
        base_output_path = getattr(self.cfg.training, "output_path", "./models")
        is_main = self.accelerator.is_main_process

        resume_state_dir = find_latest_accel_state(resume_path) if resume_path else None
        if resume_path and resume_state_dir is None:
            raise FileNotFoundError(
                f"resume_ckpt_path={resume_path} has no usable accel_state_step_*; full states "
                f"are only written when training.save_full_states_for_resume=true and save_steps is enabled "
                f"— use finetune_ckpt_path to warm-start from weights instead."
            )
        if resume_state_dir is not None:
            output_path = os.path.dirname(resume_state_dir)
            logger.info("[resume] reusing run dir %s (state=%s)", output_path, os.path.basename(resume_state_dir))
            # Keep the checkpoint-dir deploy artifact; refuse resume when it
            # diverges from the dataset transform (legacy or regenerated stats).
            if self.dataset is not None:
                verify_resume_normalization_stats(output_path, self.dataset)
            return output_path, resume_state_dir

        if is_main:
            from datetime import datetime

            run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if debug:
                run_dir_name += "_debug"
            output_path = os.path.join(base_output_path, run_dir_name)
            os.makedirs(output_path, exist_ok=True)
            # Self-contained deploy: backbones save assets, then config + action stats.
            # BEFORE save_config so config.yaml carries the merged reconstruction specs.
            self.architecture.save_assets_for_deployment(output_path, self.cfg)
            if self._ckpt_source_dir is not None:
                # Self-contained finetune/resume: model_path may be unreachable,
                # so relay the tokenizer files from the source ckpt dir instead.
                from openwam.train.utils.ckpt_model_loader import copy_ckpt_artifacts

                copy_ckpt_artifacts(self._ckpt_source_dir, output_path)
            save_config(output_path, self.cfg)
            provenance = getattr(self.architecture, "initialization_provenance", None)
            if provenance is not None:
                import json

                with open(os.path.join(output_path, "initialization.json"), "w") as handle:
                    json.dump(provenance, handle, indent=2)
                    handle.write("\n")
            if self.dataset is not None:
                save_normalization_stats(output_path, self.dataset)
        else:
            output_path = None
        import torch.distributed as dist

        path_list = [output_path] if is_main else [None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(path_list, src=0)
        output_path = path_list[0]
        logger.info("Checkpoints will be saved to %s", output_path)
        return output_path, None

    # (7) Called by train() — wrap architecture/optimizer/dataloader with the DeepSpeed Accelerator.
    def prepare_accelerate(self, optimizer, dataloader, scheduler):
        """Wrap architecture/optimizer/dataloader with the DeepSpeed Accelerator."""
        prepare_args = [self.architecture, optimizer, dataloader]
        if scheduler is not None:
            prepare_args.append(scheduler)
            self.architecture, optimizer, dataloader, scheduler = self.accelerator.prepare(*prepare_args)
        else:
            self.architecture, optimizer, dataloader = self.accelerator.prepare(*prepare_args)
        # After prepare, self.architecture is the wrapped handle used for the loop's
        # forward/backward. Architecture-level helpers must run on the UNDERLYING
        # module, not the wrapper: DeepSpeedEngine.__getattr__ forwards unknown attrs
        # to the inner module, but DistributedDataParallel does NOT — so calling
        # set_dtype_device/move_frozen_to_device on the wrapper would raise
        # AttributeError under a hypothetical plain-DDP accelerator (world_size>1).
        # unwrap_model returns the inner module for both backends (no-op single-GPU,
        # where Accelerate adds no wrapper). NB: this only covers these setup-time
        # helpers — the training loop's self.architecture.prepare_inputs/compute_loss
        # calls would hit the same non-forwarding wrapper if a non-DeepSpeed multi-GPU
        # path is ever reintroduced.
        arch = self.accelerator.unwrap_model(self.architecture)
        # Propagate device down through architecture; frozen modules (T5/VAE) idempotent move.
        # A sharding backend placed the model before sharding; .to() must not touch it now.
        if not hasattr(self.accelerator, "shard_model"):
            arch.set_dtype_device(arch.dtype, self.accelerator.device)
            arch.move_frozen_to_device(self.accelerator.device)
        self._configure_input_fast_paths(arch)
        self._configure_compute_options(arch)
        cfg = getattr(self, "cfg", None)
        if cfg is not None and bool(cfg.training.get("overlap_param_allgather", False)):
            stage = getattr(self.architecture, "zero_optimization_stage", lambda: None)()
            if stage not in (1, 2):
                raise ValueError("training.overlap_param_allgather needs the DeepSpeed ZeRO-1/2 engine")
            # Only known frozen preprocessing implementations may run while
            # trainable parameter shards are in flight. Unknown/custom encoders
            # use the earlier wait in compute_loss instead.
            self._overlap_frozen_inputs = type(arch).__dict__.get("supports_frozen_input_overlap", False)
            for path in getattr(arch, "preprocess_modules", ()):
                module = arch
                for part in path.split("."):
                    module = getattr(module, part, None)
                if isinstance(module, torch.nn.Module) and any(p.requires_grad for p in module.parameters()):
                    self._overlap_frozen_inputs = False
            deferred_allgather.install()
            self._owns_deferred_allgather = True
        self._configure_touchwam_p1(arch)
        logger.info(
            "architecture wrapped (%s), device=%s",
            type(self.architecture).__name__,
            self.accelerator.device,
        )
        return optimizer, dataloader, scheduler

    def _configure_touchwam_p1(self, arch) -> None:
        """``training.touchwam_p1``: host-sync removals with unchanged training values.

        - ``deferred_modality_sync``: see :class:`DeferredModalitySync`; needs the
          ModalityAwareAdamW and the DeepSpeed engine (backward and step are split).
        - ``deferred_blocking_steps``: steps that keep the blocking layout check (default 20).
        - ``input_validation_steps``: after this many steps, skip TouchWAM's pure input
          checks that read GPU values back (``.any()``/``.item()``/``unique``); null keeps them.
        - ``pinned_tactile_h2d``: collated tactile tensors go to the GPU from pinned memory
          without a stream sync per tensor.
        """
        cfg = getattr(self, "cfg", None)
        p1 = dict(cfg.training.get("touchwam_p1") or {}) if cfg is not None else {}
        unknown = set(p1) - {"deferred_modality_sync", "deferred_blocking_steps", "input_validation_steps",
                             "pinned_tactile_h2d"}
        if unknown:
            raise ValueError(f"Unknown training.touchwam_p1 keys: {sorted(unknown)}")
        self._input_validation_steps = p1.get("input_validation_steps")
        self._deferred_modality = None
        deferred = bool(p1.get("deferred_modality_sync", False))
        if not (deferred or p1.get("pinned_tactile_h2d") or self._input_validation_steps is not None):
            return
        if not hasattr(arch, "configure_training_fastpath"):
            raise ValueError("training.touchwam_p1 requires a TouchWAM regional architecture")
        arch.configure_training_fastpath(pinned_tactile_h2d=bool(p1.get("pinned_tactile_h2d", False)),
                                         encoder_activity_mask=deferred)
        if deferred:
            if getattr(self, "_modality_optimizer", None) is None:
                raise ValueError("deferred_modality_sync requires the TouchWAM modality-aware optimizer")
            if getattr(self.accelerator, "deepspeed_engine_wrapped", None) is None and not hasattr(
                    self.accelerator, "shard_model"):
                raise ValueError("deferred_modality_sync splits backward/step; it needs DeepSpeed or training.backend=fsdp2")
            from openwam.train.utils.modality_optimizer import DeferredModalitySync

            observation = getattr(arch, "tactile_spec", {}).get("observation_encoder") or {}
            count = len(observation["sensors"]) if observation.get("kind") == "ftp1_heterogeneous_v1" else 0
            self._deferred_modality = DeferredModalitySync(
                self.accelerator.device, count, blocking_steps=int(p1.get("deferred_blocking_steps", 20)))
        logger.info("touchwam_p1: %s", p1)

    def _backward_and_step(self, loss) -> None:
        """``accelerator.backward`` for DeepSpeed, with the optimizer activation resolved after
        ``engine.backward`` has queued the backward (DeepSpeedEngineWrapper.backward otherwise)."""
        deferred = getattr(self, "_deferred_modality", None)
        if deferred is None:
            self.accelerator.backward(loss)
            return
        if self.accelerator.deepspeed_engine_wrapped is None:
            # fsdp2: backward does not step; the loop's optimizer.step follows.
            self.accelerator.backward(loss)
            modality, codec, encoder_ids = deferred.resolve()
            self._modality_optimizer.activate(modality, codec, observation_encoder_ids=encoder_ids)
            return
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        sync = self.accelerator.sync_gradients
        engine.set_gradient_accumulation_boundary(is_boundary=sync)
        engine.backward(loss)
        modality, codec, encoder_ids = deferred.resolve()
        self._modality_optimizer.activate(modality, codec, observation_encoder_ids=encoder_ids)
        if sync:
            engine.step()

    def _configure_input_fast_paths(self, arch) -> None:
        """Opt-in frozen-encoder input options; runs after frozen modules reach the device.

        ``training.text_padding`` / ``training.text_dedup`` (UMT5) and
        ``training.vae_cudnn_benchmark`` / ``training.vae_channels_last`` /
        ``training.vae_fused_norm`` (VAE encode). All default to the reference path.
        """
        cfg = getattr(self, "cfg", None)  # tests may build the trainer without a cfg
        if cfg is None:
            return
        t = cfg.training
        vb = getattr(arch, "video_backbone", None)
        padding = str(t.get("text_padding", "max_length") or "max_length")
        dedup = bool(t.get("text_dedup", False))
        if padding != "max_length" or dedup:
            if not hasattr(vb, "set_text_encode_options"):
                raise ValueError("training.text_padding / text_dedup need a Wan video backbone")
            vb.set_text_encode_options(padding=padding, dedup=dedup)
            logger.info("text encode: padding=%s dedup=%s", padding, dedup)
        cudnn_benchmark = bool(t.get("vae_cudnn_benchmark", False))
        channels_last = bool(t.get("vae_channels_last", False))
        fused_norm = bool(t.get("vae_fused_norm", False))
        if cudnn_benchmark or channels_last or fused_norm:
            if not hasattr(vb, "set_vae_encode_options"):
                raise ValueError(
                    "training.vae_cudnn_benchmark / vae_channels_last / vae_fused_norm need a Wan video backbone"
                )
            vb.set_vae_encode_options(cudnn_benchmark=cudnn_benchmark, channels_last=channels_last, fused_norm=fused_norm)
            logger.info(
                "VAE encode: cudnn_benchmark=%s channels_last=%s fused_norm=%s", cudnn_benchmark, channels_last, fused_norm
            )

    def _configure_compute_options(self, arch) -> None:
        """Opt-in DiT compute options: ``training.rope_impl`` (Wan RoPE precision/kernel),
        ``training.frame_time_modulation`` (per-latent-frame AdaLN modulation) and
        ``training.dit_block_fusion`` (compiled elementwise regions per Wan block). Defaults
        keep the reference path."""
        cfg = getattr(self, "cfg", None)
        if cfg is None:
            return
        t = cfg.training
        if bool(t.get("frame_time_modulation", False)):
            vb = getattr(arch, "video_backbone", None)
            if not getattr(arch, "supports_frame_time_modulation", False) or not hasattr(
                vb, "set_frame_time_modulation"
            ):
                raise ValueError(
                    "training.frame_time_modulation needs a Wan video backbone and an architecture that "
                    f"declares supports_frame_time_modulation (got {type(arch).__name__})"
                )
            vb.set_frame_time_modulation(True)
            logger.info("Wan time modulation: per latent frame")
        rope_impl = str(t.get("rope_impl", "fp64") or "fp64")
        if rope_impl != "fp64":
            from openwam.model.video_backbone.wan.models import dit as wan_dit

            wan_dit.set_rope_impl(rope_impl)
            logger.info("Wan RoPE: %s", rope_impl)
        if bool(t.get("dit_block_fusion", False)):
            vb = getattr(arch, "video_backbone", None)
            if not hasattr(vb, "set_block_fusion"):
                raise ValueError(f"training.dit_block_fusion needs a Wan video backbone (got {type(vb).__name__})")
            vb.set_block_fusion(True)
            logger.info("Wan block fusion: compiled elementwise regions")

    def _setup_step_timer(self, samples_per_step: int, output_path: str, is_main: bool) -> None:
        """``training.step_timing``: stage timers, incl. DeepSpeed's backward / step inside
        ``accelerator.backward`` and the text / video encode inside ``prepare_inputs``."""
        self.step_timer = StepTimer.from_cfg(
            self.cfg.training.get("step_timing"),
            samples_per_step=samples_per_step,
            output_path=output_path,
            is_main=is_main,
        )
        if not self.step_timer.enabled:
            return
        engine = self.architecture
        if hasattr(engine, "backward") and hasattr(engine, "step") and hasattr(engine, "module"):
            self.step_timer.wrap(engine, "backward", "ds_backward")
            self.step_timer.wrap(engine, "step", "ds_step")
        vb = getattr(self.accelerator.unwrap_model(self.architecture), "video_backbone", None)
        if vb is not None:
            vb._stage_timer = self.step_timer

    # (8) Called by train() on the resume path — restore full state, map step -> (start_epoch, skip).
    def resume_if_configured(self, resume_state_dir: str, dataloader, grad_accum: int) -> tuple[int, int, int, int]:
        """Load full state (after prepare) and map global_step -> (start_epoch, skip_first_batches).

        Returns (global_step, opt_step, start_epoch, skip).
        """
        is_main = self.accelerator.is_main_process
        if is_main:
            logger.info("[resume] loading Accelerate state from %s", resume_state_dir)
        import json

        # Validate before mutating model/optimizer state. Old partial-window
        # checkpoints cannot faithfully resume because their gradients are absent.
        with open(os.path.join(resume_state_dir, "trainer_state.json")) as handle:
            meta = json.load(handle)
        global_step, opt_step = meta.get("global_step"), meta.get("opt_step")
        if type(opt_step) is not int or opt_step < 0:
            raise ValueError("Resume checkpoint is missing a valid saved opt_step")
        for key, expected in (("batches_per_epoch", len(dataloader)),
                              ("gradient_accumulation_steps", grad_accum)):
            if key in meta and meta[key] != expected:
                raise ValueError(f"Resume checkpoint {key} differs: saved={meta[key]}, live={expected}")
        if meta.get("optimizer_boundary", True) is not True:
            raise ValueError("Resume checkpoint was not saved at an optimizer boundary")
        start_epoch, skip, global_step = compute_resume_position(
            global_step, len(dataloader), grad_accum, require_boundary=True)
        if opt_step > optimizer_steps_through(global_step, len(dataloader), grad_accum):
            raise ValueError("Resume opt_step exceeds the number of consumed update boundaries")
        load_full_state(self.accelerator, resume_state_dir)
        if is_main:
            logger.info(
                "[resume] resumed at global_step=%d opt_step=%d epoch=%d skip_first=%d",
                global_step,
                opt_step,
                start_epoch,
                skip,
            )
        return global_step, opt_step, start_epoch, skip

    # (9) Called each step in train()'s loop — joint video-action loss dict (total/video/action).
    def compute_loss(self, batch) -> dict:
        """Compute joint video-action loss. Returns dict: total/video/action."""
        if not isinstance(batch, list):
            batch = [batch]

        active_modality = None
        deferred = getattr(self, "_deferred_modality", None)
        step = getattr(self, "_global_step", 0)
        limit = getattr(self, "_input_validation_steps", None)
        if limit is not None and step == limit:
            self.accelerator.unwrap_model(self.architecture).configure_training_fastpath(validate_inputs=False)
        if deferred is not None:
            deferred.start_layout(batch, step)
        elif getattr(self, "_modality_optimizer", None) is not None:
            from openwam.train.utils.modality_optimizer import validate_global_modality_layout

            active_modality = validate_global_modality_layout(batch, self.accelerator.device)

        # Reject an external sampler that could deadlock ZeRO-2 before any
        # model backward starts. The normal mixture loader aligns these flags.
        if (active_modality is None and deferred is None and getattr(self, "accelerator", None) is not None
                and self.accelerator.num_processes > 1
                and len(set(getattr(getattr(self, "dataset", None), "source_modalities", []))) > 1):
            import torch.distributed as dist

            from openwam.dataloader.modality_sampler import batch_modality_code

            code = batch_modality_code(batch)
            state = torch.tensor([code], dtype=torch.int64, device=self.accelerator.device)
            states = [torch.empty_like(state) for _ in range(self.accelerator.num_processes)]
            dist.all_gather(states, state)
            codes = [int(s.item()) for s in states]
            if -1 in codes or len(set(codes)) != 1:
                raise ValueError(f"Distributed mixture requires aligned global modalities; got {codes}. Use GlobalModalitySampler.")

        timer = getattr(self, "step_timer", None) or StepTimer(enabled=False)
        if not getattr(self, "_overlap_frozen_inputs", False):
            deferred_allgather.wait()
        with timer.stage("prepare_inputs"), timer.count_flops("encoder"):
            inputs = self.architecture.prepare_inputs(batch)
        has_action = any(sample.get("training_modality", "video_action")
                         in ("video_action", "video_action_tactile") for sample in batch)
        if self.lambda_action > 0 and has_action and inputs.get("actions") is None:
            raise ValueError("lambda_action > 0 but no action in data.")

        # All trainable video/action/tactile parameters must be gathered before use.
        deferred_allgather.wait()
        # fsdp2: the FSDP root's registered compute_loss gathers the root parameters.
        model_loss = getattr(self.accelerator, "training_forward", None) or self.architecture.compute_loss
        with timer.count_flops("model"):
            result = model_loss(
                **inputs,
                lambda_video=self.lambda_video,
                lambda_action=self.lambda_action,
            )
        if deferred is not None and deferred.encoder_count:
            deferred.start_encoders(result["observation_encoder_active_mask"])
        if active_modality is not None:
            from openwam.dataloader.modality_sampler import batch_tactile_codec

            observation = getattr(self.architecture, "tactile_spec", {}).get("observation_encoder") or {}
            encoder_ids = ()
            if observation.get("kind") == "ftp1_heterogeneous_v1":
                from openwam.train.utils.modality_optimizer import global_observation_encoder_ids

                encoder_ids = global_observation_encoder_ids(
                    inputs, len(observation["sensors"]), self.accelerator.device,
                    actual_ids=result.get("observation_encoder_active_ids"),
                )
            self._modality_optimizer.activate(
                active_modality, batch_tactile_codec(batch), observation_encoder_ids=encoder_ids
            )

        losses = {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
        }
        for key in ("tactile", "tactile_contact", "tactile_distance", "tactile_deform", "tactile_intensity", "tactile_rgb"):
            if f"loss_{key}" in result:
                losses[key] = result[f"loss_{key}"]
        for key in ("tactile_labeled_regions", "tactile_labeled_frames"):
            losses[key] = result.get(key, 0.0)
        modes = ("video_only", "video_action", "video_tactile", "video_action_tactile")
        for mode in modes:
            losses["samples_" + mode] = sum(s.get("training_modality", "video_action") == mode for s in batch)
        from openwam.dataloader.modality_sampler import TACTILE_CODEC_CODES

        for codec in TACTILE_CODEC_CODES:
            losses["samples_codec/" + (codec or "none")] = sum(
                (s.get("tactile_codec", "tactile_intensity_v1")
                 if "tactile" in s.get("training_modality", "video_action") else None) == codec for s in batch)
        for name in getattr(getattr(self, "dataset", None), "names", []):
            losses["samples_source/" + name] = sum(s.get("_dataset_name") == name for s in batch)
        return losses

    # (10) Called each step in train()'s loop — progress bar, wandb log, debug loss-history CSV.
    def _flush_logs(self) -> None:
        """Write buffered per-step rows (training.log_flush_every) before checkpoints / exit."""
        for name in ("_loss_csv_log", "_training_metrics_log"):
            log = getattr(self, name, None)
            if log is not None:
                log.flush()

    def log_step(
        self,
        *,
        metrics,
        global_step,
        opt_step,
        epoch,
        lr,
        steps_per_sec,
        batch_size,
        pbar,
        wandb_run,
        debug,
        output_path,
    ) -> None:
        """Update progress bar, log to wandb, and (debug) write the loss-history CSV row."""
        labels = [("action", "loss_action")]
        labels.extend((key, f"loss_{key}") for key in ("tactile", "tactile_contact", "tactile_distance", "tactile_deform", "tactile_intensity", "tactile_rgb")
                      if f"loss_{key}" in metrics)
        loss_total = metrics["loss_total"]
        loss_video = metrics["loss_video"]
        grad_norm = metrics["grad_norm"]

        if pbar is not None:
            postfix = {"loss": f"{loss_total:.4f}", "video": f"{loss_video:.4f}"}
            for name, key in labels:
                postfix[name] = f"{metrics[key]:.4f}"
            postfix["lr"] = f"{lr:.2e}"
            postfix["epoch"] = epoch
            pbar.set_postfix(postfix)
            pbar.update(1)

        if wandb_run is not None:
            num_procs = self.accelerator.num_processes if self.accelerator is not None else 1
            log_dict = {
                "train/loss": loss_total,
                "train/loss_video": loss_video,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "performance/steps_per_sec": steps_per_sec,
                "performance/samples_per_sec": steps_per_sec * batch_size * num_procs,
            }
            for name, key in labels:
                log_dict[f"train/loss_{name}"] = metrics[key]
            for key, value in metrics.items():
                if key.startswith("samples_") or key.startswith("tactile_labeled_"):
                    log_dict["data/" + key] = value
            wandb_run.log(log_dict, step=global_step)

        is_main = self.accelerator is None or self.accelerator.is_main_process
        cfg = getattr(self, "cfg", None)
        flush_every = int(cfg.training.get("log_flush_every", 1) or 1) if cfg is not None else 1
        if is_main and output_path:
            if getattr(self, "_training_metrics_log", None) is None:
                self._training_metrics_log = TrainingMetricsLog(output_path, flush_every=flush_every)
            self._training_metrics_log.add(metrics=metrics, global_step=global_step,
                                          opt_step=opt_step, epoch=epoch, lr=lr, steps_per_sec=steps_per_sec)

        cfg = getattr(self, "cfg", None)
        loss_csv = bool(cfg.training.get("loss_log_csv", False)) if cfg is not None else False
        if not debug:
            # training.loss_log_csv: the per-step CSV without the debug run overrides.
            if loss_csv and is_main and output_path:
                if getattr(self, "_loss_csv_log", None) is None:
                    # Keep one CSV schema when V/VA/intensity/RGB batches alternate.
                    csv_labels = [("action", "loss_action")]
                    if getattr(self.architecture, "model_version", "").startswith("touchwam"):
                        csv_labels.extend((key, "loss_" + key) for key in (
                            "tactile", "tactile_contact", "tactile_distance", "tactile_deform",
                            "tactile_intensity", "tactile_rgb"))
                    self._loss_csv_log = LossCsvLog(
                        output_path, labels=csv_labels, flush_every=flush_every
                    )
                self._loss_csv_log.add(
                    metrics=metrics,
                    global_step=global_step,
                    opt_step=opt_step,
                    epoch=epoch,
                    lr=lr,
                    steps_per_sec=steps_per_sec,
                )
            return
        if not is_main:
            return
        loss_parts = " ".join(f"{name}={metrics[key]:.6f}" for name, key in labels)
        msg = (
            f"[debug][step {global_step:04d} opt {opt_step:04d}] "
            f"loss={loss_total:.6f} video={loss_video:.6f} {loss_parts} "
            f"grad_norm={grad_norm:.6f} lr={lr:.3e} epoch={epoch} "
            f"steps_per_sec={steps_per_sec:.3f}"
        )
        logger.info(msg)
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg, flush=True)

        if output_path:
            write_debug_loss_row(
                output_path,
                labels=labels,
                metrics=metrics,
                global_step=global_step,
                opt_step=opt_step,
                epoch=epoch,
                lr=lr,
                steps_per_sec=steps_per_sec,
            )

    # (11) Called on normal completion — final weights/state, retention, close wandb.
    def finish_training(
        self,
        output_path: str,
        global_step: int,
        save_steps,
        keep_last_k: int,
        is_main: bool,
        wandb_run,
        *,
        opt_step: int,
        epoch: int,
    ) -> None:
        """Save a resumable final update when configured; never discard full states.

        Falsy save_steps disables checkpoint IO, including retention. The final
        full state is kept under the same retention policy as periodic saves.
        """
        self._flush_logs()
        deferred_allgather.wait()
        if save_steps:
            if not self.accelerator.sync_gradients:
                raise RuntimeError("Cannot finish training with unapplied accumulated gradients")
            if self._last_checkpoint_step != global_step:
                save_weights(self.accelerator, self.architecture, output_path, global_step, final=True)
                if cfg_get(self.cfg.training, "save_full_states_for_resume", False):
                    save_full_state(self.accelerator, output_path, global_step, opt_step, epoch,
                                    batches_per_epoch=self._batches_per_epoch,
                                    grad_accum=int(self.cfg.training.gradient_accumulation_steps))
        self.accelerator.wait_for_everyone()
        if is_main and save_steps:
            manage_checkpoints(output_path, keep_last_k)
        if wandb_run is not None:
            wandb_run.finish()
