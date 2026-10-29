# TouchWAM 64-GPU batch-12 production launch

## Running job

- Job: `job-tq51vxdotsub` (`touchwam-64gpu-bs12-20261029-r2`).
- Eight nodes, eight NVIDIA A800 80GB GPUs per node; all 64 ranks passed
  the inter-node NCCL and DDP backward probe.
- Frozen training commit: `5499b62a644a922544ef0b8f4809d9620d704fba`.
  Later monitoring/documentation commits do not change the running release.
- Config: `configs/train_touchwam_six_source_64gpu_bs12_20261029.yaml`.
- Per-GPU batch 12, accumulation 1, global batch 768, BF16, ZeRO-2.
- All explicit activation recomputation disabled, including mixed attention.
- Peak LR `5e-5`, cosine decay, 5% warmup, minimum LR ratio 0.01.
- Full OpenWAM Alpha initialization and pretrained FTP tactile encoder.
- One mixture epoch, 138,386 optimizer updates; checkpoints every 2,000 updates,
  including resume state, retaining the latest three.

Setup and evidence on dev-cpu:
`/mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_bs12_20261029_r2`.

Output run:
`/mnt/pfs/v89atp/workspace/outputs/touchwam_six_source_64gpu_bs12_20261029/2026-10-29_23-28-07`.

The initial job `job-681wn34aur36` failed before training because a resolved PFS
path was unavailable before bootstrap. The r2 launcher preserves the public
`/mnt/pfs/v89atp` mount path. Its frozen model/trainer/config and data contracts
were unchanged; the original failure receipt is retained separately.

## Initial production verification

All eight pods were Running, with no node failure reports, and the TensorBoard
API exposed real training metrics through step 150. At step 100, optimizer
updates were also 100 and loss/gradient metrics were finite.

The timing window ending at step 100 measured 2.112 seconds per update
(median of ranks), 2.123 seconds for the slowest rank, and 363.62 samples/second
across 64 GPUs. Peak allocated memory across ranks was 54.08 GiB and peak
reserved memory 58.54 GiB. These are early production measurements, not a
guarantee of later step timing or peak memory. The first full checkpoint has
not yet been verified at this stage.

## Shared TensorBoard

Use the existing dashboard URL: <http://127.0.0.1:16006/#scalars>.

- `touchwam-64gpu-bs12`: training curves; horizontal axis is training step.
- `touchwam-64gpu-bs12-system`: CPU/GPU resource metrics; horizontal axis is
  elapsed monitoring seconds, not training step.
- Five previously visible historical runs remain in the same dashboard.
  Previously hidden test runs remain hidden.

The original curated log directory remains
`/mnt/pfs/v89atp/workspace/setup/training_monitor_20261022/tensorboard_formal`.
Two symlinks point to the new bridge's `train` and `system` event directories;
historical events are unchanged. Selection metadata is in
`dashboard_run_selection.json`, with the previous selection backed up as
`dashboard_run_selection.before_touchwam_20261029.json`.

The current TensorBoard server listens on dev-cpu loopback port 6006, forwarded
to local loopback port 16006. Its active PID, command and bridge command are in
`tensorboard_monitor/services.json` under the r2 setup directory. The older
monitor root's `shared_dashboard.json` points to that service metadata. The
legacy `start_services.py` uses remote port 16006 and also starts an old bridge;
it is not the launcher for the active shared service.

The independent bridge polls every 10 seconds and flushes events within five
seconds; the trainer flushes metric records every 50 steps. Expect training
curves in batches, roughly every two minutes at the early observed speed.
System metrics update more frequently. Refresh the existing browser tab after
changing the visible run set.

For comparisons, select the historical and new runs and use the Step axis.
Shared tags include `loss/video`, `loss/total`, `speed/seconds_per_step` and
`speed/global_samples_per_second`; the Alpha pretraining run also exposes
`train/learning_rate`. Different batch sizes imply different sample counts at
the same step. The new total includes tactile objectives and different data
mixtures, so it is not an equivalent quality score to the previous total.
`loss_by_modality/*` separates the new run's modality-specific curves.

## Read-only status commands

```bash
ssh dev-cpu '/root/aihc-gpu-smoke-20261022/venv/bin/python /mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_bs12_20261029_r2/pipeline/cluster_job.py status'
ssh dev-cpu 'python3 /mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_bs12_20261029_r2/progress.py'
```

Training metrics are written to `training_metrics.jsonl`; aggregate timing is
in `step_timing.jsonl`. Resource samples and per-node logs are in the setup's
`results/` directory. TensorBoard runs on dev-cpu and does not require a GPU or
a restart of the training job.
