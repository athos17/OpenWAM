# TouchWAM fresh 64-GPU batch-48 launch — 2026-10-30

Job `job-v19j6o92tot0` (r2) was submitted at 11:19:08 CST with frozen commit
`236b9cc501924122994bc73287a75350ff5f7cb3` from
`codex/touchwam-five-source-ready`.

This is a fresh experiment initialized from the complete OpenWAM Alpha
checkpoint and pinned pretrained FTP tactile encoders. It does not inherit
the batch-12 run's trained weights, Adam history, scheduler or data position.
The new tactile parameters use the configured initialization and seed 42.

## Configuration

- Eight nodes, eight A800 80GB GPUs per node; BF16 and ZeRO-2.
- Microbatch 48 per GPU, accumulation one, global batch 3,072.
- Eight DataLoader workers per rank, default prefetch factor two.
- S9 activation recomputation: layers 0–19, no separate video-only override,
  mixed-attention recomputation enabled, CPU offload disabled.
- Peak LR `1e-4`, Adam betas `(0.9, 0.95)`, weight decay `0.01`, grad clip `1.0`.
- Linear warmup 5%, cosine schedule, minimum LR ratio `0.01`.
- One full mixture epoch: 34,599 updates, including per-modality tail padding,
  and 1,729 warmup updates, both confirmed by the running trainer.
- Full checkpoints every 500 updates, retaining the latest three.
- The first 1,000 steps additionally write per-rank stage traces, including
  data wait. Buffered local traces add no new distributed collective and
  preserve slow steps that window medians omit.

The six source families and all 14 enabled sub-sources are unchanged. The
resolved data configuration and per-source window counts were compared with
the preceding production preflight. No data was reconverted.

## Verified preparation

- 26 relevant tests passed locally and in the actual dev-cpu environment.
- The full bounded-memory Alpha audit checked all 2,089 original tensors and
  proved byte-preserving assignment under the final model configuration.
- Production preflight passed for all 14 enabled sources, including real
  sample decoding, normalization, tactile collation and initialization checks.
- The old batch-12 job `job-399y3w4cxhmg` was stopped only after the complete
  update-2,000 checkpoint was published: deploy weights, 64 optimizer shards,
  64 RNG files and the final completion marker were verified.
- Scheduler status confirmed all old pods stopped at 10:45:59 CST. The old
  checkpoint and TensorBoard history remain available for comparison.
- The initial attempt, `job-tgfbooot9a1z` (r1), passed all 64 NCCL/DDP probe
  ranks and loaded pretrained weights. Its saved runtime configuration confirms
  batch 48, eight workers, S9 recomputation, LR `1e-4` and no resume checkpoint.
  It completed one update with finite loss before a missing EgoDex video caused
  DataLoader failures. All pods had exited by 11:11:46 CST.
- An audit of all 5,208 EgoDex videos found one missing file and no size
  mismatches. The video existed during dataset construction, but its published
  path was subsequently absent; the responsible process has not been identified.
  The surviving original publication source was hardlinked back after checking
  size, hashing it and decoding boundaries of all five selected episodes. The
  complete file audit then passed, with the frozen episode index unchanged.
- R2 uses the identical frozen release. All 14 sources passed a new preflight;
  model/training/data configuration and source window counts match r1, apart
  from output directories. The full Alpha audit is reused after verifying the
  checkpoint stat/header and resolved model configuration still match.

## Paths and monitoring

On dev-cpu, under `/mnt/pfs/v89atp/workspace/`:

- Setup: `setup/touchwam_64gpu_bs48_20261030_r2`.
- Frozen code: setup's `release/`; execution settings:
  `results/launch_config_resolved.json`.
- Preflight: `results/preflight.json`; full tensor audit:
  `results/alpha_bitwise_audit.json`.
- Output alias: `outputs/touchwam_six_source_64gpu_bs48_20261030`, a symlink
  to this setup's `training_output/`.
- Logs/resources: `results/full_node*.log`, `results/full_resources_node*.jsonl`.
- Per-rank startup traces: run directory's `step_trace_rank*.jsonl`.
- Repair evidence: `results/egodex_restore.json`; comparison:
  `results/config_comparison.json`; relaunch history: `results/relaunch.json`.

The existing dashboard remains <http://127.0.0.1:16006/#scalars>.
Runs are `touchwam-64gpu-bs48-r2` and `touchwam-64gpu-bs48-r2-system`.
Historical runs remain visible; the stopped batch-12 monitor was terminated
and its terminal state recorded without restarting the shared dashboard.
The failed r1 monitor was also stopped, with its failure marked in TensorBoard.

```bash
ssh dev-cpu 'python /mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_bs48_20261030_r2/progress.py'
ssh dev-cpu '/root/aihc-gpu-smoke-20261022/venv/bin/python /mnt/pfs/v89atp/workspace/setup/touchwam_64gpu_bs48_20261030_r2/pipeline/cluster_job.py status'
```

Submission and CPU preflight do not establish GPU throughput or long-run
memory stability. These must be measured from the new job after compilation
and initial data prefetch have settled.
