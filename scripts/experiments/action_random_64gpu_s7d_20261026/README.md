# OpenWAM 8 nodes / 64 A800: three-source pretraining with s7d acceleration

Same recipe, seed and data order as `action_random_64gpu_20261022` (EgoDex video-only +
AgiBot + SharpaDex; fresh OpenWAM Alpha Wan + proprio, random ActionDiT seed 42; batch 48,
GA 1, global batch 3072; AdamW 1e-4, cosine with 5% warmup over one epoch = 23,845 updates;
ZeRO-2, bf16; weights every 2,000 updates, all kept; 116 CPU / 928 GiB / 128 GiB shm per node).

Added (`configs/train_action_random_64gpu_s7d.yaml`): DeepSpeed overlap_comm + buckets, text
padding to the longest prompt + dedup, MoT split attention, full recompute on layers 0-23 of 30.
VAE options off. Step timing every 50 steps with MFU (312 TFLOPS), per-step loss CSV.
Allocator: expandable segments.

Purpose: speed and training quality against the 2026-10-22 run. Same sample ids per step make
`compare_with_baseline.py` a direct step-by-step comparison.

Run on dev-cpu (verify: NCCL, 8-step smoke with save at step 4, resume 4 -> 8):

```sh
PY=/root/aihc-gpu-smoke-20261022/venv/bin/python
R=/mnt/pfs/v89atp/workspace/setup/action_random_64gpu_s7d_20261026
$PY $R/pipeline/cluster_job.py create verify
$PY $R/pipeline/cluster_job.py status verify
python3 $R/pipeline/inspect_progress.py
$PY $R/pipeline/cluster_job.py create formal
python3 $R/pipeline/compare_with_baseline.py
```

Outputs: `/mnt/pfs/v89atp/workspace/outputs/action_random_64gpu_s7d/<timestamp>`.
