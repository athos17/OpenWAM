# OpenWAM acceleration comparison: 4 nodes / 32 A800, base vs s7d

Two 4-node jobs with the 8-node three-source recipe (`train_action_random_64gpu`):
EgoDex video-only + AgiBot + SharpaDex, fresh OpenWAM Alpha Wan + proprio with a random
ActionDiT (seed 42), per-GPU batch 48, GA 1 (global batch 1536), AdamW 1e-4, ZeRO-2, bf16.

- `base` (`configs/train_accel_cmp_base.yaml`): every acceleration option off.
- `s7d` (`configs/train_accel_cmp_s7d.yaml`): DeepSpeed overlap_comm + buckets, text padding
  to the longest prompt + dedup, MoT split attention, full recompute on layers 0-23 of 30.
  The VAE options (cuDNN benchmark, channels_last) stay off.

Both run 1200 updates on the 8-node run's LR curve (cosine over 23,845 updates, warmup
1192), with step timing every 10 steps and MFU at 312 TFLOPS. Same seed and world size,
so both runs see the same samples at every step (`metrics.jsonl` records them).

Run on dev-cpu:

```sh
PY=/root/aihc-gpu-smoke-20261022/venv/bin/python
R=/mnt/pfs/v89atp/workspace/setup/accel_cmp_32gpu_20261026
$PY $R/pipeline/cluster_job.py create base
$PY $R/pipeline/cluster_job.py create s7d
$PY $R/pipeline/cluster_job.py status base
python3 $R/pipeline/inspect_progress.py
python3 $R/pipeline/compare_runs.py
```

Outputs: `/mnt/pfs/v89atp/workspace/outputs/accel_cmp_32gpu/{base,s7d}/<timestamp>`.
`release/` and `results/release_manifest.json` freeze the source by SHA256.
