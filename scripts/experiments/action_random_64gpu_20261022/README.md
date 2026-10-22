# OpenWAM fresh pretraining: 8 nodes / 64 A800 GPUs

Fresh initialization: OpenWAM Alpha pretrained Wan and proprio encoder, with all
824 ActionDiT state tensors initialized from the constructor using seed 42.
Source checkpoint: `checkpoint_step_154000.safetensors`, SHA256
`180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d`.
Formal training does not resume the previous 32-GPU run or verification runs.

## Recipe and resources

- 8 nodes × 8 A800 80 GB; per-GPU batch 48; gradient accumulation 1;
  global batch 3072. BF16 and DeepSpeed ZeRO stage 2, gradient checkpointing.
- Each node requests 116 CPU cores, 928 GiB RAM and 128 GiB shared-memory limit.
  Total: 64 GPUs, 928 CPU cores, 7424 GiB RAM. The node hosting dev-cpu has
  116 cores / 941 GiB remaining; this uniform request fits all eight nodes.
- 8 DataLoader workers per GPU, 4 compute threads per rank, OpenBLAS 1 thread.
  Usage is sampled every 15 seconds to `results/*_resources_node*.jsonl`.
- One epoch, AdamW LR 1e-4, betas (0.9, 0.95), weight decay 0.01,
  clip norm 1, cosine schedule with 5% warmup; full-state checkpoint every
  100 optimizer updates, retain two checkpoints.
- EgoDex video-only + AgiBot + SharpaDex mixed by the existing modality sampler.
  AgiBot quantile normalization and SharpaDex min-max normalization unchanged.

## Release and launch

Server repository: `/mnt/pfs/v89atp/workspace/OpenWAM`.
Experiment root: `/mnt/pfs/v89atp/workspace/setup/action_random_64gpu_20261022`.
The server Git commit contains the training config and this pipeline under
`scripts/experiments/action_random_64gpu_20261022`. A frozen `release/` and
`results/release_manifest.json` identify every source/runtime file by SHA256.
Debug and formal training use the same frozen source and runtime.

Run on **dev-cpu**, using the existing authenticated AIHC client:

```sh
/root/aihc-gpu-smoke-20261022/venv/bin/python /mnt/pfs/v89atp/workspace/setup/action_random_64gpu_20261022/pipeline/cluster_job.py create verify
/root/aihc-gpu-smoke-20261022/venv/bin/python /mnt/pfs/v89atp/workspace/setup/action_random_64gpu_20261022/pipeline/cluster_job.py status verify
python3 /mnt/pfs/v89atp/workspace/setup/action_random_64gpu_20261022/pipeline/inspect_progress.py
/root/aihc-gpu-smoke-20261022/venv/bin/python /mnt/pfs/v89atp/workspace/setup/action_random_64gpu_20261022/pipeline/cluster_job.py create formal
```

Verification runs 64-rank NCCL all-reduce and DDP, eight actual training updates
at batch 48, saves full state at step 4, and resumes from step 4 to step 8.
Checks include exact retained pretrained weights, identical fresh ActionDiT on
all 64 ranks, all three datasets, finite losses/parameters, synchronized model
updates, restored optimizer state, sample order, and learning rate.
Formal submission is gated on all 64 ranks passing each phase with the current
release manifest. Verification uses constant LR; formal uses the full cosine
schedule and starts from fresh initialization.

Formal outputs: `/mnt/pfs/v89atp/workspace/outputs/action_random_64gpu/<timestamp>`.
The previous stopped experiment and its checkpoints remain under
`outputs/action_random_32gpu` and `setup/action_random_32gpu_20261022`.
