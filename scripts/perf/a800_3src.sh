#!/usr/bin/env bash
# A800 probe of the three-source recipe (configs/train_h20_3src_*.yaml) on the vePFS copies under
# /gaozt-test1/lyh: same EgoDex/AgiBot subsets + SharpaDex and the same pretrain checkpoint as the H20
# probe, with A800 settings (no allocator cap: real 80 GB; 312 TFLOPS bf16 peak for MFU).
#
#   TAG=<run name> [CONFIG=train_h20_3src_s8b_gc20] [PROFILE=1] \
#   [NNODES=2 NODE_RANK=<0|1> MASTER_ADDR=<node-0 ip>] bash scripts/perf/a800_3src.sh [extra hydra overrides]
#
# Two nodes shard ZeRO-2 state 16-way instead of 64-way: ~3.3 GiB/GPU more than the 64-GPU job
# (8-way costs ~7.7 GiB more), so a 2-node peak near 80 GiB still fits the 64-GPU run.
# Output: $PERF_ROOT/$TAG/<timestamp>/ (step_timing.jsonl, debug_loss_history.csv) and
# $PERF_ROOT/$TAG.node<rank>.log, with the git commit in $PERF_ROOT/$TAG.sha.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${TAG:?set TAG}"
CONFIG=${CONFIG:-train_h20_3src_s8b_gc20}
ROOT=${A800_ROOT:-/gaozt-test1/lyh}
PERF_ROOT=${PERF_ROOT:-$ROOT/perf_runs/a800_3src}
OPENWAM_ENV=${OPENWAM_ENV:-$ROOT/envs/openwam}
DATA=$ROOT/datasets
NODE_RANK=${NODE_RANK:-0}
mkdir -p "$PERF_ROOT"
export PATH="$OPENWAM_ENV/bin:$PATH" VIRTUAL_ENV="$OPENWAM_ENV"
export PYTHONPATH="$PWD" WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NNODES=${NNODES:-1} NODE_RANK MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29500}
LOG="$PERF_ROOT/$TAG.node$NODE_RANK.log"
[ "$NODE_RANK" = 0 ] && { git rev-parse HEAD; git status --porcelain --untracked-files=no; } > "$PERF_ROOT/$TAG.sha" 2>/dev/null
PROFILE_ARGS=()
[ "${PROFILE:-0}" = 1 ] && PROFILE_ARGS=(training.torch_profile.enabled=true)
bash scripts/train.sh --config-name "$CONFIG" \
  dataloader.datasets.egodex.dataset_dir="$DATA/h20_profile_subset/egodex_300h_video_v1_sub" \
  dataloader.datasets.agibot.dataset_dir="$DATA/h20_profile_subset/agibot_pretrain_v1_sub" \
  dataloader.datasets.sharpa.dataset_dir="$DATA/sharpadex_eef_global_v1/buckets" \
  training.finetune_ckpt_path="$ROOT/models/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors" \
  training.max_gpu_memory_gib=null training.step_timing.peak_tflops=312 \
  training.output_path="$PERF_ROOT/$TAG" project.output_dir="$PERF_ROOT/$TAG" \
  "${PROFILE_ARGS[@]}" "$@" > "$LOG" 2>&1
status=$?
echo "exit $status" >> "$LOG"
exit "$status"
