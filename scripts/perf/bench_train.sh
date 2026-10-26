#!/usr/bin/env bash
# Short throughput probe for training-acceleration work (SharpaDex Tianji global recipe).
#
#   TAG=<run name> [STEPS=40] [DATALOADER=<dataloader group>] [NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1] \
#     bash scripts/perf/bench_train.sh [extra hydra overrides]
#
# Runs scripts/train.sh with the 2026-10-24 sharpadex_global recipe (bs48/GPU, GA1,
# full gradient checkpointing, color jitter, Alpha warm start), step timing every 10
# steps and the allocator capped at 79 GiB (A800 80 GB). Every override can be
# replaced from the command line. Output: $PERF_ROOT/$TAG/ (step_timing.jsonl) and
# $PERF_ROOT/$TAG.node$NODE_RANK.log.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${TAG:?set TAG}"
STEPS=${STEPS:-40}
PERF_ROOT=${PERF_ROOT:-/data/lyh/perf_runs}
OPENWAM_ENV=${OPENWAM_ENV:-/data/apps/miniforge3/envs/openwam}
DATASET_DIR=${DATASET_DIR:-/data/lyh/datasets/sharpadex_eef_global_v1/buckets}
FINETUNE_CKPT=${FINETUNE_CKPT:-/data/models/OpenWAM-Alpha-Pretrain-Foundation-Model}
mkdir -p "$PERF_ROOT"
export PATH="$OPENWAM_ENV/bin:$PATH" CONDA_PREFIX="$OPENWAM_ENV"
export PYTHONPATH="$PWD" WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NODE_RANK=${NODE_RANK:-0}
LOG="$PERF_ROOT/$TAG.node$NODE_RANK.log"
git rev-parse HEAD > "$PERF_ROOT/$TAG.node$NODE_RANK.sha" 2>/dev/null
# DATALOADER=<group> swaps the dataset (its yaml then supplies dataset_dir / jitter).
if [ -n "${DATALOADER:-}" ]; then
  DATA_ARGS=("dataloader=$DATALOADER")
else
  DATA_ARGS=(dataloader=sharpadex_tianji_global dataloader.dataset_dir="$DATASET_DIR" dataloader.color_jitter.enabled=true)
fi
bash scripts/train.sh \
  "${DATA_ARGS[@]}" \
  training.finetune_ckpt_path="$FINETUNE_CKPT" \
  training.batch_size=48 training.gradient_accumulation_steps=1 training.use_gradient_checkpointing=true \
  training.learning_rate=1e-4 training.rtc.enabled=false \
  training.num_epochs=null training.max_steps="$STEPS" training.save_steps=0 training.dataset_num_workers=8 \
  training.step_timing.enabled=true training.step_timing.summary_every=10 training.max_gpu_memory_gib=79 \
  training.output_path="$PERF_ROOT/$TAG" project.output_dir="$PERF_ROOT/$TAG" project.seed=42 project.wandb=null \
  "$@" > "$LOG" 2>&1
status=$?
echo "exit $status" >> "$LOG"
exit "$status"
