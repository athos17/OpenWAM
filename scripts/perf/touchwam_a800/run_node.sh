#!/usr/bin/env bash
# One node of a TouchWAM P0/P1 run on a800_1/a800_2 (see README.md in this directory).
#
#   TAG=<run name> CONFIG=<train_touchwam_a800_*|nccl_bench> NODE_RANK=<0|1> \
#   [NNODES=2] [PROFILE=1] bash scripts/perf/touchwam_a800/run_node.sh [extra hydra overrides]
#
# Output: $PERF_ROOT/$TAG/<timestamp>/ (step_timing.jsonl, step_trace_rank*.jsonl,
# debug_loss_history.csv), node logs $PERF_ROOT/$TAG.node<rank>.log, commit in $PERF_ROOT/$TAG.sha.
set -uo pipefail
cd "$(dirname "$0")/../../.."
: "${TAG:?set TAG}" "${CONFIG:?set CONFIG}"
ROOT=${A800_ROOT:-/gaozt-test1/lyh}
PERF_ROOT=${PERF_ROOT:-$ROOT/perf_runs/touchwam_a800}
# Shared conda env on vePFS (torch 2.7.1+cu128, deepspeed 0.18.9, accelerate 1.14.0: the same versions
# as bdy's production env), used read-only; PYTHONPATH puts this checkout first.
OPENWAM_ENV=${OPENWAM_ENV:-/gaozt-test1/zhanxch/miniconda3/envs/openwam}
NODE_RANK=${NODE_RANK:-0}
mkdir -p "$PERF_ROOT"
export PATH="$OPENWAM_ENV/bin:$PATH" CONDA_PREFIX="$OPENWAM_ENV"
export PYTHONPATH="$PWD" WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOUCHWAM_SUBSET_ROOT=${TOUCHWAM_SUBSET_ROOT:-$ROOT/datasets/touchwam_subset_v1}
export TOUCHWAM_OPENWAM_CKPT=${TOUCHWAM_OPENWAM_CKPT:-$ROOT/models/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors}
export TOUCHWAM_FTP1_ENCODER_ROOT=${TOUCHWAM_FTP1_ENCODER_ROOT:-$ROOT/models/FTP-1-Tactile-Encoder/d6e5b73e473d3e70fb5f53132a2b1c35b5031156}
export NNODES=${NNODES:-2} NODE_RANK MASTER_ADDR=${MASTER_ADDR:-192.168.0.235} MASTER_PORT=${MASTER_PORT:-29500}
# a800_1/a800_2 reach each other only over eth1 (a800_1: 192.168.0.235). Bootstrap sockets only:
# NCCL still finds the RoCE NICs. gloo (training.touchwam_p1.deferred_modality_sync) uses eth1 too.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth1} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth1}
LOG="$PERF_ROOT/$TAG.node$NODE_RANK.log"
[ "$NODE_RANK" = 0 ] && { echo "$CONFIG $*"; git rev-parse HEAD; git status --porcelain --untracked-files=no; } \
  > "$PERF_ROOT/$TAG.sha" 2>/dev/null
PROFILE_ARGS=()
# Eight consecutive steps (several modalities: 23-30) after 20 waited + 2 warmup steps, ranks 0 and 8 (one per node).
[ "${PROFILE:-0}" = 1 ] && PROFILE_ARGS=(training.torch_profile.enabled=true training.torch_profile.wait=20
  training.torch_profile.warmup=2 training.torch_profile.active=8 "training.torch_profile.ranks=[0,8]")
if [ "$CONFIG" = nccl_bench ]; then
  export PERF_ROOT TAG
  torchrun --nnodes "$NNODES" --nproc_per_node 8 --node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" scripts/perf/touchwam_a800/nccl_bench.py > "$LOG" 2>&1
else
  bash scripts/train.sh --config-name "$CONFIG" \
    training.output_path="$PERF_ROOT/$TAG" project.output_dir="$PERF_ROOT/$TAG" \
    "${PROFILE_ARGS[@]}" "$@" > "$LOG" 2>&1
fi
status=$?
echo "exit $status" >> "$LOG"
exit "$status"
