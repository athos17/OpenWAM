#!/usr/bin/env bash
# Single-node H20 probe of the three-source recipe (configs/train_h20_3src_*.yaml).
#
#   TAG=<run name> [CONFIG=train_h20_3src_s7d] [PROFILE=1] bash scripts/perf/h20_3src.sh [extra hydra overrides]
#
# PROFILE=1 turns on training.torch_profile (steps 13-16 on rank 0); analyse with
# scripts/perf/analyze_torch_profile.py <run dir>/torch_profile/rank0.json.gz.
# Output: $PERF_ROOT/$TAG/<timestamp>/ (step_timing.jsonl, debug_loss_history.csv)
# and $PERF_ROOT/$TAG.log, with the git commit in $PERF_ROOT/$TAG.sha.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${TAG:?set TAG}"
CONFIG=${CONFIG:-train_h20_3src_s7d}
PERF_ROOT=${PERF_ROOT:-/data/lyh/perf_runs/h20_3src}
OPENWAM_ENV=${OPENWAM_ENV:-/data/apps/miniforge3/envs/openwam}
mkdir -p "$PERF_ROOT"
export PATH="$OPENWAM_ENV/bin:$PATH" CONDA_PREFIX="$OPENWAM_ENV"
export PYTHONPATH="$PWD" WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$PERF_ROOT/$TAG.log"
{ git rev-parse HEAD; git status --porcelain --untracked-files=no; } > "$PERF_ROOT/$TAG.sha" 2>/dev/null
PROFILE_ARGS=()
[ "${PROFILE:-0}" = 1 ] && PROFILE_ARGS=(training.torch_profile.enabled=true)
bash scripts/train.sh --config-name "$CONFIG" \
  training.output_path="$PERF_ROOT/$TAG" project.output_dir="$PERF_ROOT/$TAG" \
  "${PROFILE_ARGS[@]}" "$@" > "$LOG" 2>&1
status=$?
echo "exit $status" >> "$LOG"
exit "$status"
