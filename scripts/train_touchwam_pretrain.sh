#!/usr/bin/env bash
# Mount the complete data/model publication, activate OpenWAM's environment,
# then explicitly select check, train, or resume. No job submission occurs here.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TOUCHWAM_LAUNCH_DIR="${TOUCHWAM_LAUNCH_DIR:-${TOUCHWAM_OUTPUT_ROOT:-outputs/touchwam_pretrain_v1}/launch}"
mkdir -p "$TOUCHWAM_LAUNCH_DIR"
NODE_ID="${NODE_RANK:-${RANK:-0}}"
OVERRIDES=("$@")
case "$MODE" in
  check|train) ;;
  resume)
    : "${TOUCHWAM_RESUME_DIR:?Set TOUCHWAM_RESUME_DIR to a run with full accelerator states}"
    OVERRIDES+=("training.resume_ckpt_path=$TOUCHWAM_RESUME_DIR"
      "training.finetune_ckpt_path=null" "training.initialize_touch_from_openwam=false"
      "training.initialize_touch_from_legacy=false" "training.initialize_touch_registry=false")
    ;;
  *) echo "Usage: $0 {check|train|resume} [Hydra overrides...]" >&2; exit 2 ;;
esac
PREFLIGHT=("$PYTHON_BIN" scripts/preflight_touchwam.py
  --output "$TOUCHWAM_LAUNCH_DIR/preflight-node${NODE_ID}.json")
if [[ "$MODE" != resume ]]; then
  : "${TOUCHWAM_ALPHA_AUDIT:?Set TOUCHWAM_ALPHA_AUDIT to the passed full OpenWAM tensor audit JSON}"
  PREFLIGHT+=(--alpha-audit "$TOUCHWAM_ALPHA_AUDIT")
fi
"${PREFLIGHT[@]}" "${OVERRIDES[@]}"
if [[ "$MODE" == check ]]; then exit 0; fi
exec bash scripts/train.sh --config-name train_touchwam_pretrain "${OVERRIDES[@]}"
