#!/usr/bin/env bash
# txy_2 (1 x 8 H20): color-cup semantic post-training (configs/train_marvin_color_cup_semantic_accel.yaml)
# with an optional offline UMT5 / Wan-VAE encode cache.
#
#   ENCODE_CACHE=1 bash scripts/experiments/color_cup_semantic_20261029/run_txy2.sh [hydra overrides...]
#
#   ENCODE_CACHE   1 (default): train from the cache under CACHE_ROOT, building it first if it is
#                  missing (8 GPU shards -> finalize -> verify -> bitwise check); 0: encode online.
#   CACHE_KINDS    "text latent" (default) | "text" | "latent"
#   CACHE_ROOT     default /data/dataset/openwam_cache/color_cup_semantic_20261027_156ep_b48
#                  (b48 = the per-GPU batch; the VAE is batch-size sensitive, so a cache built for
#                  one batch is only bit-exact for that batch)
#   BUILD_ONLY=1   build / verify the cache and exit
#   WANDB_API_KEY  read from the environment only; without it pass project.wandb=null
#
# Hydra overrides after the script name go to both the cache builder and train.py, so pass the
# dataset / batch / VAE options there, not only to one of them.
set -euo pipefail
cd "$(dirname "$0")/../../.."

CONFIG=train_marvin_color_cup_semantic_accel
ENCODE_CACHE="${ENCODE_CACHE:-1}"
CACHE_KINDS="${CACHE_KINDS:-text latent}"
CACHE_ROOT="${CACHE_ROOT:-/data/dataset/openwam_cache/color_cup_semantic_20261027_156ep_b48}"
NGPU="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
BATCH="${CACHE_BATCH:-48}"
LOG_DIR="${CACHE_ROOT}/logs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/openwam-triton}"

build() {
  local py=(python scripts/precompute/build_encode_cache.py)
  local common=(--out "$CACHE_ROOT" --kinds $CACHE_KINDS --config-name "$CONFIG" --batch "$BATCH")
  mkdir -p "$LOG_DIR"
  echo "$(date +%T) cache: encode $CACHE_KINDS on $NGPU GPUs -> $CACHE_ROOT"
  local pids=()
  for ((i = 0; i < NGPU; i++)); do
    CUDA_VISIBLE_DEVICES=$i LOCAL_RANK=0 "${py[@]}" encode "${common[@]}" --shard "$i" --num-shards "$NGPU" \
      --workers 12 -- "$@" >"$LOG_DIR/encode-$i.log" 2>&1 &
    pids+=($!)
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  if ((failed)); then
    echo "cache: an encode shard failed; see $LOG_DIR/encode-*.log" >&2
    exit 1
  fi
  "${py[@]}" finalize "${common[@]}" --num-shards "$NGPU" -- "$@" 2>&1 | tee "$LOG_DIR/finalize.log"
  "${py[@]}" verify --out "$CACHE_ROOT" --kinds $CACHE_KINDS 2>&1 | tee "$LOG_DIR/verify.log"
  CUDA_VISIBLE_DEVICES=0 "${py[@]}" check "${common[@]}" -- "$@" 2>&1 | tee "$LOG_DIR/check.log"
  echo "$(date +%T) cache: ready"
}

cache_ready() {
  local k
  for k in $CACHE_KINDS; do
    compgen -G "$CACHE_ROOT/$k/VERIFIED-$(hostname)*.json" >/dev/null || return 1
  done
  compgen -G "$CACHE_ROOT/check-$(hostname)*.json" >/dev/null
}

overrides=("$@")
if [[ "$ENCODE_CACHE" == "1" ]]; then
  cache_ready || build "$@"
  for k in $CACHE_KINDS; do
    overrides+=("dataloader.${k}_cache_dir=$CACHE_ROOT/$k")
  done
  [[ "${BUILD_ONLY:-0}" == "1" ]] && exit 0
elif [[ "$ENCODE_CACHE" != "0" ]]; then
  echo "ENCODE_CACHE must be 0 or 1, got $ENCODE_CACHE" >&2
  exit 2
fi

echo "$(date +%T) train: ENCODE_CACHE=$ENCODE_CACHE ${overrides[*]}"
exec bash scripts/train.sh --config-name "$CONFIG" "${overrides[@]}"
