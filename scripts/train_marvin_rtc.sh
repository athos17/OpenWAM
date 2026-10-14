#!/usr/bin/env bash
# Identical on both training hosts. Supply machine paths through environment.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${RTC_CKPT_DIR:?Set RTC_CKPT_DIR to a self-contained Marvin checkpoint directory}"
: "${DATASET_DIR:?Set DATASET_DIR to the converted Marvin dataset}"
if [[ -n "${OPENWAM_ENV_PREFIX:-}" ]]; then
    export PATH="${OPENWAM_ENV_PREFIX}/bin:${PATH}"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m torch.distributed.run \
    --nnodes "${NNODES:-1}" --nproc_per_node "${NPROC_PER_NODE:-4}" \
    --node_rank "${NODE_RANK:-0}" --master_addr "${MASTER_ADDR:-127.0.0.1}" \
    --master_port "${MASTER_PORT:-29500}" \
    scripts/train.py dataloader=marvin \
    "dataloader.dataset_dir=${DATASET_DIR}" \
    "training.finetune_ckpt_path=${RTC_CKPT_DIR}" \
    training.rtc.enabled=true training.rtc.max_delay_steps=16 \
    training.rtc.delay_distribution=uniform \
    "training.learning_rate=${LEARNING_RATE:-5e-5}" \
    "training.batch_size=${BATCH_SIZE:-32}" \
    "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS:-4}" \
    "training.max_steps=${MAX_STEPS:-null}" \
    training.save_full_states_for_resume=true \
    "$@"
