#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep the launcher identical on both training hosts; explicit environment
# variables always override these defaults for the known workspace locations.
case "$PWD" in
    /data_all/*)
        DEFAULT_ENV_PREFIX=/data_all/zhangdaoxuan/miniconda3/envs/openwam
        DEFAULT_FOUNDATION_CKPT=/data_all/share/checkpoints/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model
        DEFAULT_DATASET_DIR=/data_all/share/datasets/pick_place_1_openwam_v3
        ;;
    *)
        DEFAULT_ENV_PREFIX=/gaozt-test1/zhanxch/miniconda3/envs/openwam
        DEFAULT_FOUNDATION_CKPT=/gaozt-test1/fyhong/OpenWAM/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model
        DEFAULT_DATASET_DIR=/gaozt-test1/zdx/datasets/pick_place_1_openwam_v3
        ;;
esac
OPENWAM_ENV_PREFIX="${OPENWAM_ENV_PREFIX:-$DEFAULT_ENV_PREFIX}"
FOUNDATION_CKPT="${FOUNDATION_CKPT:-$DEFAULT_FOUNDATION_CKPT}"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"

if [[ ! -x "${OPENWAM_ENV_PREFIX}/bin/python" ]]; then
    echo "OpenWAM Python environment not found: ${OPENWAM_ENV_PREFIX}" >&2
    exit 1
fi
if [[ ! -f "${FOUNDATION_CKPT}/checkpoint_step_154000.safetensors" ]]; then
    echo "Foundation checkpoint not found: ${FOUNDATION_CKPT}" >&2
    exit 1
fi

DATASET_OVERRIDE=("dataloader.dataset_dir=${DATASET_DIR}")
for arg in "$@"; do
    case "${arg}" in
        dataloader.dataset_dir=*|+dataloader.dataset_dir=*|++dataloader.dataset_dir=*)
            DATASET_OVERRIDE=()
            break
            ;;
    esac
done

if (( ${#DATASET_OVERRIDE[@]} )) && [[ ! -d "${DATASET_DIR}" ]]; then
    echo "Marvin dataset not found: ${DATASET_DIR}" >&2
    exit 1
fi

export PATH="${OPENWAM_ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${OPENWAM_ENV_PREFIX}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
MAX_STEPS="${MAX_STEPS:-null}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
ZERO_STAGE="${ZERO_STAGE:-2}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-8}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_FULL_STATES_FOR_RESUME="${SAVE_FULL_STATES_FOR_RESUME:-true}"
KEEP_LAST_K_CKPTS="${KEEP_LAST_K_CKPTS:-10}"

exec bash scripts/train.sh \
    dataloader=marvin \
    "${DATASET_OVERRIDE[@]}" \
    "training.finetune_ckpt_path=${FOUNDATION_CKPT}" \
    "training.batch_size=${BATCH_SIZE}" \
    "training.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
    "training.num_epochs=${NUM_EPOCHS}" \
    "training.max_steps=${MAX_STEPS}" \
    "training.learning_rate=${LEARNING_RATE}" \
    "training.mixed_precision=${MIXED_PRECISION}" \
    "training.zero_stage=${ZERO_STAGE}" \
    "training.use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING}" \
    "training.dataset_num_workers=${DATASET_NUM_WORKERS}" \
    "training.save_steps=${SAVE_STEPS}" \
    "training.save_full_states_for_resume=${SAVE_FULL_STATES_FOR_RESUME}" \
    "training.keep_last_k_ckpts=${KEEP_LAST_K_CKPTS}" \
    "$@"
