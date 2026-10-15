#!/usr/bin/env bash
# Subtask (language-following) + training-RTC fine-tuning from the pretrain foundation model.
# Wraps train_marvin.sh; machine paths and caches come from the environment:
#   DATASET_DIR              converted Marvin dataset the annotation was made for (required;
#                            train_marvin.sh would otherwise fall back to its pick_place_1 default)
#   SUBTASK_ANNOTATION       subtasks.parquet from scripts/subtask/vocab.py (required)
#   RTC_MAX_DELAY            RTC max delay steps (default 16; dataloader and training stay in sync)
#   LATENT_CACHE_DIR, BOUNDARY_LATENT_CACHE_DIR, TEXT_CACHE_DIR   optional offline encoder caches
#   HOLDOUT_EPISODES         e.g. "[3,30,57,70,80,85]" (default: none)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET_DIR:?Set DATASET_DIR to the dataset the subtask annotation was made for}"
: "${SUBTASK_ANNOTATION:?Set SUBTASK_ANNOTATION to subtasks.parquet}"
export DATASET_DIR
RTC_MAX_DELAY="${RTC_MAX_DELAY:-16}"
CACHE_ARGS=()
[[ -n "${LATENT_CACHE_DIR:-}" ]] && CACHE_ARGS+=("dataloader.latent_cache_dir=${LATENT_CACHE_DIR}" "dataloader.color_jitter.enabled=false")
[[ -n "${BOUNDARY_LATENT_CACHE_DIR:-}" ]] && CACHE_ARGS+=("dataloader.boundary_latent_cache_dir=${BOUNDARY_LATENT_CACHE_DIR}")
[[ -n "${TEXT_CACHE_DIR:-}" ]] && CACHE_ARGS+=("dataloader.text_cache_dir=${TEXT_CACHE_DIR}")
exec bash scripts/train_marvin.sh \
    "dataloader.subtask.annotation_path=${SUBTASK_ANNOTATION}" \
    "dataloader.subtask.rtc_max_delay_steps=${RTC_MAX_DELAY}" \
    "dataloader.subtask.holdout_episodes=${HOLDOUT_EPISODES:-[]}" \
    training.rtc.enabled=true "training.rtc.max_delay_steps=${RTC_MAX_DELAY}" training.rtc.delay_distribution=uniform \
    "${CACHE_ARGS[@]}" \
    "$@"
