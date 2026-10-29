#!/usr/bin/env bash
# Frozen 2026-10-29 recipe; check remains the default and never starts training.
set -euo pipefail
export TOUCHWAM_CONFIG=train_touchwam_six_source_20261029
export TOUCHWAM_ALPHA_AUDIT="${TOUCHWAM_ALPHA_AUDIT:-/mnt/pfs/v89atp/workspace/setup/touchwam_six_source_ready_20261029/alpha_bitwise_audit.json}"
exec bash "$(dirname "$0")/train_touchwam_pretrain.sh" "$@"
