#!/usr/bin/env bash
# Frozen 2026-10-29 recipe; check remains the default and never starts training.
set -euo pipefail
export TOUCHWAM_CONFIG="${TOUCHWAM_CONFIG:-train_touchwam_six_source_bs8_norecompute_20261029}"
if [[ "$TOUCHWAM_CONFIG" == train_touchwam_six_source_bs8_norecompute_20261029 ||
      "$TOUCHWAM_CONFIG" == train_touchwam_six_source_bs12_norecompute_20261029 ||
      "$TOUCHWAM_CONFIG" == train_touchwam_six_source_64gpu_bs12_20261029 ]]; then
  DEFAULT_ALPHA_AUDIT=/mnt/pfs/v89atp/workspace/setup/touchwam_bs8_norecompute_20261029/alpha_bitwise_audit.json
elif [[ "$TOUCHWAM_CONFIG" == train_touchwam_six_source_20261029_s9 ]]; then
  DEFAULT_ALPHA_AUDIT=/mnt/pfs/v89atp/workspace/setup/touchwam_s9_integration_20261029/alpha_bitwise_audit.json
else
  DEFAULT_ALPHA_AUDIT=/mnt/pfs/v89atp/workspace/setup/touchwam_six_source_ready_20261029/alpha_bitwise_audit.json
fi
export TOUCHWAM_ALPHA_AUDIT="${TOUCHWAM_ALPHA_AUDIT:-$DEFAULT_ALPHA_AUDIT}"
exec bash "$(dirname "$0")/train_touchwam_pretrain.sh" "$@"
