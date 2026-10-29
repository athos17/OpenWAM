#!/usr/bin/env bash
# Explicit heterogeneous-encoder entry; retains preflight and publication gates.
set -euo pipefail
export TOUCHWAM_CONFIG=train_touchwam_ftp1
exec bash "$(dirname "$0")/train_touchwam_pretrain.sh" "$@"
