#!/usr/bin/env bash
# Pod entry: install the verified system packages, then hand over to node_launcher.py.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive
# Verified Ubuntu 22.04 dependency closure (copied from the baseline experiment); APT sees only
# this local source, so there is no repository synchronization race across the 8 pods.
(cd "$TASK_ROOT/system_debs" && sha256sum -c SHA256SUMS > /tmp/openwam-deb-verification.log)
mkdir -p /tmp/openwam-apt-lists/partial
printf 'deb [trusted=yes] file:%s/system_debs ./\n' "$TASK_ROOT" > /tmp/openwam-apt-sources.list
APT_LOCAL=(-o Dir::Etc::sourcelist=/tmp/openwam-apt-sources.list -o Dir::Etc::sourceparts=- -o Dir::State::lists=/tmp/openwam-apt-lists)
apt-get "${APT_LOCAL[@]}" update > /tmp/openwam-local-index.log 2>&1
if ! apt-get "${APT_LOCAL[@]}" -y --no-install-recommends -o Dpkg::Options::="--force-confold" \
    install gcc g++ ninja-build libaio1 libibverbs1 ibverbs-providers librdmacm1 libgl1 libglib2.0-0 ffmpeg \
    > /tmp/openwam-system-deps.log 2>&1; then
    cat /tmp/openwam-system-deps.log
    exit 1
fi
export CUDA_HOME=/mnt/pfs/v89atp/workspace/toolchains/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/mnt/pfs/v89atp/workspace/envs/openwam/bin:$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_EXTENSIONS_DIR=/tmp/openwam-torch-extensions
export TRITON_CACHE_DIR=/tmp/openwam-triton
export NCCL_DEBUG=INFO NCCL_IB_DISABLE=0
# The allocator the s9 options were measured with on 2 x 8 A800 (allocator only, no numeric effect).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /mnt/pfs/v89atp/workspace/envs/openwam/bin/python "$TASK_ROOT/pipeline/node_launcher.py" "$@"
