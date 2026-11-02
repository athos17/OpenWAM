#!/usr/bin/env bash
# Pod entry, as in the r2 production job (setup/touchwam_64gpu_bs48_20261030_r2) plus the gloo interface.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DEBS=/mnt/pfs/v89atp/workspace/setup/alpha_native_64gpu_s9_full_20261029/system_debs
if [[ ! -e /mnt/pfs/ls3r34 ]]; then ln -s /mnt/pfs/v89atp /mnt/pfs/ls3r34; fi
[[ /mnt/pfs/ls3r34 -ef /mnt/pfs/v89atp ]] || { echo 'PFS alias mismatch' >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive
(cd "$TASK_DEBS" && sha256sum -c SHA256SUMS > /tmp/touchwam-debs.log)
mkdir -p /tmp/touchwam-apt-lists/partial
printf 'deb [trusted=yes] file:%s ./\n' "$TASK_DEBS" > /tmp/touchwam-apt-sources.list
APT_LOCAL=(-o Dir::Etc::sourcelist=/tmp/touchwam-apt-sources.list -o Dir::Etc::sourceparts=- -o Dir::State::lists=/tmp/touchwam-apt-lists)
apt-get "${APT_LOCAL[@]}" update > /tmp/touchwam-apt-index.log 2>&1
apt-get "${APT_LOCAL[@]}" -y --no-install-recommends -o Dpkg::Options::="--force-confold" install gcc g++ ninja-build libaio1 libibverbs1 ibverbs-providers librdmacm1 libgl1 libglib2.0-0 ffmpeg > /tmp/touchwam-system-deps.log 2>&1
export CUDA_HOME=/mnt/pfs/v89atp/workspace/toolchains/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/mnt/pfs/v89atp/workspace/envs/openwam/bin:$PATH"
export PYTHONPATH="$TASK_ROOT/release"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export TORCH_EXTENSIONS_DIR=/tmp/touchwam-torch-extensions TRITON_CACHE_DIR=/tmp/touchwam-triton
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=0
# training.touchwam_p1.deferred_modality_sync opens a gloo group; the pods' NCCL bootstrap uses eth0 too.
export GLOO_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python "$TASK_ROOT/pipeline/node_launcher.py"
