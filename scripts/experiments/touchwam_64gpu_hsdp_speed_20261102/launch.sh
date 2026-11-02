#!/usr/bin/env bash
# Run on dev-cpu (bdy_cpu) from the frozen experiment directory. See README.md.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PY=/mnt/pfs/v89atp/workspace/envs/openwam/bin/python
API_PY=/root/aihc-gpu-smoke-20261022/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
case "${1:-help}" in
  preflight)
    PYTHONPATH="$TASK_ROOT/release" exec "$TASK_PY" "$TASK_ROOT/pipeline/preflight.py"
    ;;
  request|create|status|logs|stop)
    exec "$API_PY" "$TASK_ROOT/pipeline/cluster_job.py" "$1"
    ;;
  progress)
    exec python3 "$TASK_ROOT/pipeline/progress.py"
    ;;
  compare)
    shift
    exec python3 "$TASK_ROOT/pipeline/compare_with_baseline.py" "$@"
    ;;
  *)
    echo 'Usage: bash launch.sh preflight|request|create|status|logs|stop|progress|compare'
    ;;
esac
