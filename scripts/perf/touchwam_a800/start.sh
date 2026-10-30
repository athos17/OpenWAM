#!/usr/bin/env bash
# Start one 2-node run in the background on a800_1 (rank 0) and a800_2 (rank 1), from any host
# with the a800_1/a800_2 ssh aliases. Returns immediately; follow with status.sh.
#
#   bash scripts/perf/touchwam_a800/start.sh <TAG> <CONFIG> [PROFILE=1] [-- extra hydra overrides]
set -euo pipefail
TAG=$1 CONFIG=$2
shift 2
PROFILE=0
if [ "${1:-}" = PROFILE=1 ]; then PROFILE=1; shift; fi
[ "${1:-}" = -- ] && shift
REPO=${A800_REPO:-/gaozt-test1/lyh/OpenWAM-touchwam}
EXTRA=$(printf ' %q' "$@")
for rank in 0 1; do
  host=a800_$((rank + 1))
  # Refuse to start on a node that is still busy (another run or a stale torchrun).
  if ssh -n "$host" "pgrep -f '[t]orch.distributed.run|[t]orchrun' >/dev/null"; then
    echo "$host: a torchrun is already running; not starting $TAG" >&2
    exit 1
  fi
done
for rank in 1 0; do
  host=a800_$((rank + 1))
  ssh -n "$host" "cd $REPO && (TAG=$TAG CONFIG=$CONFIG NODE_RANK=$rank PROFILE=$PROFILE setsid nohup \
    bash scripts/perf/touchwam_a800/run_node.sh$EXTRA > /dev/null 2>&1 < /dev/null &)"
  echo "started $TAG rank $rank on $host"
done
