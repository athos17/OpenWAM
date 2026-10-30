#!/usr/bin/env bash
# Run a queue of 2-node runs without any ssh between the nodes (a800_1/a800_2 have none): start
# one worker per node; both read the same queue file on vePFS and meet at file barriers there.
#
#   NODE_RANK=<0|1> [WAIT_FOR_DATA=1] setsid nohup bash scripts/perf/touchwam_a800/queue_worker.sh <queue file> &
#
# Queue file: one run per line, "TAG CONFIG [PROFILE=1] [hydra overrides...]"; '#' comments.
# A finished run leaves $PERF_ROOT/$TAG.node<rank>.done (holding the exit code) and is skipped on a
# restart; delete those files (and the run directory) to repeat a run. WAIT_FOR_DATA=1 holds the
# first run until both copy logs on vePFS report COMPLETE.
set -uo pipefail
cd "$(dirname "$0")/../../.."
QUEUE=${1:?queue file}
: "${NODE_RANK:?set NODE_RANK}"
ROOT=${A800_ROOT:-/gaozt-test1/lyh}
PERF_ROOT=${PERF_ROOT:-$ROOT/perf_runs/touchwam_a800}
OTHER=$((1 - NODE_RANK))
BARRIER_TIMEOUT=${BARRIER_TIMEOUT:-1800}
mkdir -p "$PERF_ROOT"
log() { echo "$(date '+%F %T') node$NODE_RANK $*" | tee -a "$PERF_ROOT/queue.node$NODE_RANK.log"; }

wait_for() {  # wait_for <file> <seconds>
  local waited=0
  until [ -e "$1" ]; do
    sleep 10; waited=$((waited + 10))
    [ "$waited" -ge "$2" ] && return 1
  done
}

if [ "${WAIT_FOR_DATA:-0}" = 1 ]; then
  log "waiting for the data and FTP weight copies to complete"
  until grep -q '^COMPLETE' "$ROOT/datasets/touchwam_subset_v1/.copy_state/pull.log" 2>/dev/null &&
        grep -q '^COMPLETE' "$ROOT/models/.ftp_pull.log" 2>/dev/null; do
    sleep 60
  done
  log "copies complete"
fi

while read -r TAG CONFIG REST; do
  [ -e "$PERF_ROOT/$TAG.node$NODE_RANK.done" ] && { log "$TAG already done, skipping"; continue; }
  PROFILE=0
  read -r -a EXTRA <<< "${REST:-}"
  if [ "${EXTRA[0]:-}" = PROFILE=1 ]; then PROFILE=1; EXTRA=("${EXTRA[@]:1}"); fi
  touch "$PERF_ROOT/$TAG.ready$NODE_RANK"
  if ! wait_for "$PERF_ROOT/$TAG.ready$OTHER" "$BARRIER_TIMEOUT"; then
    log "$TAG: node$OTHER never became ready; stopping the queue"
    exit 1
  fi
  log "$TAG: start ($CONFIG profile=$PROFILE ${EXTRA[*]:-})"
  TAG=$TAG CONFIG=$CONFIG PROFILE=$PROFILE bash scripts/perf/touchwam_a800/run_node.sh "${EXTRA[@]}" < /dev/null
  status=$?
  echo "$status" > "$PERF_ROOT/$TAG.node$NODE_RANK.done"
  log "$TAG: exit $status"
  # Leave the node only after the peer finished too, so the next run starts on two idle nodes.
  wait_for "$PERF_ROOT/$TAG.node$OTHER.done" "$BARRIER_TIMEOUT" || log "$TAG: node$OTHER did not finish in time"
done < <(grep -v '^\s*\(#\|$\)' "$QUEUE")
log "queue finished"
