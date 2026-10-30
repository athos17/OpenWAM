#!/usr/bin/env bash
# After the copy: drop stale chunk files, restore bdy mtimes (SharpaDex binds them) and build the
# EgoDex subset index (README.md).
set -euo pipefail
cd "$(dirname "$0")/../../.."
SUBSET=${TOUCHWAM_SUBSET_ROOT:-/gaozt-test1/lyh/datasets/touchwam_subset_v1}
PY=${OPENWAM_ENV:-/gaozt-test1/zhanxch/miniconda3/envs/openwam}/bin/python
# A .part next to a finished file is a leftover of an interrupted first pass of the copier.
find "$SUBSET" -name '*.part' | while read -r part; do
  [ -e "${part%.part}" ] && rm -f "$part"
done
left=$(find "$SUBSET" -name '*.part' | wc -l)
[ "$left" = 0 ] || { echo "$left unfinished files remain in $SUBSET" >&2; exit 1; }
"$PY" scripts/perf/touchwam_a800/sync_mtimes.py "$SUBSET/.copy_state/copy_list.tsv" "$SUBSET"
"$PY" scripts/perf/touchwam_a800/make_egodex_subset.py "$SUBSET/egodex_300h_video_v1" "$SUBSET/egodex_300h_video_v1_sub"
