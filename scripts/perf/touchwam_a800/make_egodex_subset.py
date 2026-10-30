"""EgoDex reader root for the a800 subset: only episodes whose video was copied.

EgoDexReuseVideoDataset indexes episodes.parquet globally and requires every listed video inside its
root, so a partial copy cannot be read directly. This writes <dest> with
  - cached_500h/... hardlinks of the copied videos (same vePFS, no extra space),
  - episodes.parquet filtered to those videos (rows, order and fields unchanged),
  - dataset_info.json with the matching index_sha256 / total_episodes / total_frames.

    python make_egodex_subset.py <subset>/egodex_300h_video_v1 <subset>/egodex_300h_video_v1_sub
Rerun after more EgoDex buckets arrive; it rebuilds <dest> from scratch.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    args = ap.parse_args()
    table = pq.read_table(args.source / "episodes.parquet")
    present = sorted(p for p in set(table.column("video_path").to_pylist()) if (args.source / p).is_file())
    if not present:
        raise SystemExit("no EgoDex video of the index is present yet")
    if args.dest.exists():
        shutil.rmtree(args.dest)
    for rel in present:
        (args.dest / rel).parent.mkdir(parents=True, exist_ok=True)
        os.link(args.source / rel, args.dest / rel)
    kept = table.filter(pc.is_in(table.column("video_path"), value_set=pa.array(present)))
    pq.write_table(kept, args.dest / "episodes.parquet")
    info = json.loads((args.source / "dataset_info.json").read_text())
    info.update(total_episodes=kept.num_rows, total_frames=int(pc.sum(kept.column("length")).as_py()),
                index_sha256=sha256(args.dest / "episodes.parquet"), videos=len(present),
                tasks=len(set(kept.column("task").to_pylist())),
                subset_of={"index_sha256": json.loads((args.source / "dataset_info.json").read_text())["index_sha256"],
                           "note": "a800 benchmark subset: episodes whose videos were copied"})
    (args.dest / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"{kept.num_rows} episodes, {len(present)} videos, {info['tasks']} tasks -> {args.dest}")


if __name__ == "__main__":
    main()
