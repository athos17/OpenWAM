"""Mirror the finished part of the a800 subset copy as a symlink tree (for runs before the copy ends).

    python link_complete_buckets.py <copy_list.tsv> <subset root> <dest> [--rounds 0 1]

Root files and small metadata directories (round 0) are linked as they are; a bucket is linked only
when every file the copy list holds for it is present at full size. Bucket alias symlinks (SharpaDex
buckets/ -> partitions/) are recreated, so they resolve inside <dest>. Point
TOUCHWAM_SUBSET_ROOT=<dest> at the result; the copy itself is never touched.
"""
import argparse
import collections
import os
from pathlib import Path

# Directories whose children are buckets (relative to the dataset root), as in build_copy_list.py.
PARENTS = ("egodex_300h_video_v1/cached_500h/train", "robocoin_200h_v1_20261029", "agibot_pretrain_v1",
           "sharpadex_native_tcp_v1_20261026/partitions/*", "touchwam_pretrain_v1_20261028/sharpadex/partitions/*",
           "touchscale_lerobot_v3_full_v1_20261029/buckets", "ftp1_frozen25h_v1_20261029/*")
SMALL_DIRS = ("meta", "provenance", "control")


def bucket_of(rel):
    parts = rel.split("/")
    for parent in PARENTS:
        p = parent.split("/")
        if len(parts) > len(p) + 1 and all(a == b or b == "*" for a, b in zip(parts, p)) and parts[len(p)] not in SMALL_DIRS:
            return "/".join(parts[:len(p) + 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("copy_list")
    ap.add_argument("root", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    files, links, buckets = [], [], collections.defaultdict(list)
    for line in open(args.copy_list):
        rnd, kind, rel, val = line.rstrip("\n").split("\t")
        if int(rnd) not in args.rounds:
            continue
        b = bucket_of(rel) if kind == "f" else None
        if kind == "l":
            links.append((rel, val))
        elif b:
            buckets[b].append((rel, int(val)))
        else:
            files.append((rel, int(val)))
    done = {b for b, entries in buckets.items()
            if all((args.root / rel).is_file() and (args.root / rel).stat().st_size == size for rel, size in entries)}
    missing_root = [rel for rel, size in files if not ((args.root / rel).is_file() and (args.root / rel).stat().st_size == size)]
    if missing_root:
        raise SystemExit(f"{len(missing_root)} root/metadata files not copied yet, e.g. {missing_root[:3]}")
    for rel, _ in files:
        target = args.dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(args.root / rel)
    for b in sorted(done):
        target = args.dest / b
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(args.root / b)
    alias = 0
    for rel, value in links:
        target = args.dest / rel
        resolved = Path(os.path.normpath(args.dest / os.path.dirname(rel) / value))  # parent may not exist yet
        if bucket_of(rel) is None and resolved.exists() and not target.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(value)
            alias += 1
    by_source = collections.Counter(b.split("/")[0] + ("/" + b.split("/")[1] if b.startswith("ftp1") else "") for b in done)
    print(f"linked {len(done)}/{len(buckets)} buckets, {alias} aliases:", dict(sorted(by_source.items())))


if __name__ == "__main__":
    main()
