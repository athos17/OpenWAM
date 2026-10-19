"""P1: extract per-segment keyframes for review and LLM labeling.

PICK  head: start / grasp / peak / release / after (release + 1 s) / end;  wrist (that hand): pre60 / pre40 / pre20 (frames before grasp) / grasp / lift
STILL head: start / mid / end
Plus one per-episode overview sheet: the head frame at every PICK grasp, numbered.

Example:
  python scripts/subtask/keyframes.py --dataset data/pick_place_2_3_openwam_v3 --out outputs/subtask --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HEAD_CAMERA, WRIST_CAMERA, episode_table, read_frames, save_jpeg, video_path  # noqa: E402


def plan_episode(segs: pd.DataFrame) -> list:
    """[(seg_id, camera, tag, frame)] for one episode."""
    items = []
    n = int(segs.end.max())
    for s in segs.itertuples():
        last = s.end - 1
        if s.type == "PICK":
            ev = json.loads(s.events)
            g, p, r = ev["grasp"], ev["peak"], ev["release"]
            for tag, f in (("start", s.start), ("grasp", g), ("peak", p), ("release", r), ("end", last)):
                items.append((s.seg_id, HEAD_CAMERA, tag, min(max(f, s.start), last) if tag != "release" else r))
            # The next pick (other hand) can start right after a release, so judge the drop
            # from a fixed delay instead of the segment end.
            items.append((s.seg_id, HEAD_CAMERA, "after", min(r + 30, n - 1)))
            # Small objects (coffee capsules, dice, caps) vanish inside the closed hand; the wrist
            # camera sees them clearly 0.5-2 s before the lift starts.
            for tag, f in (("pre60", g - 60), ("pre40", g - 40), ("pre20", g - 20), ("grasp", g), ("lift", (g + p) // 2)):
                items.append((s.seg_id, WRIST_CAMERA[s.hand], tag, max(f, s.start)))
        else:
            for tag, f in (("start", s.start), ("mid", (s.start + last) // 2), ("end", last)):
                items.append((s.seg_id, HEAD_CAMERA, tag, f))
    return items


def overview_sheet(frames: dict, picks: list, cols: int = 6, tile_w: int = 320):
    import cv2

    tiles = []
    for seg_id, f in picks:
        img = frames[f]
        h = round(img.shape[0] * tile_w / img.shape[1])
        t = cv2.resize(img, (tile_w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (64, 30), (0, 0, 0), -1)
        cv2.putText(t, f"#{seg_id}", (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=0)


def run_episode(job):
    dataset, out, ep, ep_row, segs = job
    items = plan_episode(segs)
    root = out / "keyframes" / f"ep{ep:03d}"
    manifest = []
    by_cam = {}
    for seg_id, cam, tag, f in items:
        by_cam.setdefault(cam, set()).add(f)
    picks = [(s.seg_id, json.loads(s.events)["grasp"]) for s in segs.itertuples() if s.type == "PICK"]
    by_cam.setdefault(HEAD_CAMERA, set()).update(f for _, f in picks)
    decoded = {}
    for cam, frames in by_cam.items():
        decoded[cam] = read_frames(video_path(dataset, ep_row, cam), frames, max_width=640 if cam == HEAD_CAMERA else 512)
    for seg_id, cam, tag, f in items:
        short = "head" if cam == HEAD_CAMERA else ("lwrist" if "left" in cam else "rwrist")
        path = root / f"seg{seg_id:03d}_{short}_{tag}_{f:05d}.jpg"
        if not path.exists():
            save_jpeg(path, decoded[cam][f])
        manifest.append({"episode_index": ep, "seg_id": int(seg_id), "camera": short, "tag": tag, "frame": int(f),
                         "path": str(path.relative_to(out))})
    if picks:
        sheet = root / "overview.jpg"
        save_jpeg(sheet, overview_sheet(decoded[HEAD_CAMERA], picks), quality=85)
        manifest.append({"episode_index": ep, "seg_id": -1, "camera": "head", "tag": "overview", "frame": -1,
                         "path": str(sheet.relative_to(out))})
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/subtask"))
    ap.add_argument("--episodes", type=str, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    segs = pd.read_parquet(args.out / "segments_rules.parquet")
    eps = episode_table(args.dataset).set_index("episode_index", drop=False)
    selected = [int(e) for e in args.episodes.split(",")] if args.episodes else sorted(segs.episode_index.unique())
    jobs = [(args.dataset, args.out, ep, eps.loc[ep], segs[segs.episode_index == ep]) for ep in selected]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, m in enumerate(pool.map(run_episode, jobs), 1):
            rows.extend(m)
            print(f"\r{i}/{len(jobs)} episodes", end="", flush=True)
    print()
    path = args.out / "keyframes.parquet"
    df = pd.DataFrame(rows)
    if path.exists() and args.episodes:
        old = pd.read_parquet(path)
        df = pd.concat([old[~old.episode_index.isin(selected)], df], ignore_index=True)
    df.sort_values(["episode_index", "seg_id", "camera", "tag"]).to_parquet(path)
    print(f"{len(df)} keyframes -> {path}")


if __name__ == "__main__":
    main()
