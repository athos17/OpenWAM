#!/usr/bin/env python3
"""Build the sim-vs-real comparison datasets (LeRobot v3, Marvin layout) from a real and a sim bucket.

Creates three self-contained datasets under ``--out-root`` that share ONE recomputed
normalization file (union of every real and sim frame, so neither source is clipped and
all arms live in the same normalized action space):

  sim<N>/            all sim episodes
  sim<N>_real<K>/    all sim episodes, then K sampled real episodes
  real<K>/           the K sampled real episodes

Real episodes are sampled with a fixed seed from the real bucket minus ``--holdout``
(kept for evaluation), stratified by the short (< --long-from) / long episode groups.
Videos are hard-linked (no copy); data / episode tables are rewritten with new
episode / global indices. ``meta/composition.json`` maps every new episode to its
source (dataset dir + episode index) -- scripts/precompute/build_encode_cache.py
``assemble`` uses it to build caches from the source caches without re-encoding.

Both sources must use one data / video file per episode (chunk 0, file == episode).

Example:
  python scripts/sim2real/build_mix_datasets.py \
      --real /gaozt-test1/zdx/datasets/pick_place_2_3_openwam_v3 \
      --sim /gaozt-test1/zdx/datasets/sim_generated/h20_cleanup_twin200_20261015_099/dataset \
      --out-root /gaozt-test1/lyh/datasets/sim2real_099 --n-real 20 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from convert_marvin_lerobot_v3 import _stats  # noqa: E402  (same definition as the converter)
from openwam.dataloader.marvin import marvin_raw_vector  # noqa: E402

CAMERAS = ("observation.images.left_head", "observation.images.left_wrist", "observation.images.right_wrist")
EPISODES = "meta/episodes/chunk-000/file-000.parquet"


def episodes(root: Path) -> pd.DataFrame:
    eps = pd.read_parquet(root / EPISODES)
    for col in ["data/file_index"] + [f"videos/{c}/file_index" for c in CAMERAS]:
        if not (eps[col].to_numpy() == eps["episode_index"].to_numpy()).all():
            raise ValueError(f"{root}: {col} != episode_index (one file per episode required)")
    for col in ["data/chunk_index"] + [f"videos/{c}/chunk_index" for c in CAMERAS]:
        if (eps[col] != 0).any():
            raise ValueError(f"{root}: {col} must be 0")
    return eps


def check_compatible(real: Path, sim: Path) -> dict:
    ir, is_ = (json.loads((p / "meta/info.json").read_text()) for p in (real, sim))
    for key in ("codebase_version", "robot_type", "fps", "features", "openwam_contract"):
        if ir[key] != is_[key]:
            raise ValueError(f"real and sim differ in info.json[{key!r}]")
    tr, ts = (pd.read_parquet(p / "meta/tasks.parquet") for p in (real, sim))
    if not tr.equals(ts) or len(tr) != 1:
        raise ValueError("real and sim must share the same single task")
    return ir


def raw_frames(root: Path, eps: pd.DataFrame, column: str) -> np.ndarray:
    parts = [np.stack(pd.read_parquet(root / f"data/chunk-000/file-{e:03d}.parquet", columns=[column])[column].values)
             for e in eps.episode_index]
    return marvin_raw_vector(np.concatenate(parts).astype(np.float32))


def sample_real(eps: pd.DataFrame, n: int, seed: int, holdout: set, long_from: int) -> list:
    pool = eps[~eps.episode_index.isin(holdout)].episode_index.to_numpy()
    short, long_ = pool[pool < long_from], pool[pool >= long_from]
    n_long = int(round(n * len(long_) / len(pool)))
    rng = np.random.default_rng(seed)
    picked = np.concatenate([rng.choice(short, n - n_long, replace=False), rng.choice(long_, n_long, replace=False)])
    return sorted(int(e) for e in picked)


def write_dataset(out: Path, members: list, info_base: dict, tasks_src: Path, stats_file: Path, provenance: dict):
    """members: [(source_name, source_root, source_eps_row)] in new episode order."""
    if out.exists():
        raise FileExistsError(f"{out} exists; remove it first")
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for sub in ["meta/episodes/chunk-000", "data/chunk-000"] + [f"videos/{c}/chunk-000" for c in CAMERAS]:
        (tmp / sub).mkdir(parents=True)
    rows, composition, start = [], [], 0
    template = pq.read_schema(members[0][1] / EPISODES)
    for new, (name, root, src) in enumerate(members):
        old = int(src.episode_index)
        n = int(src.length)
        table = pq.read_table(root / f"data/chunk-000/file-{old:03d}.parquet")
        if table.num_rows != n or set(table.column("episode_index").to_pylist()) != {old}:
            raise ValueError(f"{root} episode {old}: data file does not hold exactly that episode")
        frame = np.asarray(table.column("frame_index").to_pylist())
        if not (frame == np.arange(n)).all():
            raise ValueError(f"{root} episode {old}: frame_index is not 0..{n - 1}")
        for col, values in (("episode_index", np.full(n, new)), ("index", start + frame), ("task_index", np.zeros(n))):
            i = table.schema.get_field_index(col)
            table = table.set_column(i, table.schema.field(i), pa.array(values.tolist(), type=table.schema.field(i).type))
        pq.write_table(table, tmp / f"data/chunk-000/file-{new:03d}.parquet")
        for cam in CAMERAS:
            os.link(root / f"videos/{cam}/chunk-000/file-{old:03d}.mp4", tmp / f"videos/{cam}/chunk-000/file-{new:03d}.mp4")
        row = {"episode_index": new, "tasks": list(src.tasks), "length": n,
               "dataset_from_index": start, "dataset_to_index": start + n, "data/chunk_index": 0, "data/file_index": new}
        for cam in CAMERAS:
            row[f"videos/{cam}/chunk_index"], row[f"videos/{cam}/file_index"] = 0, new
        rows.append(row)
        composition.append({"episode_index": new, "source": name, "source_dir": str(root),
                            "source_episode_index": old, "length": n})
        start += n
    pq.write_table(pa.Table.from_pylist(rows, schema=template), tmp / EPISODES)
    shutil.copy2(tasks_src, tmp / "meta/tasks.parquet")
    shutil.copy2(stats_file, tmp / "meta/normalization_stats.npy")
    info = {k: v for k, v in info_base.items() if k != "simulation_provenance"}
    info.update({"total_episodes": len(rows), "total_frames": start, "total_tasks": 1,
                 "total_videos": len(rows) * len(CAMERAS), "total_chunks": 1, "splits": {"train": f"0:{len(rows)}"},
                 "openwam_composition": "meta/composition.json"})
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    (tmp / "meta/composition.json").write_text(json.dumps({**provenance, "episodes": composition}, indent=2))
    tmp.rename(out)
    return len(rows), start


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--sim", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-real", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", default="3,30,57,70,80,85", help="real episodes never sampled (evaluation)")
    ap.add_argument("--long-from", type=int, default=64, help="real episodes >= this index form the long group")
    args = ap.parse_args(argv)

    info = check_compatible(args.real, args.sim)
    eps_r, eps_s = episodes(args.real), episodes(args.sim)
    holdout = {int(e) for e in args.holdout.split(",") if e}
    picked = sample_real(eps_r, args.n_real, args.seed, holdout, args.long_from)

    # Union stats over EVERY real and sim frame: all arms (and later real-only / full-real
    # runs and offline evaluation on held-out real episodes) share one unclipped space.
    args.out_root.mkdir(parents=True, exist_ok=True)
    stats_file = args.out_root / "normalization_stats.npy"
    action = np.concatenate([raw_frames(args.real, eps_r, "action"), raw_frames(args.sim, eps_s, "action")])
    state = np.concatenate([raw_frames(args.real, eps_r, "observation.state"), raw_frames(args.sim, eps_s, "observation.state")])
    np.save(stats_file, {"eef": _stats(action), "eef_state": _stats(state)}, allow_pickle=True)
    provenance = {
        "real_dir": str(args.real), "sim_dir": str(args.sim), "seed": args.seed, "n_real": args.n_real,
        "holdout_real_episodes": sorted(holdout), "sampled_real_episodes": picked,
        "normalization": {"definition": "scripts/convert_marvin_lerobot_v3.py::_stats (rot6d identity)",
                          "frames": {"real": int(eps_r.length.sum()), "sim": int(eps_s.length.sum())},
                          "sources": "all real episodes + all sim episodes"},
    }
    (args.out_root / "composition.json").write_text(json.dumps(provenance, indent=2))

    sim = [("sim", args.sim, r) for r in eps_s.itertuples()]
    real = [("real", args.real, r) for r in eps_r.set_index("episode_index", drop=False).loc[picked].itertuples()]
    n_sim = len(eps_s)
    for name, members in ((f"sim{n_sim}", sim), (f"sim{n_sim}_real{args.n_real}", sim + real), (f"real{args.n_real}", real)):
        n_eps, n_frames = write_dataset(args.out_root / name, members, info, args.real / "meta/tasks.parquet",
                                        stats_file, provenance)
        print(f"{name}: {n_eps} episodes, {n_frames} frames")
    print("sampled real episodes:", picked)


if __name__ == "__main__":
    main()
