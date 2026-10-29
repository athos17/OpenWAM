#!/usr/bin/env python3
"""Freeze a seeded global LeRobot episode budget; reference existing media in place."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openwam.dataloader.publication import publication_contract
from openwam.dataloader.touchwam_pretrain import _load_video_episodes
from openwam.dataloader.utils.lerobotv3 import apply_info_splits, parse_info_json


def freeze(root, output, *, max_hours, seed=42, split="train"):
    root, output = Path(root).resolve(strict=True), Path(output)
    if not np.isfinite(max_hours) or max_hours <= 0:
        raise ValueError("max_hours must be positive")
    publication = publication_contract(root)
    population, entries = [], {}
    roots = [root] if (root / "meta/info.json").is_file() else sorted(
        p.parent.parent for p in root.rglob("meta/info.json"))
    for bucket in roots:
        info = parse_info_json(bucket)
        eps = apply_info_splits(_load_video_episodes(bucket, []), split, info.get("splits", {}))
        name = str(bucket.relative_to(root))
        metadata = {"meta/info.json", "meta/source_definition.json", "tactile/manifest.json"}
        metadata |= {str(p.relative_to(bucket)) for p in (bucket / "meta/episodes").rglob("*.parquet")}
        entries[name] = {"fps": float(info["fps"]), "episode_indices": [], "frames": 0,
                         "metadata_sha256": {relative: hashlib.sha256((bucket / relative).read_bytes()).hexdigest()
                                             for relative in sorted(metadata) if (bucket / relative).is_file()}}
        for row in eps.itertuples():
            population.append((name, int(row.episode_index), int(row.length), float(row.length) / info["fps"]))
    available_s = sum(row[3] for row in population)
    if available_s <= max_hours * 3600:
        selected = population
        algorithm = "all_available_train_episodes_below_budget"
    else:
        selected, seconds = [], 0.
        for i in np.random.RandomState(seed).permutation(len(population)):
            row = population[i]
            selected.append(row)
            seconds += row[3]
            if seconds >= max_hours * 3600:
                break
        algorithm = "seeded_global_episode_prefix_until_budget"
    for name, episode, length, _ in selected:
        entries[name]["episode_indices"].append(episode)
        entries[name]["frames"] += length
    entries = {name: entry for name, entry in entries.items() if entry["episode_indices"]}
    for entry in entries.values():
        entry["episode_indices"].sort()
    result = {"schema": "touchwam_episode_selection_v1", "dataset_root": str(root), "split": split,
              "seed": seed, "algorithm": algorithm, "requested_hours": max_hours,
              "available_hours": available_s / 3600, "selected_hours": sum(row[3] for row in selected) / 3600,
              "selected_episodes": len(selected), "selected_frames": sum(row[2] for row in selected),
              "publication": publication, "buckets": entries}
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != raw:
        raise FileExistsError(f"Refusing to replace a different frozen selection: {output}")
    if not output.exists():
        with output.open("xb") as handle:
            handle.write(raw)
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "buckets": len(entries), **{k: result[k] for k in
                      ("algorithm", "seed", "requested_hours", "selected_hours", "selected_episodes", "selected_frames")}}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-hours", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    freeze(args.dataset_root, args.output, max_hours=args.max_hours, seed=args.seed, split=args.split)
