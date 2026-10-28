#!/usr/bin/env python3
"""Freeze a small, independent FTP-1 episode census without converting data.

The existing full-release census and source definitions are never modified.
Prefer already validated LeRobot v3 tasks, then minimize the estimated selected
payload. Selection samples complete episodes across one task's episode range.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from convert import atomic_json, bucket_path, resolve_selection


def select_episodes(group, ids):
    if "source_census" in group:
        raise ValueError("Select from an original census, not another pilot")
    original = copy.deepcopy(group)
    ends = original["episode_ends"]
    if not ids or ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < len(ends) for i in ids):
        raise ValueError("Pilot episode IDs must be valid, unique and ascending")
    lengths = [ends[i] - (ends[i - 1] if i else 0) for i in ids]
    result = dict(original)
    result.update(
        {
            "selection_schema": "touchwam.ftp1.episode_selection.v1",
            "source_census": original,
            "selected_source_episode_indices": ids,
            "episodes": len(ids),
            "frames": sum(lengths),
            "episode_ends": np.cumsum(lengths).tolist(),
        }
    )
    resolve_selection(result)
    return result


def make_plan(census, definitions_root, episodes_per_source=5, reuse_root=None, sources=None, episode_overrides=None):
    if not 1 <= episodes_per_source <= 10:
        raise ValueError("Pilot scope is 1–10 complete episodes per source")
    available = {s["dataset"] for s in census if s["dataset"] != "sharpa"}
    requested = set(sources or available)
    if requested - available or "sharpa" in requested:
        raise ValueError(f"Unknown or excluded sources: {sorted(requested - available)}")
    episode_overrides = episode_overrides or {}
    if set(episode_overrides) - requested:
        raise ValueError("Episode override source is not selected")
    selected, summaries = [], []
    for source in census:
        name = source["dataset"]
        if name not in requested:
            continue
        candidates = []
        override = episode_overrides.get(name)
        for group in source["groups"]:
            if override and group["group"] != override["group"]:
                continue
            if not group["episode_ends"] or group["frames"] == 0:
                continue
            ids = np.linspace(
                0, len(group["episode_ends"]) - 1, min(episodes_per_source, len(group["episode_ends"])), dtype=int
            ).tolist()
            if override:
                ids = override["episode_indices"]
                if not 1 <= len(ids) <= 10:
                    raise ValueError("Explicit pilot overrides allow 1–10 complete episodes")
            chosen = select_episodes(group, ids)
            estimate = (group["end_offset"] - group["offset"]) * chosen["frames"] / group["frames"]
            existing = bucket_path(reuse_root, name, group) if reuse_root else None
            reused = bool(
                existing
                and (existing / "meta/validation.json").exists()
                and (existing / "meta/conversion_complete.json").exists()
            )
            candidates.append((not reused, estimate, group["group"], chosen, existing if reused else None))
        if not candidates:
            raise ValueError(f"No task contains {episodes_per_source} complete episodes: {name}")
        _, estimate, _, chosen, existing = min(candidates, key=lambda row: row[:3])
        definition = json.loads((Path(definitions_root) / f"{name}.json").read_text())
        selected.append(
            {
                "dataset": name,
                "scope": "pilot_episodes",
                "groups": [chosen],
                "frames": chosen["frames"],
                "episodes": chosen["episodes"],
                "source_total_frames": source.get("frames", sum(g["frames"] for g in source["groups"])),
            }
        )
        summaries.append(
            {
                "source": name,
                "group": chosen["group"],
                "episodes": chosen["episodes"],
                "source_episode_indices": chosen["selected_source_episode_indices"],
                "frames": chosen["frames"],
                "seconds": chosen["frames"] / definition["nominal_fps"] if definition["nominal_fps"] else None,
                "estimated_compressed_input_bytes": round(estimate),
                "estimate_basis": "task_archive_bytes_times_selected_frame_fraction; exact_chunk_size_checked_before_staging",
                "reuse_bucket": str(existing) if existing else None,
                "tactile_representation": definition["tactile_representation"],
                "tactile_calibration_verified": False,
            }
        )
    plan = {
        "schema": "touchwam.ftp1.pilot_plan.v1",
        "scope": "pilot_episodes",
        "excluded_sources": ["sharpa"],
        "episodes_per_source": episodes_per_source,
        "sources": summaries,
        "total_episodes": sum(x["episodes"] for x in summaries),
        "total_frames": sum(x["frames"] for x in summaries),
        "estimated_compressed_input_bytes": sum(x["estimated_compressed_input_bytes"] for x in summaries),
        "census_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        "publication_scope": "only selected episodes; never a claim that the full FTP release is complete",
    }
    if episode_overrides:
        plan["episode_overrides"] = episode_overrides
    return selected, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", default="configs/data_sources/ftp1/archive_census.json")
    parser.add_argument("--definitions", default="configs/data_sources/ftp1")
    parser.add_argument("--output", required=True, help="New pilot selection directory, never the original definitions")
    parser.add_argument("--episodes-per-source", type=int, default=5)
    parser.add_argument("--reuse-root", help="Existing full v3 output to prefer over re-reading raw tar")
    parser.add_argument("--sources", nargs="*")
    parser.add_argument(
        "--episode-overrides", help="JSON source -> {group, episode_indices, reason} for smaller explicit pilots"
    )
    args = parser.parse_args()
    source_census = Path(args.census).resolve(strict=True)
    output = Path(args.output).resolve()
    if output == source_census.parent or output == Path(args.definitions).resolve():
        parser.error("Pilot files must use a separate output namespace")
    census, plan = make_plan(
        json.loads(source_census.read_text()),
        args.definitions,
        args.episodes_per_source,
        args.reuse_root,
        args.sources,
        json.loads(Path(args.episode_overrides).read_text()) if args.episode_overrides else None,
    )
    plan["source_census_file_sha256"] = hashlib.sha256(source_census.read_bytes()).hexdigest()
    for name, value in [("pilot_census.json", census), ("pilot_plan.json", plan)]:
        target = output / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError(f"Pilot plan is immutable; choose a new namespace: {target}")
        atomic_json(target, value)
    print(json.dumps({"census": str(output / "pilot_census.json"), "plan": plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
