#!/usr/bin/env python3
"""Validate converted FTP-1 v3 buckets and finish interoperable metadata.

Safe to run while convert.py is active: only conversion_complete buckets are
visited. --watch continues until the full converter catalog is complete. Full
frame/episode indexing is checked; each video container and its decoded boundary
frames are checked. No raw tactile signal is promoted to pressure supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def stats_array(value):
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        return None
    arr = arr.astype(np.float64)
    finite = np.isfinite(arr)
    safe = np.where(finite, arr, np.nan)
    if arr.ndim == 1:
        safe = safe[:, None]
    return {
        "min": np.nanmin(safe, axis=0).tolist(),
        "max": np.nanmax(safe, axis=0).tolist(),
        "mean": np.nanmean(safe, axis=0).tolist(),
        "std": np.nanstd(safe, axis=0).tolist(),
        "count": [len(arr)],
    }


def asset_identity(bucket, info):
    """Bind cached checks to actual immutable data assets, not only receipts."""
    paths = [
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
    ]
    paths += [
        info["video_path"].format(video_key=key, chunk_index=0, file_index=0)
        for key, spec in info["features"].items()
        if spec["dtype"] == "video"
    ]
    return {name: [(bucket / name).stat().st_size, (bucket / name).stat().st_mtime_ns] for name in paths}


def _validate_bucket(bucket, regions_root=None):
    bucket = Path(bucket)
    complete = json.loads((bucket / "meta/conversion_complete.json").read_text())
    definition_path = bucket / "meta/source_definition.json"
    definition = json.loads(definition_path.read_text())
    if regions_root is not None:
        regions_file = Path(regions_root) / f"{definition['source']}.json"
        if regions_file.exists():
            definition["tactile_region_definition"] = json.loads(regions_file.read_text())
    layout = definition["camera_layout"]
    definition.setdefault("camera_provenance", sorted(set(v for v in layout.values() if v)))
    for role in ("left_wrist", "right_wrist"):
        if layout.get(role) == layout.get("head"):
            layout[role] = None
    definition["physical_time_verified"] = bool(definition.get("physical_time_valid", False))
    atomic_json(definition_path, definition)
    info_path = bucket / "meta/info.json"
    info = json.loads(info_path.read_text())
    corrected_shape = False
    for feature in info["features"].values():
        if feature["dtype"] != "video" and len(feature.get("shape", [])) > 1:
            feature["native_shape"] = feature["shape"]
            feature["shape"] = [int(np.prod(feature["native_shape"]))]
            corrected_shape = True
    if corrected_shape:
        if (bucket / "tactile/manifest.json").exists():
            raise ValueError(
                "Cannot change info fingerprint after tactile publication; regenerate in a new dataset version"
            )
        atomic_json(info_path, info)
    result_path = bucket / "meta/validation.json"
    assets = asset_identity(bucket, info)
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if (
            previous.get("conversion_signature") == complete["signature"]
            and previous.get("schema") == "ftp1-v3-validation-3"
            and previous.get("assets") == assets
        ):
            return previous
    info = json.loads((bucket / "meta/info.json").read_text())
    definition = json.loads((bucket / "meta/source_definition.json").read_text())
    definition["physical_time_verified"] = bool(definition.get("physical_time_valid", False))
    table = pq.read_table(bucket / "data/chunk-000/file-000.parquet")
    eps = pq.read_table(bucket / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert len(table) == info["total_frames"] == complete["frames"]
    assert len(eps) == info["total_episodes"]
    assert np.array_equal(table["index"].to_numpy(), np.arange(len(table)))
    ep_idx = table["episode_index"].to_numpy()
    frame_idx = table["frame_index"].to_numpy()
    ts = table["timestamp"].to_numpy()
    tasks = pq.read_table(bucket / "meta/tasks.parquet").to_pydict()
    assert "task_index" in tasks and "task" in tasks
    assert len(tasks["task_index"]) == info["total_tasks"]
    assert set(table["task_index"].to_pylist()) <= set(tasks["task_index"])
    last = 0
    for i, ep in enumerate(eps):
        a, b = ep["dataset_from_index"], ep["dataset_to_index"]
        assert ep["episode_index"] == i
        assert ep["source/group"] == definition["source_group"]
        assert ep["data/chunk_index"] == ep["data/file_index"] == 0
        assert a == last and b - a == ep["length"] and b <= len(table)
        assert np.array_equal(ep_idx[a:b], np.full(b - a, i))
        assert np.array_equal(frame_idx[a:b], np.arange(b - a))
        assert np.allclose(ts[a:b], np.arange(b - a) / info["fps"])
        last = b
    assert last == len(table)
    stats, video_results = {}, []
    for feature, spec in info["features"].items():
        if spec["dtype"] != "video":
            assert feature in table.column_names and table[feature].null_count == 0, feature
            if feature.endswith(".valid"):
                assert str(table[feature].type) == "bool", feature
            elif spec["dtype"] != "string":
                value = np.asarray(table[feature].to_pylist())
                expected_shapes = [(len(table), *spec["shape"])]
                if spec["shape"] == [1]:
                    expected_shapes.append((len(table),))
                assert value.shape in expected_shapes, (feature, value.shape, spec["shape"])
                if len(spec["shape"]) > 1:
                    raise ValueError(f"LeRobot numeric features must be flat: {feature}")
            if spec["dtype"] != "string":
                val = stats_array(table[feature].to_pylist())
                if val is not None:
                    stats[feature] = val
            continue
        path = bucket / info["video_path"].format(video_key=feature, chunk_index=0, file_index=0)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            assert stream.frames == len(table), (path, stream.frames, len(table))
            assert stream.height == spec["shape"][0] and stream.width == spec["shape"][1]
            assert abs(float(stream.average_rate) - info["fps"]) < 1e-6
            first = next(container.decode(video=0)).to_ndarray(format="rgb24")
            # Seek to the final GOP, then decode through the last frame.
            target = max(0, len(table) - 1) / info["fps"]
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            tail = list(container.decode(video=0))
            assert tail, path
            last_frame = tail[-1].to_ndarray(format="rgb24")
            assert np.isclose(float(tail[-1].time), target, atol=1 / info["fps"] * 0.25)
        sample = np.stack([first, last_frame]).astype(np.float64) / 255
        stats[feature] = {
            "min": sample.min(axis=(0, 1, 2))[:, None, None].tolist(),
            "max": sample.max(axis=(0, 1, 2))[:, None, None].tolist(),
            "mean": sample.mean(axis=(0, 1, 2))[:, None, None].tolist(),
            "std": sample.std(axis=(0, 1, 2))[:, None, None].tolist(),
            "count": [2],
        }
        valid_key = feature + ".valid"
        assert valid_key in table.column_names
        for ep in eps:
            assert ep[f"videos/{feature}/chunk_index"] == ep[f"videos/{feature}/file_index"] == 0
            assert np.isclose(ep[f"videos/{feature}/from_timestamp"], ep["dataset_from_index"] / info["fps"])
            assert np.isclose(ep[f"videos/{feature}/to_timestamp"], ep["dataset_to_index"] / info["fps"])
        video_results.append(
            {
                "feature": feature,
                "frames": len(table),
                "bytes": path.stat().st_size,
                "valid_frames": int(table[valid_key].to_numpy().sum()),
            }
        )
    atomic_json(bucket / "meta/stats.json", stats)
    atomic_json(bucket / "meta/source_definition.json", definition)
    result = {
        "schema": "ftp1-v3-validation-3",
        "bucket": str(bucket),
        "conversion_signature": complete["signature"],
        "assets": assets,
        "frames": len(table),
        "episodes": len(eps),
        "videos": video_results,
        "training_ready": definition["training_ready"],
        "physical_time_verified": definition["physical_time_verified"],
        "tactile_enabled": False,
        "image_stats_sampling": "first_and_last_frame_per_group",
        "validated_at": time.time(),
    }
    atomic_json(result_path, result)
    return result


def validate_bucket(bucket, regions_root=None):
    import fcntl

    lock_path = Path(bucket) / "meta/.validation.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _validate_bucket(bucket, regions_root)


def publish_sources(root, census, done, errors):
    publications = {}
    for source in census:
        name = source["dataset"]
        if name == "sharpa":
            continue
        expected = [g for g in source["groups"] if g["frames"] > 0]
        source_root = root / name
        source_root.mkdir(parents=True, exist_ok=True)
        completed = list(source_root.glob("*/meta/conversion_complete.json"))
        valid = [v for k, v in done.items() if Path(k).parent == source_root]
        local_errors = {k: v for k, v in errors.items() if Path(k).parent == source_root}
        expected_buckets = {}
        for group in expected:
            task = group["group"].split("/", 1)[1].removesuffix(".zarr")
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task).strip("_")
            key = str(source_root / (safe + "-" + hashlib.sha256(task.encode()).hexdigest()[:8]))
            expected_buckets[key] = group
        for value in valid:
            group = expected_buckets.get(value["bucket"])
            if group is None:
                local_errors[value["bucket"]] = "Validated bucket is absent from the release census"
            elif value["frames"] != group["frames"] or value["episodes"] != len(group["episode_ends"]):
                local_errors[value["bucket"]] = "Validated bucket frame/episode counts differ from release census"
        frames = sum(v["frames"] for v in valid)
        episodes = sum(v["episodes"] for v in valid)
        all_done = (
            {v["bucket"] for v in valid} == set(expected_buckets)
            and frames == source["frames"]
            and episodes == source["episodes"]
            and not local_errors
        )
        time_valid = all(v["physical_time_verified"] for v in valid)
        ready = all_done and time_valid and not local_errors
        publication = {
            "schema_version": "touchwam.ftp1.publication.v1",
            "source_id": "ftp1/" + name,
            "source": name,
            "status": "ready"
            if ready
            else "blocked_time"
            if all_done and not time_valid
            else "validating"
            if len(completed) == len(expected)
            else "converting",
            "ready": ready,
            "training_ready": ready,
            "complete": all_done,
            "expected_groups": len(expected),
            "empty_groups": len(source["groups"]) - len(expected),
            "completed_groups": len(completed),
            "validated_groups": len(valid),
            "expected_frames": source["frames"],
            "expected_episodes": source["episodes"],
            "frames": frames,
            "episodes": episodes,
            "physical_time_verified": time_valid if valid else False,
            "tactile_enabled": False,
            "tactile_status": "raw_preserved_calibration_or_representation_required",
            "training_modalities": ["video"],
            "action_enabled": False,
            "errors": local_errors,
            "buckets": sorted(v["bucket"] for v in valid),
            "updated_at": time.time(),
        }
        atomic_json(source_root / "publication.json", publication)
        publications[name] = publication
    atomic_json(
        root / "publication.json",
        {
            "schema_version": "touchwam.ftp1.publication.v1",
            "excluded_sources": ["sharpa"],
            "ready": all(v["ready"] for v in publications.values()),
            "sources": publications,
            "updated_at": time.time(),
        },
    )
    return publications


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--census", required=True)
    args = p.parse_args()
    root = Path(args.root)
    census = json.loads(Path(args.census).read_text())
    done = {}
    errors = {}
    while True:
        buckets = [
            x.parent.parent for x in root.glob("*/*/meta/conversion_complete.json") if str(x.parent.parent) not in done
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(validate_bucket, b, Path(args.census).parent / "regions"): str(b) for b in buckets}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    done[name] = future.result()
                    errors.pop(name, None)
                except Exception as exc:
                    errors[name] = repr(exc)
                    print(json.dumps({"event": "validation_error", "bucket": name, "error": repr(exc)}), flush=True)
                atomic_json(
                    root / "validation_progress.json",
                    {
                        "validated_buckets": len(done),
                        "frames": sum(x["frames"] for x in done.values()),
                        "errors": errors,
                    },
                )
        publications = publish_sources(root, census, done, errors)
        # A pilot catalog must not end a concurrently running complete-release validation.
        full_completed = all(source["complete"] for source in publications.values())
        if not args.watch or full_completed:
            atomic_json(
                root / "validation_summary.json",
                {
                    "complete": full_completed and not errors,
                    "validated_buckets": len(done),
                    "frames": sum(x["frames"] for x in done.values()),
                    "errors": errors,
                },
            )
            if errors:
                raise SystemExit(1)
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
