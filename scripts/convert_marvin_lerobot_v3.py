#!/usr/bin/env python3
"""Convert a Marvin Pro LeRobot v2.1 dataset to OpenWAM-compatible v3.

The source dataset is never modified. Data parquet files are rewritten because
the legacy rot6d values use the first two matrix rows; videos are hard-linked
into the v3 layout (with a copy fallback for cross-filesystem output paths).

The input path is the only required argument. By default, a source directory
named ``pick_place_2`` is written beside it as ``pick_place_2_openwam_v3``::

    python scripts/convert_marvin_lerobot_v3.py ../datasets/pick_place_2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openwam.dataloader.marvin import (  # noqa: E402
    LEFT_EEF,
    OPENWAM_ROT6D_CONVENTION,
    OPENWAM_ROT6D_LAYOUT,
    RAW_ACTION_DIM,
    RIGHT_EEF,
    ROT6D_RAW_DIMS,
    marvin_raw_vector,
    rot6d_rows_to_columns,
)
from openwam.dataloader.utils.normalization import STAT_KEYS  # noqa: E402

SOURCE_ROT6D_LAYOUT = "[R00,R01,R02,R10,R11,R12]"
CAMERAS = (
    "observation.images.left_head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)

EXPECTED_SOURCE_NAMES = {
    "action": {
        "left_hand_prefix": "hand_left.joint_commands.left_",
        "right_hand_prefix": "hand_right.joint_commands.right_",
        "left_eef_prefix": "tj.control.eef_left.",
        "right_eef_prefix": "tj.control.eef_right.",
    },
    "observation.state": {
        "left_hand_prefix": "hand_left.joint_states.left_",
        "right_hand_prefix": "hand_right.joint_states.right_",
        "left_eef_prefix": "tj.info.eef_left.",
        "right_eef_prefix": "tj.info.eef_right.",
    },
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _convert_source_vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != 76:
        raise ValueError(f"expected source vectors with shape (N, 76), got {values.shape}")
    for block in (LEFT_EEF, RIGHT_EEF):
        rot = slice(block.start + 3, block.stop)
        values[:, rot] = rot6d_rows_to_columns(values[:, rot])
    return values


def _replace_list_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"parquet table is missing {name!r}")
    original_type = table.schema.field(index).type
    replacement = pa.array(values.tolist(), type=original_type)
    return table.set_column(index, table.schema.field(index), replacement)


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    values64 = np.asarray(values, dtype=np.float64)
    result = {
        "mean": values64.mean(axis=0),
        "std": values64.std(axis=0),
        "min": values64.min(axis=0),
        "max": values64.max(axis=0),
        "q01": np.quantile(values64, 0.01, axis=0),
        "q99": np.quantile(values64, 0.99, axis=0),
    }
    for key, identity in {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}.items():
        result[key][list(ROT6D_RAW_DIMS)] = identity
    return {key: np.asarray(result[key], dtype=np.float32).tolist() for key in STAT_KEYS}


def _update_feature_names(info: dict) -> None:
    rot_names = ["r00", "r10", "r20", "r01", "r11", "r21"]
    for feature, prefix_left, prefix_right in (
        ("action", "tj.control.eef_left", "tj.control.eef_right"),
        ("observation.state", "tj.info.eef_left", "tj.info.eef_right"),
    ):
        names = info["features"][feature]["names"]
        if len(names) != 76:
            raise ValueError(f"{feature} must expose 76 names, got {len(names)}")
        names[61:67] = [f"{prefix_left}.{suffix}" for suffix in rot_names]
        names[70:76] = [f"{prefix_right}.{suffix}" for suffix in rot_names]


def _validate_feature_schema(info: dict) -> None:
    features = info.get("features", {})
    for feature, expected in EXPECTED_SOURCE_NAMES.items():
        spec = features.get(feature, {})
        if spec.get("shape") != [76]:
            raise ValueError(f"{feature} must have shape [76], got {spec.get('shape')!r}")
        names = spec.get("names")
        if not isinstance(names, list) or len(names) != 76:
            raise ValueError(f"{feature} must expose 76 named dimensions")
        checks = (
            (range(14, 36), expected["left_hand_prefix"], "left hand"),
            (range(36, 58), expected["right_hand_prefix"], "right hand"),
            (range(58, 67), expected["left_eef_prefix"], "left EEF"),
            (range(67, 76), expected["right_eef_prefix"], "right EEF"),
        )
        for indices, prefix, label in checks:
            bad = [index for index in indices if not names[index].startswith(prefix)]
            if bad:
                raise ValueError(
                    f"{feature} does not match the Marvin {label} layout at dimensions {bad}: "
                    f"expected names starting with {prefix!r}"
                )

    for camera in CAMERAS:
        if features.get(camera, {}).get("dtype") != "video":
            raise ValueError(f"required Marvin camera {camera!r} is missing or is not a video feature")


def _source_rot6d_provenance(source: Path) -> list[tuple[Path, str]]:
    candidates = (
        (source / "meta" / "rotation6d_migration.json", ("target_layout",)),
        (source / "meta" / "eef_action_backfill.json", ("rotation_6d_layout",)),
        (source / "meta" / "marvin_conversion.json", ("eef_action", "rotation_6d_layout")),
        # Merged datasets record the shared layout at the top level. Check
        # both locations so contradictory markers are still rejected.
        (source / "meta" / "marvin_conversion.json", ("rotation_6d_layout",)),
    )
    found: list[tuple[Path, str]] = []
    for path, keys in candidates:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            found.append((path, str(value)))
    return found


def _validate_source(source: Path) -> tuple[dict, list[dict], list[dict]]:
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"expected LeRobot v2.1 source, got {info.get('codebase_version')!r}")
    if info.get("robot_type") != "marvin_pro_sharpa_wave":
        raise ValueError(
            "this converter only supports the Marvin Pro contract; "
            f"got robot_type={info.get('robot_type')!r}"
        )
    _validate_feature_schema(info)

    provenance = _source_rot6d_provenance(source)
    if not provenance:
        raise ValueError(
            "cannot verify the source rot6d convention: expected a layout marker in "
            "rotation6d_migration.json, eef_action_backfill.json, or marvin_conversion.json"
        )
    incompatible = [(str(path), layout) for path, layout in provenance if layout != SOURCE_ROT6D_LAYOUT]
    if incompatible:
        raise ValueError(
            f"source rot6d provenance must be {SOURCE_ROT6D_LAYOUT}; incompatible markers: {incompatible}"
        )

    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    tasks = _read_jsonl(source / "meta" / "tasks.jsonl")
    if not episodes:
        raise ValueError("source dataset has no episodes")
    if len(episodes) != int(info.get("total_episodes", -1)):
        raise ValueError(
            f"episodes.jsonl has {len(episodes)} rows but info.json declares "
            f"total_episodes={info.get('total_episodes')!r}"
        )
    expected_indices = list(range(len(episodes)))
    actual_indices = [int(episode.get("episode_index", -1)) for episode in episodes]
    if actual_indices != expected_indices:
        raise ValueError("episode indices must be ordered and contiguous from 0")
    total_frames = sum(int(episode.get("length", -1)) for episode in episodes)
    if total_frames != int(info.get("total_frames", -1)):
        raise ValueError(
            f"episodes.jsonl sums to {total_frames} frames but info.json declares "
            f"total_frames={info.get('total_frames')!r}"
        )
    if not tasks or len(tasks) != int(info.get("total_tasks", -1)):
        raise ValueError(
            f"tasks.jsonl has {len(tasks)} rows but info.json declares total_tasks={info.get('total_tasks')!r}"
        )
    task_indices = {int(task["task_index"]) for task in tasks}
    unknown_tasks = sorted({int(episode["task_index"]) for episode in episodes} - task_indices)
    if unknown_tasks:
        raise ValueError(f"episodes reference task indices absent from tasks.jsonl: {unknown_tasks}")
    return info, episodes, tasks


def convert(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    info, episodes, tasks = _validate_source(source)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    action_chunks: list[np.ndarray] = []
    state_chunks: list[np.ndarray] = []
    episode_rows: list[dict] = []
    video_modes: set[str] = set()
    dataset_offset = 0

    try:
        chunks_size = int(info.get("chunks_size", 1000))
        if chunks_size <= 0:
            raise ValueError(f"info.json chunks_size must be positive, got {chunks_size}")
        for position, episode in enumerate(episodes):
            episode_index = int(episode["episode_index"])
            if episode_index != position:
                raise ValueError(f"episode indices must be contiguous; position {position} has {episode_index}")
            source_chunk = episode_index // chunks_size
            target_chunk = episode_index // chunks_size
            target_file = episode_index % chunks_size
            source_data = source / info["data_path"].format(
                episode_chunk=source_chunk,
                episode_index=episode_index,
            )
            table = pq.read_table(source_data)
            length = int(episode["length"])
            if table.num_rows != length:
                raise ValueError(f"episode {episode_index}: metadata length {length} != parquet rows {table.num_rows}")

            action = _convert_source_vectors(np.asarray(table["action"].to_pylist(), dtype=np.float32))
            state = _convert_source_vectors(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
            table = _replace_list_column(table, "action", action)
            table = _replace_list_column(table, "observation.state", state)

            data_destination = temp / f"data/chunk-{target_chunk:03d}/file-{target_file:03d}.parquet"
            data_destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_destination, compression="zstd")

            action_chunks.append(marvin_raw_vector(action))
            state_chunks.append(marvin_raw_vector(state))

            row = {
                "episode_index": episode_index,
                "tasks": episode.get("tasks", []),
                "length": length,
                "dataset_from_index": dataset_offset,
                "dataset_to_index": dataset_offset + length,
                "data/chunk_index": target_chunk,
                "data/file_index": target_file,
            }
            for camera in CAMERAS:
                source_video = source / info["video_path"].format(
                    episode_chunk=source_chunk,
                    video_key=camera,
                    episode_index=episode_index,
                )
                destination_video = temp / f"videos/{camera}/chunk-{target_chunk:03d}/file-{target_file:03d}.mp4"
                video_modes.add(_link_or_copy(source_video, destination_video))
                row[f"videos/{camera}/chunk_index"] = target_chunk
                row[f"videos/{camera}/file_index"] = target_file
            episode_rows.append(row)
            dataset_offset += length

        episodes_path = temp / "meta/episodes/chunk-000/file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

        task_frame = pd.DataFrame(
            {"task_index": [int(task["task_index"]) for task in tasks]},
            index=pd.Index([str(task["task"]) for task in tasks], name="task"),
        )
        task_frame.to_parquet(temp / "meta/tasks.parquet")

        converted_info = json.loads(json.dumps(info))
        converted_info.update(
            {
                "codebase_version": "v3.0",
                "total_chunks": math.ceil(len(episodes) / chunks_size),
                "total_videos": len(episodes) * len(CAMERAS),
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "openwam_contract": {
                    "action_representation": "absolute_eef_and_absolute_hand_joints",
                    "action_pose_frame": "base_link",
                    "raw_action_dim": RAW_ACTION_DIM,
                    "unified_action_dim": 80,
                    "hand_dof_per_side": 22,
                    "rotation_6d_layout": OPENWAM_ROT6D_LAYOUT,
                    "rotation_6d_convention": OPENWAM_ROT6D_CONVENTION,
                },
            }
        )
        _update_feature_names(converted_info)
        _write_json(temp / "meta/info.json", converted_info)

        action_all = np.concatenate(action_chunks, axis=0)
        state_all = np.concatenate(state_chunks, axis=0)
        if len(action_all) != info["total_frames"] or len(state_all) != info["total_frames"]:
            raise ValueError("converted frame count does not match info.json")
        stats_payload = {"eef": _stats(action_all), "eef_state": _stats(state_all)}
        np.save(temp / "meta/normalization_stats.npy", stats_payload, allow_pickle=True)
        _write_json(
            temp / "meta/openwam_conversion.json",
            {
                "source": str(source),
                "source_dataset_name": source.name,
                "source_format": "lerobot_v2.1",
                "target_format": "lerobot_v3.0",
                "episodes": len(episodes),
                "frames": int(len(action_all)),
                "source_rotation_6d_layout": SOURCE_ROT6D_LAYOUT,
                "target_rotation_6d_layout": OPENWAM_ROT6D_LAYOUT,
                "target_rotation_6d_convention": OPENWAM_ROT6D_CONVENTION,
                "video_materialization": sorted(video_modes),
                "source_preserved": True,
            },
        )

        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="source LeRobot v2.1 dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory (default: <source>_openwam_v3 beside the source)",
    )
    args = parser.parse_args()
    output = args.output or args.source.with_name(f"{args.source.name}_openwam_v3")
    convert(args.source, output)
    print(f"Converted Marvin dataset: {output.resolve()}")


if __name__ == "__main__":
    main()
