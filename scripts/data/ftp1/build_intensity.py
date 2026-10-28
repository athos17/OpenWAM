#!/usr/bin/env python3
"""Build verified regional intensity sidecars from converted FTP-1 v3 data.

No source has an implicit RGB-to-pressure transform. Every adapter supplies
validated semantics, region identity, fixed normalization and baseline evidence.
Optical differences require a separately approved optical-change proxy contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def registry_names():
    # Read the registry without importing the GPU/model stack.
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "openwam/tactile/registry.py"
    import sys

    spec = importlib.util.spec_from_file_location("ftp1_region_registry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.REGION_NAMES


def validate_definition(definition, known_regions):
    if definition.get("schema") != "touchwam.ftp1.intensity_definition.v1":
        raise ValueError("Unsupported intensity adapter schema")
    verification = definition.get("verification", {})
    if verification.get("status") != "validated" or not verification.get("evidence"):
        raise ValueError("Intensity adapters require validation evidence; uncalibrated RGB is disabled")
    if not isinstance(verification["evidence"], list) or not all(
        isinstance(v, str) and v.strip() for v in verification["evidence"]
    ):
        raise ValueError("Validation evidence must be a nonempty list of references")
    regions = definition.get("regions", [])
    names = [row.get("region") for row in regions]
    if not names or len(set(names)) != len(names) or set(names) - set(known_regions):
        raise ValueError("Use unique canonical region identities; compose multiple sensors upstream")
    for row in regions:
        if not isinstance(row.get("sensor_name"), str) or not row["sensor_name"].strip():
            raise ValueError("Every region needs a sensor/source identity")
        mode = row.get("mode")
        if mode not in {
            "spatial_axis",
            "calibrated_gray_lut",
            "verified_video_channel",
            "optical_change_proxy",
            "depth_reconstruction",
        }:
            raise ValueError(f"Unsupported intensity conversion mode: {mode}")
        if mode == "optical_change_proxy":
            if (
                definition.get("semantics") != "optical_change_proxy"
                or verification.get("proxy_use_explicitly_approved") is not True
            ):
                raise ValueError("RGB difference is an optical proxy, not force; explicit proxy authorization required")
        elif definition.get("semantics") != "normal_contact_intensity":
            raise ValueError("Physical adapters must explicitly declare normal-contact semantics")
        bounds = np.asarray(row.get("normalization", {}).get("bounds", []), dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
            raise ValueError("Fixed finite normalization bounds are required; no per-frame rescaling")
        if row.get("sign", 1) not in (-1, 1):
            raise ValueError("Normal axis/sign must be explicit +1 or -1")
        baseline = row.get("baseline", {})
        if baseline.get("verified_no_contact") is not True or not baseline.get("evidence"):
            raise ValueError("Baseline needs verified no-contact evidence")
        if baseline.get("kind") not in {"constant", "npy"}:
            raise ValueError("Only fixed constant or fingerprinted NPY baselines are supported")
        if baseline["kind"] == "constant" and not np.isfinite(baseline.get("value", np.nan)):
            raise ValueError("Nonfinite baseline")
        if mode == "spatial_axis" and (
            not isinstance(row.get("spatial_shape"), list)
            or len(row["spatial_shape"]) != 2
            or min(row["spatial_shape"]) < 2
        ):
            raise ValueError("A measured spatial grid is required; scalar/wrench broadcasting is forbidden")
        if mode in {"calibrated_gray_lut", "verified_video_channel"} and not row.get("monotonic_response_evidence"):
            raise ValueError("Video-derived intensity requires monotonic response evidence")
        if mode == "depth_reconstruction":
            if row.get("reconstruction_output") != "indentation_depth" or not row.get("reconstruction_unit"):
                raise ValueError("Depth plugins must declare indentation-depth units; pressure is not assumed")
            if not row.get("reconstruction_plugin") or not row.get("monotonic_response_evidence"):
                raise ValueError("Depth reconstruction requires a verified plugin and response evidence")
        if mode == "calibrated_gray_lut" and not row.get("lookup_table"):
            raise ValueError("A sensor-specific calibrated LUT is required")
    return definition


def load_asset(spec, definition_dir):
    path = (definition_dir / spec["path"]).resolve(strict=True)
    if not spec.get("sha256") or digest(path) != spec["sha256"]:
        raise ValueError(f"Calibration asset fingerprint mismatch: {path}")
    value = np.load(path, allow_pickle=False)
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError("Calibration arrays must be finite numeric data")
    return value.astype(np.float32)


def load_depth_plugin(row, definition_dir):
    import importlib.util

    spec = row["reconstruction_plugin"]
    path = (definition_dir / spec["path"]).resolve(strict=True)
    if path.suffix != ".py" or digest(path) != spec.get("sha256"):
        raise ValueError("Depth plugin source fingerprint mismatch")
    module_spec = importlib.util.spec_from_file_location("ftp1_depth_" + digest(path)[:16], path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    function = getattr(module, spec.get("entrypoint", "reconstruct"), None)
    if not callable(function):
        raise ValueError("Depth plugin must provide callable reconstruct")
    calibration = {key: load_asset(asset, definition_dir) for key, asset in row.get("calibration_assets", {}).items()}
    return {"function": function, "calibration": calibration, "options": row.get("reconstruction_options", {})}


def assets_for(row, definition_dir):
    spec = row["baseline"]
    baseline = np.float32(spec["value"]) if spec["kind"] == "constant" else load_asset(spec, definition_dir)
    if row["mode"] == "depth_reconstruction":
        return baseline, load_depth_plugin(row, definition_dir)
    lut = load_asset(row["lookup_table"], definition_dir) if row.get("lookup_table") else None
    if lut is not None and (lut.ndim != 1 or len(lut) < 2 or (np.diff(lut) < 0).any()):
        raise ValueError("Calibration lookup must be a monotonic nondecreasing vector")
    return baseline, lut


def crop(value, row):
    if "roi_xywh" in row:
        x, y, width, height = row["roi_xywh"]
        if (
            any(type(v) is not int for v in [x, y, width, height])
            or min(x, y) < 0
            or min(width, height) < 1
            or y + height > value.shape[0]
            or x + width > value.shape[1]
        ):
            raise ValueError("ROI is outside the measured image")
        value = value[y : y + height, x : x + width]
    return value


def transform_frame(frame, row, baseline, lut=None, size=32):
    """One measured grid/image -> normalized map; never guesses sensor physics."""
    value = crop(np.asarray(frame, dtype=np.float32), row)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite observed tactile measurement")
    mode = row["mode"]
    if mode == "depth_reconstruction":
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("Depth reconstruction requires RGB input")
        shape = value.shape[:2]
        value = np.asarray(
            lut["function"](image=value, baseline=baseline, calibration=lut["calibration"], options=lut["options"]),
            dtype=np.float32,
        )
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Depth plugin must return finite, spatially registered indentation depth")
    elif mode == "spatial_axis":
        if value.shape != tuple(row["spatial_shape"]):
            raise ValueError("Spatial scalar shape mismatch; vectors cannot be painted into maps")
    elif mode == "verified_video_channel":
        if value.ndim != 3 or value.shape[-1] != 3 or row.get("channel") not in [0, 1, 2]:
            raise ValueError("Video channel requires explicit RGB channel 0,1,2")
        value = value[..., row["channel"]]
    elif mode == "calibrated_gray_lut":
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("Calibrated grayscale requires RGB input")
        value = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
    elif mode == "optical_change_proxy":
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("Optical proxy requires RGB input")
    if mode != "depth_reconstruction" and np.ndim(baseline) and baseline.shape != value.shape:
        raise ValueError(f"Baseline shape {baseline.shape} does not match ROI {value.shape}")
    if mode != "depth_reconstruction":
        value = row.get("sign", 1) * (value - baseline)
    if mode == "optical_change_proxy":
        value = np.abs(value).mean(axis=-1)
    if lut is not None and mode != "depth_reconstruction":
        value = np.interp(value, np.arange(len(lut)), lut)
    lower, upper = row["normalization"]["bounds"]
    value = np.clip((value - lower) / (upper - lower), 0, 1).astype(np.float32)
    value = np.rot90(value, k=int(row.get("rotate90", 0)))
    if row.get("flip_x", False):
        value = value[:, ::-1]
    if row.get("flip_y", False):
        value = value[::-1]
    value = cv2.resize(np.ascontiguousarray(value), (size, size), interpolation=cv2.INTER_AREA)
    return np.rint(value * 255).astype(np.uint8)


def numeric_frames(bucket, info, episodes, row):
    feature = row["feature"]
    path = bucket / info["data_path"].format(chunk_index=0, file_index=0)
    table = pq.read_table(path, columns=[feature, feature + ".valid"])
    spec = info["features"][feature]
    shape = spec.get("native_shape", spec["shape"])
    values = np.asarray(table[feature].to_pylist(), dtype=np.float32).reshape((len(table), *shape))
    valid = table[feature + ".valid"].to_numpy()
    if row["mode"] != "spatial_axis":
        raise ValueError("Numeric arrays require a measured spatial-axis adapter")
    if "sensor_index" in row:
        values = values[:, int(row["sensor_index"])]
    if "normal_axis" in row:
        values = values[..., int(row["normal_axis"])]
    for value, ok in zip(values, valid):
        yield value, bool(ok)


def video_frames(bucket, info, episodes, row):
    feature = row["feature"]
    table = pq.read_table(bucket / info["data_path"].format(chunk_index=0, file_index=0), columns=[feature + ".valid"])
    valid = table[feature + ".valid"].to_numpy()
    path = bucket / info["video_path"].format(video_key=feature, chunk_index=0, file_index=0)
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= len(valid):
                raise ValueError("Video exceeds v3 frame count")
            yield frame.to_ndarray(format="rgb24"), bool(valid[i])


def build_bucket(bucket, definition_path, known_regions=None):
    bucket, definition_path = Path(bucket), Path(definition_path)
    definition = validate_definition(json.loads(definition_path.read_text()), known_regions or registry_names())
    info_path = bucket / "meta/info.json"
    info = json.loads(info_path.read_text())
    source_path = bucket / "meta/source_definition.json"
    source = json.loads(source_path.read_text())
    if source["source"] != definition["source"]:
        raise ValueError("Adapter source does not match bucket")
    if not source.get("physical_time_verified") or not (bucket / "meta/validation.json").exists():
        raise ValueError("Validate source conversion and physical clock before intensity construction")
    rows = definition["regions"]
    for row in rows:
        if row["feature"] not in info["features"]:
            raise ValueError(f"Missing raw feature {row['feature']}")
    loaded_assets = [assets_for(row, definition_path.parent) for row in rows]
    signature = hashlib.sha256(
        json.dumps(
            [digest(info_path), digest(definition_path), digest(bucket / "meta/conversion_complete.json")]
        ).encode()
    ).hexdigest()
    root = bucket / "tactile"
    generation = root / "generations" / signature[:20]
    generation.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("adapter_signature") == signature:
            return previous
        raise ValueError("A different tactile generation is already published; use an explicit new output version")
    eps = pq.read_table(bucket / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    entries = {}
    region_count = len(rows)
    initialized = (generation / "initialized.json").exists()
    for ep in eps:
        i, length = ep["episode_index"], ep["length"]
        sub = generation / f"episode_{i:06d}"
        sub.mkdir(exist_ok=True)
        if not initialized:
            maps = np.lib.format.open_memmap(
                sub / "maps.npy", mode="w+", dtype=np.uint8, shape=(length, region_count, 32, 32)
            )
            maps[:] = 0
            maps.flush()
            del maps
            np.save(sub / "valid.npy", np.zeros((length, region_count), dtype=bool), allow_pickle=False)
            np.save(sub / "timestamps.npy", np.arange(length, dtype=np.float64) / info["fps"], allow_pickle=False)
        entries[str(i)] = {
            "maps": str((sub / "maps.npy").relative_to(root)),
            "valid": str((sub / "valid.npy").relative_to(root)),
            "timestamps": str((sub / "timestamps.npy").relative_to(root)),
            "layout_present": [True] * region_count,
        }
    if not initialized:
        atomic_json(
            generation / "initialized.json",
            {"adapter_signature": signature, "episodes": len(eps), "regions": region_count},
        )
    for region_index, (row, (baseline, lut)) in enumerate(zip(rows, loaded_assets)):
        region_receipt = generation / f"region_{region_index:03d}.complete.json"
        if region_receipt.exists():
            receipt = json.loads(region_receipt.read_text())
            if receipt.get("adapter_signature") != signature or receipt.get("frames") != info["total_frames"]:
                raise ValueError("Incompatible regional resume receipt")
            continue
        video = info["features"][row["feature"]]["dtype"] == "video"
        frames = video_frames(bucket, info, eps, row) if video else numeric_frames(bucket, info, eps, row)
        ep_index = 0
        current = None
        maps = valid_array = None
        count = 0
        try:
            for global_index, (frame, valid) in enumerate(frames):
                while global_index >= eps[ep_index]["dataset_to_index"]:
                    ep_index += 1
                ep = eps[ep_index]
                if current != ep_index:
                    if current is not None:
                        maps.flush()
                        valid_array.flush()
                        maps = valid_array = None
                    entry = entries[str(ep["episode_index"])]
                    maps = np.load(root / entry["maps"], mmap_mode="r+")
                    valid_array = np.load(root / entry["valid"], mmap_mode="r+")
                    current = ep_index
                local = global_index - ep["dataset_from_index"]
                if valid:
                    maps[local, region_index] = transform_frame(frame, row, baseline, lut)
                    valid_array[local, region_index] = True
                count += 1
        finally:
            if current is not None:
                maps.flush()
                valid_array.flush()
                maps = valid_array = None
        if count != info["total_frames"]:
            raise ValueError(f"Decoded {count} frames, expected {info['total_frames']}")
        atomic_json(region_receipt, {"adapter_signature": signature, "frames": count, "region": row["region"]})
    manifest = {
        "schema": "tactile_regions_v1",
        "codec": "tactile_intensity_v1",
        "dataset_info_sha256": digest(info_path),
        "adapter_signature": signature,
        "fps": info["fps"],
        "region_names": [r["region"] for r in rows],
        "sensor_names": [r["sensor_name"] for r in rows],
        "episodes": entries,
        "semantics": definition["semantics"],
        "provenance": {
            "source_id": source["source_id"],
            "adapter_definition": definition,
            "adapter_definition_sha256": digest(definition_path),
            "source_conversion_sha256": digest(bucket / "meta/conversion_complete.json"),
            "normalization": "fixed_per_sensor_bounds_no_per_frame_scaling",
        },
    }
    # Check array contracts before making the manifest visible to the reader.
    for entry in entries.values():
        maps = np.load(root / entry["maps"], mmap_mode="r")
        valid = np.load(root / entry["valid"], mmap_mode="r")
        times = np.load(root / entry["timestamps"], mmap_mode="r")
        if (
            maps.dtype != np.uint8
            or valid.dtype != np.bool_
            or valid.shape != maps.shape[:2]
            or (np.diff(times) <= 0).any()
        ):
            raise ValueError("Invalid generated intensity contract")
    atomic_json(manifest_path, manifest)
    atomic_json(
        bucket / "meta/tactile_conversion_status.json",
        {
            "enabled": True,
            "codec": "tactile_intensity_v1",
            "semantics": definition["semantics"],
            "adapter_signature": signature,
            "manifest_sha256": digest(manifest_path),
        },
    )
    source["intensity_enabled"] = True
    source["training_modalities"] = sorted(set(source.get("training_modalities", [])) | {"tactile"})
    source["intensity_semantics"] = definition["semantics"]
    atomic_json(source_path, source)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--definition", required=True)
    args = parser.parse_args()
    result = build_bucket(args.bucket, args.definition)
    print(
        json.dumps(
            {
                "bucket": args.bucket,
                "episodes": len(result["episodes"]),
                "regions": result["region_names"],
                "semantics": result["semantics"],
            }
        )
    )


if __name__ == "__main__":
    main()
