"""Immutable episode selections over published LeRobot data, without media copies."""

import hashlib
import json
from pathlib import Path


def load_episode_selection(path, dataset_root, split, expected_sha256):
    """Bind an explicit population and its metadata to the training config."""
    path, root = Path(path), Path(dataset_root).resolve(strict=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not expected_sha256 or digest != expected_sha256:
        raise ValueError("Frozen episode selection SHA256 does not match the recipe")
    record = json.loads(raw)
    if record.get("schema") != "touchwam_episode_selection_v1" or record.get("split") != split:
        raise ValueError("Frozen episode selection schema or split does not match")
    if Path(record["dataset_root"]).resolve(strict=True) != root:
        raise ValueError("Frozen episode selection belongs to a different dataset root")
    buckets = record.get("buckets", {})
    if not buckets:
        raise ValueError("Frozen episode selection is empty")
    for name, entry in buckets.items():
        bucket = (root / name).resolve(strict=True)
        if not bucket.is_relative_to(root):
            raise ValueError("Frozen episode selection escapes its dataset root")
        ids = entry.get("episode_indices", [])
        if not ids or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Frozen episode selection needs unique non-negative episode IDs")
        metadata = entry.get("metadata_sha256", {})
        expected_paths = {"meta/info.json"} | {
            str(p.relative_to(bucket)) for p in (bucket / "meta/episodes").rglob("*.parquet")}
        if not expected_paths.issubset(metadata):
            raise ValueError("Frozen episode selection does not bind all episode metadata")
        for relative, expected in metadata.items():
            file = (bucket / relative).resolve(strict=True)
            if not file.is_relative_to(bucket) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen episode metadata changed: {name}/{relative}")
    return record, {"path": str(path.absolute()), "sha256": digest}
