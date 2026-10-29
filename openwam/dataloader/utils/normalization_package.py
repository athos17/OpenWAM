"""Versioned, independently verifiable normalization assets for mixed sources."""
import hashlib
import io
import json
from pathlib import Path
import shutil

import numpy as np

MANIFEST_NAME = "normalization_sources.json"
SCHEMA_VERSION = 1


def _payload_bytes(source):
    """Serialize a reader's resolved transform without writing into its dataset."""
    payload = source.get("stats_payload")
    if payload is None:
        return None
    if source.get("stats_path"):
        raise ValueError("A normalization source cannot provide both a path and a payload")
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Normalization payload must be a non-empty mode dictionary")
    buffer = io.BytesIO()
    np.save(buffer, payload, allow_pickle=True)
    return buffer.getvalue()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_normalization_manifest(sources):
    """Validate deployment statistics and describe them without writing files.

    Preflight uses the same checks as checkpoint publication, so missing or
    changed source statistics fail before allocating the full training model.
    """
    entries = {}
    for name, source in sources.items():
        config = source["config"]
        active = config.get("normalize_mode") not in (None, "", "none", "null")
        stats = source.get("stats_path")
        payload = _payload_bytes(source)
        has_stats = bool(stats) or payload is not None
        if active and not has_stats:
            raise ValueError(f"Normalized source {name!r} has no statistics artifact")
        if active and not config.get("action_mode"):
            raise ValueError(f"Normalized source {name!r} requires an explicit action_mode")
        relative = f"normalization/{hashlib.sha256(name.encode()).hexdigest()[:16]}/normalization_stats.npy" if has_stats else None
        checksum = hashlib.sha256(payload).hexdigest() if payload is not None else file_sha256(stats) if stats else None
        entries[name] = {"config": config, "training_modality": source["training_modality"],
                         "stats_file": relative, "stats_sha256": checksum}
    return {"schema_version": SCHEMA_VERSION, "sources": entries}


def read_normalization_package(output_dir):
    root = Path(output_dir)
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Mixed checkpoint is missing {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or not isinstance(manifest.get("sources"), dict):
        raise ValueError("Invalid mixed normalization package schema")
    for name, source in manifest["sources"].items():
        relative = source["stats_file"]
        if relative is None:
            if source["stats_sha256"] is not None or source["config"].get("normalize_mode") not in (None, "", "none", "null"):
                raise ValueError(f"Missing statistics for source {name!r}")
            continue
        path = root / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Statistics path escapes checkpoint: {relative}")
        if not path.is_file() or file_sha256(path) != source["stats_sha256"]:
            raise ValueError(f"Missing or changed normalization artifact for source {name!r}: {path}")
    return manifest


def verify_normalization_package(output_dir, sources):
    saved = read_normalization_package(output_dir)
    expected = build_normalization_manifest(sources)
    if saved != expected:
        raise ValueError("Mixed normalization sources differ: names, transforms, modality or source configuration changed")


def save_normalization_package(output_dir, sources):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_normalization_manifest(sources)
    path = root / MANIFEST_NAME
    if path.exists():
        verify_normalization_package(root, sources)
        return
    for name, source in sources.items():
        entry = manifest["sources"][name]
        if entry["stats_file"] is None:
            continue
        target = root / entry["stats_file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) != entry["stats_sha256"]:
            raise ValueError(f"Existing source artifact differs: {target}")
        temporary = target.with_suffix(".npy.tmp")
        payload = _payload_bytes(source)
        if payload is None:
            shutil.copyfile(source["stats_path"], temporary)
        else:
            temporary.write_bytes(payload)
        if file_sha256(temporary) != entry["stats_sha256"]:
            raise ValueError(f"Source statistics changed while saving {name!r}")
        temporary.replace(target)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
