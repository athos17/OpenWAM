"""Versioned, independently verifiable normalization assets for mixed sources."""
import hashlib
import json
from pathlib import Path
import shutil

MANIFEST_NAME = "normalization_sources.json"
SCHEMA_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_for(sources):
    entries = {}
    for name, source in sources.items():
        config = source["config"]
        active = config.get("normalize_mode") not in (None, "", "none", "null")
        stats = source["stats_path"]
        if active and not stats:
            raise ValueError(f"Normalized source {name!r} has no statistics artifact")
        if active and not config.get("action_mode"):
            raise ValueError(f"Normalized source {name!r} requires an explicit action_mode")
        relative = f"normalization/{hashlib.sha256(name.encode()).hexdigest()[:16]}/normalization_stats.npy" if stats else None
        entries[name] = {"config": config, "training_modality": source["training_modality"],
                         "stats_file": relative, "stats_sha256": file_sha256(stats) if stats else None}
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
    expected = _manifest_for(sources)
    if saved != expected:
        raise ValueError("Mixed normalization sources differ: names, transforms, modality or source configuration changed")


def save_normalization_package(output_dir, sources):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_for(sources)
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
        shutil.copyfile(source["stats_path"], temporary)
        if file_sha256(temporary) != entry["stats_sha256"]:
            raise ValueError(f"Source statistics changed while saving {name!r}")
        temporary.replace(target)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
