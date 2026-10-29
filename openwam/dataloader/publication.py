"""Fail closed on incomplete, changed, or ambiguous data publications."""
import hashlib
import json
from pathlib import Path


def source_publication_contract(config):
    from openwam.dataloader.utils import get_cfg
    record = publication_contract(get_cfg(config, "dataset_dir"))
    tactile_root = get_cfg(config, "tactile_root")
    if tactile_root:
        tactile = publication_contract(Path(tactile_root) / "buckets")
        if any(record[k] != tactile[k] for k in ("episodes", "frames")):
            raise ValueError("External tactile publication and reused source populations differ")
        tactile["binding"] = external_sharpa_binding(get_cfg(config, "dataset_dir"), tactile_root, record, tactile)
        record["tactile"] = tactile
    observation_root = get_cfg(config, "tactile_observation_root")
    if observation_root:
        if not tactile_root:
            raise ValueError("Sharpa video history requires its bound tactile label publication")
        record["tactile_observation"] = sharpa_observation_contract(
            observation_root, get_cfg(config, "dataset_dir"), tactile_root, record)
    return record


def sharpa_observation_contract(observation_root, source_root, tactile_root, source):
    """Bind native video history to exactly the existing source and label cache."""
    from openwam.tactile.sharpa_video_history import SCHEMA

    root = Path(observation_root).resolve(strict=True)
    path = root / "publication.json"
    record = json.loads(path.read_text())
    if (record.get("schema") != SCHEMA or record.get("status") != "complete"
            or record.get("training_ready") is not True
            or record.get("source_manifest_sha256") != source["sha256"]
            or record.get("tactile_catalog_sha256") != source["tactile"]["binding"]["catalog_sha256"]
            or any(record.get(k) != source[k] for k in ("episodes", "frames"))
            or record.get("validation_sha256") != _file_sha256(root / "validation.json")):
        raise ValueError("Sharpa native history publication is incomplete or belongs to different data")
    entries = record.get("buckets", {})
    if set(entries) != set(source["buckets"]):
        raise ValueError("Sharpa native history publication bucket population differs")
    for name, entry in entries.items():
        bucket = (Path(source_root) / name).resolve(strict=True)
        relative = Path("partitions") / bucket.parent.name / bucket.name
        manifest = root / relative / "observation.json"
        if not manifest.resolve(strict=True).is_relative_to(root) or _file_sha256(manifest) != entry.get("sha256"):
            raise ValueError(f"Sharpa native history manifest changed: {name}")
        doc = json.loads(manifest.read_text())
        if (doc.get("dataset_info_sha256") != _file_sha256(bucket / "meta/info.json")
                or doc.get("tactile_manifest_sha256") != _file_sha256(Path(tactile_root) / relative / "tactile/manifest.json")):
            raise ValueError(f"Sharpa native history metadata binding changed: {name}")
    return {"path": str(path), "sha256": _file_sha256(path), "buckets_verified": len(entries),
            "storage": SCHEMA, "observation_shape": [224, 224]}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_sharpa_binding(source_root, tactile_root, source, tactile):
    """Verify the exact source/cache identities recorded by the all-bucket audit.

    Counts alone cannot detect a cache from a different run with equal lengths.
    This checks metadata and manifest content, not every map payload. Run the
    catalog validator before launch for array inventory/clock/mask acceptance.
    """
    root, cache = Path(source_root), Path(tactile_root)
    publication = json.loads(Path(tactile["path"]).read_text())
    path = cache / "tactile_catalog.json"
    if (publication.get("schema") != "touchwam_sharpa_external_tactile_catalog_v1"
            or publication.get("tactile_catalog_sha256") != _file_sha256(path)):
        raise ValueError("External tactile catalog is missing, changed or unverified")
    catalog = json.loads(path.read_text())
    if (catalog.get("status") != "complete" or catalog.get("pending")
            or catalog.get("source_manifest_sha256") != source["sha256"]):
        raise ValueError("External tactile catalog belongs to a different source publication")
    entries = catalog.get("bucket_manifest_by_name", {})
    if set(entries) != set(source["buckets"]) or set(entries) != set(tactile["buckets"]):
        raise ValueError("External tactile catalog bucket identities differ from the source")
    for name, entry in entries.items():
        bucket = root / name
        if _file_sha256(bucket / "meta/info.json") != entry.get("source_info_sha256"):
            raise ValueError(f"Original Sharpa bucket metadata changed: {name}")
        episode_files = sorted(bucket.glob("meta/episodes/*/*.parquet"))
        episode_digest = hashlib.sha256(json.dumps([
            (str(p.relative_to(bucket)), _file_sha256(p)) for p in episode_files
        ]).encode()).hexdigest()
        if not episode_files or episode_digest != entry.get("source_episode_metadata_sha256"):
            raise ValueError(f"Original Sharpa episode identities or video offsets changed: {name}")
        # Use the actual reader path. Do not trust an arbitrary catalog pointer.
        partition = bucket.resolve(strict=True)
        manifest = cache / "partitions" / partition.parent.name / partition.name / "tactile/manifest.json"
        alias = cache / "buckets" / name / "tactile/manifest.json"
        if (alias.resolve(strict=True) != manifest.resolve(strict=True)
                or _file_sha256(manifest) != entry.get("manifest_sha256")):
            raise ValueError(f"External Sharpa tactile manifest changed or misbound: {name}")
    return {"catalog_sha256": _file_sha256(path), "source_manifest_sha256": source["sha256"],
            "buckets_verified": len(entries),
            "scope": "source publication, bucket/episode metadata and tactile manifest hashes; array payload audited separately"}


def publication_contract(dataset_dir):
    root = Path(dataset_dir).absolute()
    candidates = [root / "publication.json"]
    # Sharpa's public bucket aliases are one level below its release record.
    if root.name == "buckets":
        candidates.append(root.parent / "publication.json")
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return existing_publication_contract(root)
    raw = path.read_bytes()
    record = json.loads(raw)
    if (record.get("status") != "complete" or record.get("ready") is False
            or record.get("training_ready") is False or record.get("errors")):
        raise ValueError(f"Data publication is not complete and verified: {path}")
    for expected_key, done_keys in (
        ("buckets_expected", ("buckets_completed",)),
        ("expected_groups", ("completed_groups", "validated_groups")),
    ):
        if expected_key in record:
            expected = int(record[expected_key])
            if expected < 1 or any(int(record.get(k, -1)) != expected for k in done_keys):
                raise ValueError(f"Incomplete publication population: {path}")
    if (root / "meta/info.json").is_file():
        infos = [root / "meta/info.json"]
    elif root.name == "buckets":
        infos = sorted(p / "meta/info.json" for p in root.iterdir()
                       if p.is_dir() and (p / "meta/info.json").is_file())
    else:
        infos = sorted(root.rglob("meta/info.json"))
    expected = record.get("buckets_expected", record.get("expected_groups"))
    if expected is not None and len(infos) != int(expected):
        raise ValueError(f"Published population changed: expected {expected}, found {len(infos)} in {root}")
    if not infos:
        raise ValueError(f"Empty data publication: {root}")
    metadata = [json.loads(p.read_text()) for p in infos]
    for key, info_key in (("episodes", "total_episodes"), ("frames", "total_frames")):
        if key in record and sum(int(m[info_key]) for m in metadata) != int(record[key]):
            raise ValueError(f"Published {key} count no longer matches metadata: {root}")
    if any(m.get("codebase_version") != "v3.0" for m in metadata):
        raise ValueError(f"Publication contains a non-v3 bucket: {root}")
    if record.get("schema") == "touchwam_ftp1_frozen_training_v1":
        _validate_frozen_ftp_publication(root, record, infos)
    if record.get("schema") == "touchscale_lerobot_v3_v1":
        release = path.parent
        if _file_sha256(release / "conversion_plan.json") != record.get("conversion_plan_sha256"):
            raise ValueError("TouchScale conversion plan changed")
        for info in infos:
            bucket = info.parent.parent
            bound = record.get("buckets", {}).get(bucket.name, {})
            for rel, key in (("meta/info.json", "info_sha256"), ("tactile/manifest.json", "tactile_sha256"),
                             ("validation.json", "validation_sha256")):
                if _file_sha256(bucket / rel) != bound.get(key):
                    raise ValueError(f"TouchScale published identity changed: {bucket.name}/{rel}")
            validation = json.loads((bucket / "validation.json").read_text())
            for rel, entry in validation["files"].items():
                if rel.startswith("meta/") and _file_sha256(bucket / rel) != entry["sha256"]:
                    raise ValueError(f"TouchScale episode/source metadata changed: {rel}")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "buckets": [str(p.parent.parent.relative_to(root)) for p in infos],
            "episodes": sum(int(m["total_episodes"]) for m in metadata),
            "frames": sum(int(m["total_frames"]) for m in metadata)}


def _validate_frozen_ftp_publication(root, record, infos):
    """Bind a frozen FTP source to its snapshot, statistics and reader metadata."""
    snapshot_path = root.parent / "snapshot.json"
    inventory_path = root.parent / "publication_inventory.json"
    if (_file_sha256(snapshot_path) != record.get("snapshot_sha256")
            or _file_sha256(inventory_path) != record.get("publication_inventory_sha256")):
        raise ValueError("Frozen FTP snapshot/publication inventory changed")
    source = record.get("source")
    snapshot = json.loads(snapshot_path.read_text())
    if source != root.name or source not in snapshot.get("sources", {}):
        raise ValueError("Frozen FTP source identity differs from its snapshot")
    selected = {entry["bucket"] for entry in snapshot["sources"][source]["buckets_snapshot"]}
    actual = {p.parent.parent.relative_to(root).as_posix() for p in infos}
    if selected != actual:
        raise ValueError("Frozen FTP bucket population differs from its snapshot")
    inventory = [entry for entry in json.loads(inventory_path.read_text()) if entry["source"] == source]
    if len(inventory) != len(selected) or {entry["bucket"] for entry in inventory} != selected:
        raise ValueError("Frozen FTP original publication inventory population differs")
    for entry in inventory:
        provenance = root.parent / "provenance" / source / entry["bucket"] / "_PFS_PUBLICATION.json"
        if _file_sha256(provenance) != entry["publication_sha256"]:
            raise ValueError("Frozen FTP original publication provenance changed")
    files = {"normalization_stats.npy", "normalization_complete.json", "reader_validation.json"}
    for bucket in selected:
        required = {"info.json", "action_contract.json", "source_definition.json"}
        if record.get("training_modality") == "video_action_tactile":
            required.add("tactile_rgb.json")
        if any(not (root / bucket / "meta" / relative).is_file() for relative in required):
            raise ValueError("Frozen FTP source lacks required action/tactile metadata")
        files.update(p.relative_to(root).as_posix() for p in (root / bucket / "meta").rglob("*") if p.is_file())
    bound = record.get("metadata_sha256", {})
    if not isinstance(bound, dict) or files != set(bound):
        raise ValueError("Frozen FTP bound metadata population changed")
    for relative, fingerprint in bound.items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or _file_sha256(path) != fingerprint:
            raise ValueError(f"Frozen FTP metadata changed: {relative}")
    if (bound["normalization_stats.npy"] != record.get("normalization_sha256")
            or bound["reader_validation.json"] != record.get("reader_validation_sha256")):
        raise ValueError("Frozen FTP normalization/acceptance identity differs")


def existing_publication_contract(root):
    """Reuse previously accepted releases, without copying or rewriting them."""
    root = Path(root)
    info = root / "dataset_info.json"
    if info.is_file():
        record = json.loads(info.read_text())
        if (record.get("format") != "openwam_egodex_video_v1"
                or record.get("training_ready") is not True
                or record.get("source_split") != "train"):
            raise ValueError(f"Unverified existing EgoDex publication: {info}")
        index = root / "episodes.parquet"
        if hashlib.sha256(index.read_bytes()).hexdigest() != record["index_sha256"]:
            raise ValueError("Existing EgoDex episode index has changed")
        return {"path": str(info), "sha256": hashlib.sha256(info.read_bytes()).hexdigest(),
                "reused_existing_data": True, "format": record["format"], "buckets": ["."],
                "episodes": record["total_episodes"], "frames": record["total_frames"]}
    base = root.parent if root.name == "buckets" else root
    manifest = base / "dataset_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing completed publication record under {root}")
    record = json.loads(manifest.read_text())
    if record.get("version") == "sharpadex_native_tcp_v1_20261026":
        if record.get("status") != "data_and_reader_validated":
            raise ValueError("Native Sharpa release has not passed its prior validation")
        expected = int(record["partitions"])
        normalizer = base / "normalization_stats.npy"
        if hashlib.sha256(normalizer.read_bytes()).hexdigest() != record["normalization_sha256"]:
            raise ValueError("Existing Sharpa normalization changed")
    elif record.get("dataset_id") == "agibot_pretrain_v1":
        if record.get("training_ready") is not True:
            raise ValueError("Existing AgiBot release is not trainable")
        expected = int(record["tasks"])
        for rel, expected_hash in record.get("artifacts", {}).items():
            if hashlib.sha256((base / rel).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Existing AgiBot artifact changed: {rel}")
    else:
        raise ValueError(f"Unsupported existing publication contract: {manifest}")
    infos = sorted(p / "meta/info.json" for p in root.iterdir()
                   if p.is_dir() and (p / "meta/info.json").is_file())
    records = [json.loads(p.read_text()) for p in infos]
    if len(records) != expected or any(r.get("codebase_version") != "v3.0" for r in records):
        raise ValueError("Existing native publication population changed")
    episodes = sum(int(r["total_episodes"]) for r in records)
    frames = sum(int(r["total_frames"]) for r in records)
    if episodes != int(record["episodes"]) or frames != int(record["frames"]):
        raise ValueError("Existing native publication frame/episode counts changed")
    return {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "reused_existing_data": True, "buckets": [p.parent.parent.name for p in infos],
            "episodes": episodes, "frames": frames}
