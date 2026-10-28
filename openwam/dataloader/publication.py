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
        record["tactile"] = tactile
    return record


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
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "buckets": [str(p.parent.parent.relative_to(root)) for p in infos],
            "episodes": sum(int(m["total_episodes"]) for m in metadata),
            "frames": sum(int(m["total_frames"]) for m in metadata)}


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
