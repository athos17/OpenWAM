#!/usr/bin/env python3
"""Freeze a completed FTP snapshot without copying media or changing its producer.

Only the explicit snapshot population is admitted. Published media/parquet are
hardlinked, small metadata is copied, and source normalization is computed from
the frozen train rows. Sources without verified canonical action profiles remain
archival. Final publication requires real production-reader/collator acceptance.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCHEMA = "touchwam_ftp1_frozen_training_v1"
IDENTITY = dict(min=-1., max=1., q01=-1., q99=1., mean=0., std=1.)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".freeze.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe relative artifact path")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Artifact escapes release")
    return path


def install(src, dst, entry, *, hardlink):
    require(not src.is_symlink() and src.is_file(), f"Expected immutable regular payload: {src}")
    require(src.stat().st_size == entry["bytes"], f"Source payload size changed: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        require(not dst.is_symlink(), f"Unexpected destination symlink: {dst}")
        if hardlink:
            require(os.path.samefile(src, dst), f"Frozen payload no longer references original inode: {dst}")
        else:
            require(sha(dst) == entry["sha256"], f"Conflicting copied metadata: {dst}")
        return
    if hardlink:
        os.link(src, dst)
    else:
        require(sha(src) == entry["sha256"], f"Metadata checksum changed: {src}")
        tmp = dst.with_name(dst.name + ".freeze.tmp")
        shutil.copyfile(src, tmp)
        require(sha(tmp) == entry["sha256"], f"Copied metadata checksum mismatch: {dst}")
        tmp.replace(dst)


def freeze(snapshot_path, output):
    snapshot = read(snapshot_path)
    source = Path(snapshot["root"]).absolute()
    output = Path(output).absolute()
    require(not output.resolve().is_relative_to(source.resolve()) and
            not source.resolve().is_relative_to(output.resolve()), "Independent frozen output is required")
    identity = dict(schema=SCHEMA, snapshot_sha256=sha(snapshot_path), source=str(source), output=str(output))
    owner = output / "FREEZE_OWNER.json"
    if owner.exists():
        require(read(owner) == identity, "Frozen output belongs to another snapshot")
    else:
        require(not output.exists() or not any(output.iterdir()), "Refusing nonempty unowned output")
        atomic(owner, identity)
    complete = output / "freeze_complete.json"
    if complete.exists():
        require(read(complete)["identity"] == identity, "Frozen completion identity differs")
        verify_frozen(output)
        return read(complete)
    shutil.copyfile(snapshot_path, output / "snapshot.json")
    inventory, sources = [], {}
    for name, selection in snapshot["sources"].items():
        rows = selection["buckets_snapshot"]
        require(len(rows) == selection["buckets"] and len({r["bucket"] for r in rows}) == len(rows),
                f"Ambiguous snapshot population: {name}")
        summary = dict(buckets=len(rows), episodes=0, frames=0, nominal_hours=0., linked_bytes=0,
                       copied_bytes=0, profile_statuses={})
        sources[name] = summary
        for row in rows:
            original, target = inside(source, name + "/" + row["bucket"]), inside(output, name + "/" + row["bucket"])
            require(sha(original / "meta/info.json") == row["info_sha256"], "Snapshot info changed")
            publication_path = original / "_PFS_PUBLICATION.json"
            publication_hash = sha(publication_path)
            publication = read(publication_path)
            require(publication.get("complete") is True and publication.get("schema") == "touchwam.ftp1.pfs_publication.v1",
                    f"Incomplete or unsupported payload publication: {original}")
            require(publication.get("identity", {}).get("source") == name and
                    publication["identity"].get("key") == row["bucket"], "Payload publication identity differs")
            entries = publication["files"]
            require(len(entries) == publication["file_count"] and len({e["path"] for e in entries}) == len(entries),
                    "Ambiguous payload inventory")
            inventory_bytes = (json.dumps(entries, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
            require(hashlib.sha256(inventory_bytes).hexdigest() == publication.get("publication_id"),
                    "Payload inventory digest differs")
            require(sum(e["bytes"] for e in entries) == publication["payload_bytes"], "Payload byte inventory differs")
            for entry in entries:
                relative = entry["path"]
                linked = relative.startswith(("data/", "videos/"))
                # Metadata parquet is private; only immutable training payload is linked.
                install(inside(original, relative), inside(target, relative), entry, hardlink=linked)
                summary["linked_bytes" if linked else "copied_bytes"] += entry["bytes"]
            require(sha(publication_path) == publication_hash, "Publication changed while freezing")
            provenance = output / "provenance" / name / row["bucket"]
            provenance.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(publication_path, provenance / "_PFS_PUBLICATION.json")
            info = read(target / "meta/info.json")
            require(info["total_episodes"] == row["episodes"] and info["total_frames"] == row["frames"],
                    "Snapshot frame/episode counts differ")
            status = read(target / "meta/openwam_adapter_status.json")["action_status"]
            summary["profile_statuses"][status] = summary["profile_statuses"].get(status, 0) + 1
            summary["episodes"] += row["episodes"]
            summary["frames"] += row["frames"]
            summary["nominal_hours"] += row["frames"] / float(info["fps"]) / 3600
            inventory.append(dict(source=name, bucket=row["bucket"], publication_sha256=publication_hash,
                                  info_sha256=row["info_sha256"], payload_bytes=publication["payload_bytes"]))
        print(json.dumps(dict(event="source_frozen", source=name, **summary)), flush=True)
    atomic(output / "publication_inventory.json", inventory)
    record = dict(schema=SCHEMA, identity=identity, complete=True, sources=sources,
                  inventory_sha256=sha(output / "publication_inventory.json"),
                  integrity_scope="original completed SHA256 publication; fresh metadata hashes, payload sizes and same-inode hardlinks; no full media rehash",
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic(complete, record)
    verify_frozen(output)
    return record


def verify_frozen(output):
    """Recheck the frozen inventory and links, without rereading large media."""
    output = Path(output).absolute()
    complete = read(output / "freeze_complete.json")
    require(sha(output / "snapshot.json") == complete["identity"]["snapshot_sha256"], "Frozen snapshot changed")
    require(sha(output / "publication_inventory.json") == complete["inventory_sha256"], "Frozen inventory changed")
    snapshot = read(output / "snapshot.json")
    linked_files, linked_bytes = 0, 0
    for row in read(output / "publication_inventory.json"):
        source, name = row["source"], row["bucket"]
        original = Path(snapshot["root"]) / source / name
        target = output / source / name
        receipt = output / "provenance" / source / name / "_PFS_PUBLICATION.json"
        require(sha(receipt) == row["publication_sha256"] == sha(original / "_PFS_PUBLICATION.json"),
                "Original publication changed since freeze")
        publication = read(receipt)
        entries = publication["files"]
        encoded_inventory = (json.dumps(entries, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
        require(hashlib.sha256(encoded_inventory).hexdigest() == publication["publication_id"], "Payload inventory digest differs")
        require(sha(target / "meta/info.json") == row["info_sha256"] == sha(original / "meta/info.json"),
                "Frozen dataset info changed")
        for entry in entries:
            relative = entry["path"]
            if relative.startswith(("data/", "videos/")):
                path = inside(target, relative)
                require(not path.is_symlink() and path.is_file() and path.stat().st_size == entry["bytes"]
                        and os.path.samefile(path, inside(original, relative)), "Frozen media/parquet link changed")
                linked_files += 1
                linked_bytes += entry["bytes"]
    result = dict(schema=SCHEMA, passed=True, snapshot_sha256=sha(output / "snapshot.json"),
                  linked_files=linked_files, linked_bytes=linked_bytes,
                  scope="completed immutable SHA256 receipts, fresh receipt/info hashes, payload sizes and original inode links; media bytes not rehashed")
    atomic(output / "freeze_verification.json", result)
    return result


def reader_config(root, tactile):
    return dict(type="lerobot_robot_intensity", dataset_dir=str(root), source_kind="robot", source_family="ftp1",
                enabled=True, expected_modalities=["video_action_tactile" if tactile else "video_action"],
                require_publication=True, strict_buckets=True, action_contract="meta/action_contract.json",
                action_mode="eef", normalize_mode="min-max", unify_action=False,
                tactile_mode="required" if tactile else "off", tactile_representation="rgb",
                tactile_rgb_manifest="meta/tactile_rgb.json", tactile_rgb_map_size=128,
                tactile_observation_map_size=224, num_frames=33, video_stride=4, window_stride=1,
                sampling_fps=30, height=384, width=320)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validation_identity(bucket, tactile_spec):
    return dict(schema="touchwam_ftp_validation_identity_v1", tactile_spec_sha256=digest(tactile_spec),
                metadata_sha256={p.relative_to(bucket).as_posix(): sha(p)
                                 for p in sorted((bucket / "meta").rglob("*")) if p.is_file()},
                reader=dict(height=384, width=320, num_frames=33, video_stride=4, sampling_fps=30,
                            tactile_observation_map_size=224, tactile_rgb_map_size=128))


def legacy_cache_proof(root, repo, results, tactile_spec, commit):
    """Explicit one-time migration of a sealed acceptance with known model config.

    The caller must name its historical model commit. Ordinary resume never
    infers a missing cache identity. A durable proof permits interrupted cache
    migration while preserving the original accepted sample results verbatim.
    """
    from openwam.dataloader.publication import publication_contract
    revision = subprocess.check_output(["git", "rev-parse", commit + "^{commit}"], cwd=repo, text=True).strip()
    subprocess.run(["git", "diff", "--exit-code", revision, "--", "configs/model",
                    "configs/train_touchwam_ftp1.yaml", "configs/train_touchwam_pretrain.yaml"],
                   cwd=repo, check=True, capture_output=True)
    proof_path = root / "validation_cache_migration.json"
    if proof_path.exists():
        proof = read(proof_path)
        require(proof["model_commit"] == revision and proof["tactile_spec_sha256"] == digest(tactile_spec),
                "Legacy cache model proof changed")
    else:
        publication_contract(root)  # Full source snapshot + metadata + acceptance binding.
        publication = read(root / "publication.json")
        proof = dict(schema="touchwam_ftp_sealed_cache_migration_v1", model_commit=revision,
                     tactile_spec_sha256=digest(tactile_spec), publication_sha256=sha(root / "publication.json"),
                     metadata_sha256=publication["metadata_sha256"],
                     accepted_results_sha256={name: digest(row) for name, row in results.items()})
        atomic(proof_path, proof)
    for relative, fingerprint in proof["metadata_sha256"].items():
        if relative != "reader_validation.json":
            require(sha(inside(root, relative)) == fingerprint, "Legacy accepted metadata changed")
    for name, row in results.items():
        original = {key: value for key, value in row.items() if key not in ("cache_identity", "cache_migration_sha256")}
        require(digest(original) == proof["accepted_results_sha256"].get(name), "Legacy accepted sample result changed")
    return sha(proof_path)


def apply_finalization(root, intent):
    require(sha(root / "normalization_stats.npy") == intent["result"]["normalization_sha256"],
            "Finalization intent normalization changed")
    # Validate all files before any mutation. A restart accepts only the exact
    # pre-finalization bytes or the intended replacement, never unrelated edits.
    for relative, entry in intent["files"].items():
        path = inside(root, relative)
        after = hashlib.sha256(encoded(entry["content"])).hexdigest()
        before = entry["before_sha256"]
        require((path.exists() and sha(path) in {before, after}) or (not path.exists() and before is None),
                f"Unexpected metadata while resuming finalization: {path}")
    for relative, entry in intent["files"].items():
        atomic(inside(root, relative), entry["content"])
    atomic(root / "normalization_complete.json", intent["result"])
    return intent["result"]


def finalize(output, name):
    import numpy as np
    import pyarrow.parquet as pq
    output = Path(output).absolute()
    frozen = read(output / "freeze_complete.json")
    selection = read(output / "snapshot.json")["sources"][name]
    root = output / name
    names = [r["bucket"] for r in selection["buckets_snapshot"]]
    statuses = [read(root / n / "meta/openwam_adapter_status.json") for n in names]
    if any(s["action_status"] == "pending_verified_action_profile" for s in statuses):
        reason = "No verified native joint-to-canonical EEF kinematics/profile; optical touch does not establish action semantics"
        result = dict(schema=SCHEMA, status="archive_only", ready=False, training_ready=False,
                      buckets_expected=len(names), buckets_completed=len(names), reason=reason,
                      **{k: frozen["sources"][name][k] for k in ("episodes", "frames", "nominal_hours")})
        atomic(root / "publication.json", result)
        atomic(root / "reader_config.yaml", dict(enabled=False, dataset_dir=str(root), reason=reason))
        return result
    if (root / "normalization_complete.json").exists():
        result = read(root / "normalization_complete.json")
        require(sha(root / "normalization_stats.npy") == result["normalization_sha256"], "Frozen normalization changed")
        return result
    intent_path = root / "normalization_intent.json"
    if intent_path.exists():
        intent = read(intent_path)
        require(intent["result"]["snapshot_sha256"] == sha(output / "snapshot.json"), "Finalization snapshot changed")
        return apply_finalization(root, intent)
    contracts = [read(root / n / "meta/action_contract.pending.json") for n in names]
    first = contracts[0]
    for status, contract, bucket in zip(statuses, contracts, names):
        require(status.get("complete") is True and status["action_status"] == "awaiting_source_normalization",
                "Expected completed verified profile awaiting population statistics")
        require(all(sha(inside(root / bucket, path)) == value for path, value in status["artifact_sha256"].items()),
                "Adapter artifacts changed before finalization")
        require(contract["profile_sha256"] == first["profile_sha256"] and contract["action"] == first["action"]
                and contract["source_id"] == first["source_id"], "Source action profiles disagree")
    slots = first["action"]["slots"]
    populations = [[] for _ in slots]
    train_frames, train_episodes = 0, 0
    for bucket in names:
        info = read(root / bucket / "meta/info.json")
        split = info.get("splits", {}).get("train")
        require(isinstance(split, str) and len(split.split(":")) == 2, "Explicit contiguous train split required")
        lo, hi = map(int, split.split(":"))
        require(0 <= lo <= hi <= info["total_episodes"], "Invalid train split")
        train_episodes += hi - lo
        for path in sorted((root / bucket / "data").rglob("*.parquet")):
            table = pq.read_table(path, columns=["episode_index", "observation.state", "observation.state.valid"])
            episode = np.asarray(table["episode_index"])
            data = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            valid = np.asarray(table["observation.state.valid"].to_pylist(), dtype=bool)
            require(data.shape == valid.shape == (len(table), len(slots)), "Malformed canonical state/mask")
            train = (episode >= lo) & (episode < hi)
            train_frames += int(train.sum())
            for index in range(len(slots)):
                populations[index].append(data[train & valid[:, index], index].copy())
    stats = {key: np.full(80, value, np.float32) for key, value in IDENTITY.items()}
    counts = np.zeros(80, np.int64)
    for index, slot in enumerate(slots):
        values = np.concatenate(populations[index]).astype(np.float64)
        populations[index] = []
        require(len(values) and np.isfinite(values).all(), "No finite valid training population")
        counts[slot] = len(values)
        if slot not in (*range(3, 9), *range(37, 43)):
            measured = dict(min=values.min(), max=values.max(), mean=values.mean(), std=values.std(),
                            q01=np.quantile(values, .01), q99=np.quantile(values, .99))
            for key, value in measured.items():
                stats[key][slot] = value
    path = root / "normalization_stats.npy"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, dict(eef=stats, eef_state={k: v.copy() for k, v in stats.items()}))
    temporary.replace(path)
    fingerprint = sha(path)
    modalities = set()
    replacements = {}

    def plan(path, content):
        replacements[path.relative_to(root).as_posix()] = dict(
            before_sha256=sha(path) if path.exists() else None, content=content)
        return hashlib.sha256(encoded(content)).hexdigest()

    for bucket, status, contract in zip(names, statuses, contracts):
        target = root / bucket
        tactile = (target / "meta/tactile_rgb.json").exists()
        modalities.add("video_action_tactile" if tactile else "video_action")
        contract["verification"]["status"] = "verified"
        contract["normalization"] = dict(mode="min-max", population="train", file="../normalization_stats.npy",
                                          sha256=fingerprint, action_key="eef", proprio_key="eef_state")
        if "source_native_length" in contract["action"]["units"]:
            contract["normalization"]["native_unit_policy"] = "preserve_native_xyz_source_minmax_only"
        action_sha = plan(target / "meta/action_contract.json", contract)
        definition = read(target / "meta/source_definition.json")
        definition.update(action_enabled=True, action_status="verified_future_state_target", training_ready=True,
                          training_modalities=["video_action_tactile" if tactile else "video_action"])
        definition_sha = plan(target / "meta/source_definition.json", definition)
        config = reader_config(target, tactile)
        config.update(enabled=False, require_publication=False)
        config_sha = plan(target / "reader_config.yaml", config)
        status.update(action_status="verified_future_state_target", training_ready=False,
                      readiness_note="Frozen source normalization finalized; source publication requires actual reader acceptance")
        status["artifact_sha256"].update({"meta/action_contract.json": action_sha,
                                          "meta/source_definition.json": definition_sha,
                                          "reader_config.yaml": config_sha})
        plan(target / "meta/openwam_adapter_status.json", status)
    require(len(modalities) == 1, "Source modalities differ")
    result = dict(schema=SCHEMA, source=name, normalization_sha256=fingerprint, counts=counts.tolist(),
                  train_frames=train_frames, train_episodes=train_episodes, modality=modalities.pop(),
                  buckets=names, snapshot_sha256=sha(output / "snapshot.json"),
                  quantile_estimator="exact_native_valid_train_rows", complete=True)
    intent = dict(schema=SCHEMA, result=result, files=replacements)
    atomic(intent_path, intent)
    return apply_finalization(root, intent)


def validate(output, name, repo, legacy_model_commit=None):
    import gc
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from openwam.dataloader.robot_intensity import LeRobotRobotIntensityDataset
    from openwam.dataloader.registry import build_dataset
    from scripts.preflight_touchwam import validate_sample
    torch.set_num_threads(1)
    output, repo = Path(output).absolute(), Path(repo).absolute()
    root = output / name
    if not (root / "normalization_complete.json").exists():
        return read(root / "publication.json")
    norm = read(root / "normalization_complete.json")
    tactile = norm["modality"] == "video_action_tactile"
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        config = compose(config_name="train_touchwam_ftp1")
    tactile_spec = OmegaConf.to_container(config.model.architecture.tactile, resolve=True)
    record_path = root / "reader_validation.json"
    results = read(record_path) if record_path.exists() else {}
    proof = None
    if legacy_model_commit and any("cache_identity" not in row for row in results.values()):
        proof = legacy_cache_proof(root, repo, results, tactile_spec, legacy_model_commit)
    atomic(root / "publication.json", dict(status="validating", ready=False, training_ready=False))
    for bucket in norm["buckets"]:
        identity = validation_identity(root / bucket, tactile_spec)
        if bucket in results:
            require(results[bucket]["status"] == "passed", "Prior bucket acceptance failed")
            require(results[bucket]["action_contract_sha256"] == sha(root / bucket / "meta/action_contract.json"),
                    "Action contract changed after bucket acceptance")
            if proof and "cache_identity" not in results[bucket]:
                results[bucket].update(cache_identity=identity, cache_migration_sha256=proof)
                atomic(record_path, results)
            if results[bucket].get("cache_identity") == identity:
                continue
        dataset = LeRobotRobotIntensityDataset(root / bucket, tactile_mode="required" if tactile else "off",
            tactile_representation="rgb", tactile_observation_map_size=224, tactile_rgb_map_size=128,
            height=384, width=320)
        # Every frozen bucket is initialized and physically decoded. Boundary
        # samples additionally cover three evenly spaced buckets per source.
        indices = {len(dataset)//2}
        if bucket in {norm["buckets"][0], norm["buckets"][len(norm["buckets"])//2], norm["buckets"][-1]}:
            indices.update((0, len(dataset)-1))
        samples = []
        for index in sorted(indices):
            sample = dataset[index]
            result = validate_sample(sample, norm["modality"], tactile_spec=config.model.architecture.tactile,
                                     source_kind="robot", expected_modalities=[norm["modality"]],
                                     expected_tactile_codec="tactile_rgb_v1" if tactile else None)
            samples.append(dict(index=index, result=result))
        results[bucket] = dict(status="passed", windows=len(dataset), samples=samples,
                               action_contract_sha256=sha(root / bucket / "meta/action_contract.json"),
                               cache_identity=identity)
        atomic(record_path, results)
        del dataset, sample
        gc.collect()
        print(json.dumps(dict(event="bucket_reader_passed", source=name, bucket=bucket,
                              completed=len(results), expected=len(norm["buckets"]))), flush=True)
    frozen = read(output / "freeze_complete.json")["sources"][name]
    complete = dict(schema=SCHEMA, status="complete", ready=True, training_ready=True,
                    buckets_expected=len(norm["buckets"]), buckets_completed=len(results),
                    **{key: frozen[key] for key in ("episodes", "frames", "nominal_hours")},
                    snapshot_sha256=sha(output / "snapshot.json"), normalization_sha256=norm["normalization_sha256"],
                    reader_validation_sha256=sha(record_path), training_modality=norm["modality"],
                    train_frames=norm["train_frames"], train_episodes=norm["train_episodes"])
    complete["source"] = name
    complete["publication_inventory_sha256"] = sha(output / "publication_inventory.json")
    metadata_files = [root / "normalization_stats.npy", root / "normalization_complete.json", record_path]
    for bucket in norm["buckets"]:
        metadata_files.extend(p for p in (root / bucket / "meta").rglob("*") if p.is_file())
    complete["metadata_sha256"] = {p.relative_to(root).as_posix(): sha(p) for p in sorted(metadata_files)}
    atomic(root / "publication.json", complete)
    reader = reader_config(root, tactile)
    atomic(root / "reader_config.yaml", reader)
    try:
        dataset = build_dataset(reader, split="train")
        sample = dataset[0]
        validate_sample(sample, norm["modality"], tactile_spec=config.model.architecture.tactile,
                        source_kind="robot", expected_modalities=[norm["modality"]],
                        expected_tactile_codec="tactile_rgb_v1" if tactile else None)
        complete["registry_validation"] = dict(windows=len(dataset), publication_verified=True,
                                                modality=sample["training_modality"])
    except Exception:
        atomic(root / "publication.json", dict(status="registry_failed", ready=False, training_ready=False))
        raise
    atomic(root / "publication.json", complete)
    return complete


def catalog(output):
    output = Path(output).absolute()
    frozen = read(output / "freeze_complete.json")
    sources = {}
    for name in frozen["sources"]:
        path = output / name / "publication.json"
        sources[name] = read(path) if path.exists() else dict(status="pending", training_ready=False)
    complete = all(s.get("status") in ("complete", "archive_only") for s in sources.values())
    result = dict(schema=SCHEMA, status="complete" if complete else "pending", snapshot_sha256=sha(output / "snapshot.json"),
                  frozen_hours=sum(s["nominal_hours"] for s in frozen["sources"].values()),
                  training_hours=sum(s.get("nominal_hours", 0) for s in sources.values() if s.get("training_ready")),
                  sources=sources, training_launch_allowed=False,
                  note="Use only source reader_config files with enabled=true; model/training preflight remains separate")
    atomic(output / "training_catalog.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "verify", "finalize", "validate", "catalog", "all"), required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source")
    parser.add_argument("--legacy-model-commit", help="Explicit historical model-config commit for sealed legacy cache migration")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    if args.phase in ("freeze", "all"):
        require(args.snapshot is not None, "--snapshot required")
        result = freeze(args.snapshot, args.output)
    if args.phase == "verify":
        result = verify_frozen(args.output)
    if args.phase in ("finalize", "validate"):
        require(args.source is not None, "--source required")
        result = finalize(args.output, args.source) if args.phase == "finalize" else validate(
            args.output, args.source, args.repo, args.legacy_model_commit)
    if args.phase == "all":
        # A fresh process per source releases Arrow/PyTorch caches on small CPU hosts.
        for name in read(args.output / "snapshot.json")["sources"]:
            for phase in ("finalize", "validate"):
                command = [sys.executable, __file__, "--phase", phase, "--output", str(args.output),
                           "--source", name, "--repo", str(args.repo)]
                if args.legacy_model_commit:
                    command += ["--legacy-model-commit", args.legacy_model_commit]
                subprocess.run(command, check=True)
    if args.phase in ("all", "catalog"):
        result = catalog(args.output)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
