#!/usr/bin/env python3
"""Publish a small immutable directory through a POSIX BOS mount.

Example (one writer per destination; stop source writers before publishing)::

    python scripts/data/ftp1/publish_bos.py --source /pfs/pilot \
        --destination /dataset/TouchWAM/ftp1_lerobot_v3_pilot_20261028 \
        --max-bytes 10737418240

Only _BOS_PUBLICATION.json commits a publication. Consumers must validate that
manifest and its file hashes, not infer completion from payload file presence.
No flock, rename, pickle, credentials, or source deletion is used. BOS mounts
must provide read-after-close visibility; verification reopens each destination
object but cannot bypass a mount's own cache. Concurrent destination writers are
unsupported, including external edits while a publication is incomplete.

An intent precedes each upload; a receipt follows a fresh destination SHA256.
On interruption only an object with this plan's complete intent and WITHOUT a
complete receipt may be rewritten. Published/verified conflicts always fail.
Private state records survive for auditing/resume; their bytes count against the
budget. Conversion/preview metadata is copied verbatim: publication certifies
transport integrity only, never calibration, scientific validity or trainability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

SCHEMA = "touchwam.ftp1.bos_publication.v1"
MANIFEST = "_BOS_PUBLICATION.json"
STATE = "_BOS_PUBLISH_STATE"
CHUNK_BYTES = 8 * 1024**2
TEMPORARY = re.compile(r"(^|[._-])(partial|tmp|temp)([._-]|$)", re.IGNORECASE)
VERIFICATION = {
    "method": "posix_mount_reopen_sha256",
    "independent_bos_api_verified": False,
    "mount_cache_bypassed": False,
}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def safe_relative(value):
    """Reject traversal and ambiguous Windows/POSIX spellings before joining."""
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in (".", "..") for part in path.parts)
        or "\\" in value
        or ":" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def regular_path(path):
    """Do not follow file links or accept devices/FIFOs."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected regular file, refusing symlink/special file: {path}")


def checked_path(root, relative):
    path = root
    for component in safe_relative(relative).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in path: {path}")
    return path


def sha256_file(path):
    regular_path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def snapshot(path):
    regular_path(path)
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def raise_walk_error(error):
    # os.walk otherwise silently omits unreadable directories.
    raise error


def walk_source(root):
    paths, excluded = [], []
    for folder, dirs, files in os.walk(root, followlinks=False, onerror=raise_walk_error):
        for name in sorted(dirs + files):
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            safe_relative(relative)
            if TEMPORARY.search(name) or name.endswith(("~", ".swp", ".swo")):
                excluded.append(relative)
                if name in dirs:
                    dirs.remove(name)
                continue
            if relative in (MANIFEST, STATE):
                raise ValueError(f"Source uses reserved publication path: {relative}")
            if path.is_symlink():
                raise ValueError(f"Refusing source symlink: {path}")
            if name in files:
                regular_path(path)
                paths.append(relative)
    return sorted(paths), sorted(excluded)


def normalized_roots(source, destination):
    # Resolve mount aliases to detect overlap. Descendant links are rejected.
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Source must be a directory")
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Source and destination must not overlap or nest")
    if destination.exists() and not destination.is_dir():
        raise ValueError("Destination must be a directory")
    return source, destination


def record_state(path, expected):
    """Deterministic small records can recover a prefix left by interruption."""
    if not path.exists():
        return "missing"
    regular_path(path)
    with path.open("rb") as handle:
        actual = handle.read(len(expected) + 1)
    if actual == expected:
        return "complete"
    if expected.startswith(actual):
        return "partial"
    raise ValueError(f"Conflicting publication record: {path}")


def write_record(path, expected):
    state = record_state(path, expected)
    if state != "complete":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb" if state == "missing" else "wb") as handle:
            handle.write(expected)
        # A successful close/size alone is insufficient, even for records.
        if sha256_file(path) != (len(expected), hashlib.sha256(expected).hexdigest()):
            raise ValueError(f"Publication record readback checksum mismatch: {path}")


def copy_stream(source, destination, expected, *, overwrite):
    """This function is deliberately separate for interrupted-write tests."""
    digest = hashlib.sha256()
    count = 0
    regular_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb" if overwrite else "xb") as writer:
        while block := reader.read(CHUNK_BYTES):
            count += len(block)
            if count > expected["bytes"]:
                raise ValueError(f"Source grew during copy: {source}")
            writer.write(block)
            digest.update(block)
    if (count, digest.hexdigest()) != (expected["bytes"], expected["sha256"]):
        raise ValueError(f"Source checksum changed during copy: {source}")


def matches(path, entry):
    # Hash even equal-size objects. Size is never an integrity check.
    return sha256_file(path) == (entry["bytes"], entry["sha256"])


def publication_plan(source, *, max_bytes):
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    paths, excluded = walk_source(source)
    if not paths:
        raise ValueError("No publishable regular files in source")
    snapshots = {p: snapshot(checked_path(source, p)) for p in paths}
    total_bytes = sum(value[2] for value in snapshots.values())
    if total_bytes > max_bytes:
        raise ValueError(f"Directory exceeds byte budget: payload {total_bytes} > {max_bytes}")
    entries = []
    for relative in paths:
        path = checked_path(source, relative)
        size, digest = sha256_file(path)
        if snapshot(path) != snapshots[relative] or size != snapshots[relative][2]:
            raise ValueError(f"Source changed during inventory: {path}")
        entries.append({"path": relative, "bytes": size, "sha256": digest})
    publication_id = hashlib.sha256(json_bytes(entries)).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "publication_id": publication_id,
        "complete": True,
        "files": entries,
        "payload_bytes": total_bytes,
        "file_count": len(entries),
        "integrity_scope": "byte_exact_transport_only",
        "scientific_metadata": "preserved_verbatim_without_reclassification",
        "source_retention": "unchanged_no_automatic_deletion",
        "verification": VERIFICATION,
    }
    manifest_bytes = json_bytes(manifest)
    plan_bytes = json_bytes({
        "schema": SCHEMA,
        "publication_id": publication_id,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "complete": False,
    })
    records = {}
    for entry in entries:
        key = hashlib.sha256(entry["path"].encode("utf-8")).hexdigest()
        records[entry["path"]] = (
            f"{STATE}/{key}.intent.json",
            f"{STATE}/{key}.verified.json",
            json_bytes({"schema": SCHEMA, "publication_id": publication_id, "file": entry}),
        )
    required_bytes = total_bytes + len(manifest_bytes) + len(plan_bytes)
    required_bytes += sum(2 * len(record[2]) for record in records.values())
    if required_bytes > max_bytes:
        raise ValueError(f"Directory exceeds byte budget including publication records: {required_bytes} > {max_bytes}")
    return manifest, manifest_bytes, plan_bytes, records, snapshots, excluded, required_bytes


def check_destination_layout(destination, allowed):
    if not destination.exists():
        return
    for folder, dirs, files in os.walk(destination, followlinks=False, onerror=raise_walk_error):
        for name in dirs + files:
            path = Path(folder) / name
            relative = path.relative_to(destination).as_posix()
            safe_relative(relative)
            if path.is_symlink():
                raise ValueError(f"Refusing destination symlink: {path}")
            if name in files:
                regular_path(path)
                if relative not in allowed:
                    raise ValueError(f"Unexpected destination object: {relative}")


def publish(source, destination, *, max_bytes, dry_run=False):
    source, destination = normalized_roots(source, destination)
    manifest, manifest_bytes, plan_bytes, records, snapshots, excluded, required = publication_plan(
        source, max_bytes=max_bytes
    )
    entries = manifest["files"]
    plan_path = checked_path(destination, f"{STATE}/plan.json")
    manifest_path = checked_path(destination, MANIFEST)
    allowed = {entry["path"] for entry in entries} | {MANIFEST, f"{STATE}/plan.json"}
    allowed.update(path for record in records.values() for path in record[:2])
    check_destination_layout(destination, allowed)
    plan_state = record_state(plan_path, plan_bytes)
    manifest_state = record_state(manifest_path, manifest_bytes)
    if manifest_state != "missing" and plan_state != "complete":
        raise ValueError("Publication manifest without a complete matching plan")

    # Preflight every conflict before any mutation, so a mismatched existing
    # object can never acquire an intent that would authorize overwriting it.
    actions = []
    for entry in entries:
        relative = entry["path"]
        target = checked_path(destination, relative)
        intent_name, receipt_name, record_bytes = records[relative]
        intent_state = record_state(checked_path(destination, intent_name), record_bytes)
        receipt_state = record_state(checked_path(destination, receipt_name), record_bytes)
        if (intent_state != "missing" or receipt_state != "missing") and plan_state != "complete":
            raise ValueError("File records without a complete matching plan")
        present = target.exists()
        correct = present and matches(target, entry)
        if not correct:
            committed = manifest_state == "complete" or receipt_state == "complete"
            owned_incomplete = intent_state == "complete" and receipt_state != "complete"
            if committed or (present and not owned_incomplete):
                raise ValueError(f"Destination checksum conflict or missing committed object: {relative}")
        actions.append((entry, correct, present))

    summary = {
        "event": "bos_publication_planned" if dry_run else "bos_publication_complete",
        "source": str(source),
        "destination": str(destination),
        "publication_id": manifest["publication_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "payload_bytes": manifest["payload_bytes"],
        "publication_bytes": required,
        "budget_bytes": max_bytes,
        "files": len(entries),
        "copied_files": sum(not correct for _, correct, _ in actions),
        "skipped_files": sum(correct for _, correct, _ in actions),
        "recovered_files": sum(present and not correct for _, correct, present in actions),
        "excluded_temporary_paths": excluded,
        "scientific_metadata": manifest["scientific_metadata"],
        "verification": VERIFICATION,
        "complete": not dry_run,
    }
    if dry_run:
        return summary

    write_record(plan_path, plan_bytes)
    for entry, correct, present in actions:
        relative = entry["path"]
        target = checked_path(destination, relative)
        intent_name, receipt_name, record_bytes = records[relative]
        if not correct:
            source_path = checked_path(source, relative)
            if snapshot(source_path) != snapshots[relative]:
                raise ValueError(f"Source changed before copy: {relative}")
            write_record(checked_path(destination, intent_name), record_bytes)
            copy_stream(source_path, target, entry, overwrite=present)
            if not matches(checked_path(destination, relative), entry):
                raise ValueError(f"Destination readback checksum mismatch: {relative}")
        write_record(checked_path(destination, receipt_name), record_bytes)

    # Recheck all objects at the commit barrier, including objects skipped on
    # resume, and reject source changes/additions before exposing completion.
    current_paths, _ = walk_source(source)
    if current_paths != sorted(snapshots):
        raise ValueError("Source file inventory changed during publication")
    for entry in entries:
        relative = entry["path"]
        if snapshot(checked_path(source, relative)) != snapshots[relative]:
            raise ValueError(f"Source changed during publication: {relative}")
        if not matches(checked_path(destination, relative), entry):
            raise ValueError(f"Destination checksum conflict at commit: {relative}")
    write_record(manifest_path, manifest_bytes)
    return summary


def verify_published(destination, *, expected_manifest_sha256, max_bytes):
    """Reverify a committed copy using a trusted receipt, without local source."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    destination = Path(destination).expanduser().resolve(strict=True)
    path = checked_path(destination, MANIFEST)
    regular_path(path)
    if path.stat().st_size > min(max_bytes, 64 * 1024**2):
        raise ValueError("Publication manifest exceeds verification byte budget")
    size, digest = sha256_file(path)
    if digest != expected_manifest_sha256:
        raise ValueError("Publication manifest checksum differs from trusted receipt")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("Not a completed BOS publication")
    entries = manifest["files"]
    if hashlib.sha256(json_bytes(entries)).hexdigest() != manifest["publication_id"]:
        raise ValueError("Publication file identity mismatch")
    paths = [entry["path"] for entry in entries]
    if paths != sorted(set(paths)) or manifest["file_count"] != len(paths):
        raise ValueError("Publication file list is not unique and sorted")
    if sum(entry["bytes"] for entry in entries) != manifest["payload_bytes"]:
        raise ValueError("Publication payload byte count mismatch")
    plan_bytes = json_bytes({"schema": SCHEMA, "publication_id": manifest["publication_id"],
                             "manifest_sha256": digest, "complete": False})
    plan_path = checked_path(destination, f"{STATE}/plan.json")
    if record_state(plan_path, plan_bytes) != "complete":
        raise ValueError("Missing or incomplete publication plan")
    allowed = {MANIFEST, f"{STATE}/plan.json"}
    for entry in entries:
        relative = entry["path"]
        if safe_relative(relative).parts[0] in (STATE, MANIFEST):
            raise ValueError("Payload uses a reserved publication path")
        target = checked_path(destination, relative)
        if not matches(target, entry):
            raise ValueError(f"Destination checksum conflict: {relative}")
        key = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        receipt_name = f"{STATE}/{key}.verified.json"
        receipt_bytes = json_bytes({"schema": SCHEMA, "publication_id": manifest["publication_id"], "file": entry})
        if record_state(checked_path(destination, receipt_name), receipt_bytes) != "complete":
            raise ValueError(f"Missing or incomplete verified receipt: {relative}")
        allowed.update((relative, receipt_name, f"{STATE}/{key}.intent.json"))
    check_destination_layout(destination, allowed)
    total = sum(checked_path(destination, p).stat().st_size for p in allowed if checked_path(destination, p).exists())
    if total > max_bytes:
        raise ValueError(f"Published directory exceeds byte budget: {total} > {max_bytes}")
    return {"event": "bos_publication_verified", "complete": True,
            "destination": str(destination), "manifest": str(path), "manifest_sha256": digest,
            "publication_id": manifest["publication_id"], "files": len(entries),
            "payload_bytes": manifest["payload_bytes"], "publication_bytes": total,
            "verification": VERIFICATION}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--max-bytes", required=True, type=int, help="Hard limit including payload and private records")
    parser.add_argument("--dry-run", action="store_true", help="Hash/preflight and report without writing anything")
    args = parser.parse_args(argv)
    try:
        result = publish(args.source, args.destination, max_bytes=args.max_bytes, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        print(json.dumps({"event": "bos_publication_failed", "complete": False, "error": str(error)}), flush=True)
        return 1
    except KeyboardInterrupt:
        print(json.dumps({"event": "bos_publication_interrupted", "complete": False}), flush=True)
        return 130
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
