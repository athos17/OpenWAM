#!/usr/bin/env python3
"""Stage one contiguous FTP-1 task byte range to PFS for a CPU cluster worker.

Source archives remain untouched. Each <=1 GiB segment has a SHA256 receipt;
partial copies resume at a completed segment boundary. The final manifest binds
archive file identities, exact original offsets, the census and the range hash.
Staged segments are retained after successful conversion; no automatic deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from convert import Parts, atomic_json, log

SCHEMA = "touchwam.ftp1.task_range.v1"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def archive_identity(files):
    rows = [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files]
    return rows, hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def stage_group(source_root, staging_root, source, group, segment_bytes=1024**3):
    if segment_bytes <= 0 or segment_bytes % 512:
        raise ValueError("Segment size must be a positive tar-block multiple")
    files = sorted(p for p in (Path(source_root) / source).iterdir() if ".tar" in p.name)
    identities, fingerprint = archive_identity(files)
    begin, end = group["offset"], group["end_offset"]
    if begin < 0 or begin % 512 or end <= begin or end > sum(row[1] for row in identities):
        raise ValueError("Invalid census archive range")
    name = hashlib.sha256(group["group"].encode()).hexdigest()[:20]
    folder = Path(staging_root) / source / name
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path, verify=True)
        if manifest["source_fingerprint"] != fingerprint or manifest["group"] != group:
            raise ValueError("Staged source/census identity changed")
        return manifest_path
    parts = []
    aggregate = hashlib.sha256()
    with Parts(files) as stream:
        offset = begin
        index = 0
        while offset < end:
            count = min(segment_bytes, end - offset)
            filename = f"segment-{index:05d}.bin"
            path = folder / filename
            receipt_path = path.with_suffix(".json")
            expected = {"source_fingerprint": fingerprint, "source_offset": offset, "bytes": count}
            if path.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if any(receipt.get(k) != v for k, v in expected.items()) or path.stat().st_size != count:
                    raise ValueError("Incompatible staged segment receipt")
                segment_hash = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024**2), b""):
                        segment_hash.update(block)
                        aggregate.update(block)
                if segment_hash.hexdigest() != receipt["sha256"]:
                    raise ValueError("Staged segment checksum mismatch")
            else:
                temporary = path.with_suffix(".partial")
                stream.seek(offset)
                remaining = count
                segment_hash = hashlib.sha256()
                with temporary.open("wb") as handle:
                    while remaining:
                        block = stream.read(min(8 * 1024**2, remaining))
                        if not block:
                            raise EOFError("Original archive ended before census range")
                        handle.write(block)
                        segment_hash.update(block)
                        aggregate.update(block)
                        remaining -= len(block)
                temporary.replace(path)
                receipt = dict(expected, sha256=segment_hash.hexdigest())
                atomic_json(receipt_path, receipt)
            parts.append({"file": filename, "bytes": count, "sha256": receipt["sha256"]})
            offset += count
            index += 1
            log(
                "staging_segment_complete",
                source=source,
                group=group["group"],
                copied_bytes=offset - begin,
                total_bytes=end - begin,
            )
    if archive_identity(files) != (identities, fingerprint):
        raise ValueError("Source archive changed during staging")
    manifest = {
        "schema": SCHEMA,
        "source": source,
        "source_files": identities,
        "source_fingerprint": fingerprint,
        "source_root": str(source_root),
        "source_offset": begin,
        "source_end_offset": end,
        "range_bytes": end - begin,
        "range_sha256": aggregate.hexdigest(),
        "group": group,
        "segments": parts,
        "retention": "retain_until_explicit_cleanup",
        "complete": True,
    }
    atomic_json(manifest_path, manifest)
    return manifest_path


def load_manifest(path, verify=True):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or value.get("complete") is not True:
        raise ValueError("Incomplete task-range manifest")
    base, end = value["source_offset"], value["source_end_offset"]
    if (
        base != value["group"]["offset"]
        or end != value["group"]["end_offset"]
        or end - base != value["range_bytes"]
        or base % 512
    ):
        raise ValueError("Task range/census offset mismatch")
    fingerprint = hashlib.sha256(json.dumps(value["source_files"]).encode()).hexdigest()
    if fingerprint != value["source_fingerprint"]:
        raise ValueError("Archive identity fingerprint mismatch")
    count = 0
    aggregate = hashlib.sha256()
    for row in value["segments"]:
        segment = (path.parent / row["file"]).resolve(strict=True)
        if not segment.is_relative_to(path.parent) or segment.stat().st_size != row["bytes"]:
            raise ValueError("Staged segment path/size mismatch")
        if verify:
            h = hashlib.sha256()
            with segment.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024**2), b""):
                    h.update(block)
                    aggregate.update(block)
            if h.hexdigest() != row["sha256"]:
                raise ValueError("Staged segment checksum mismatch")
        count += row["bytes"]
    if count != value["range_bytes"] or (verify and aggregate.hexdigest() != value["range_sha256"]):
        raise ValueError("Staged archive range checksum/size mismatch")
    return value


class RangeArchive(io.RawIOBase):
    """Expose staged bytes at their original absolute tar offsets.

    TarInfo.offset and offset_data remain identical to the original split tar.
    Thus existing compact indices, video receipts and conversion signatures are
    compatible with direct-source conversion. Prefix bytes are never fabricated.
    """

    def __init__(self, manifest_path, manifest=None):
        path = Path(manifest_path)
        self.meta = manifest or load_manifest(path, verify=True)
        self.base = self.meta["source_offset"]
        self.end = self.meta["source_end_offset"]
        self.raw = Parts([path.parent / row["file"] for row in self.meta["segments"]])
        self.pos = self.base

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.end + offset
        if not self.base <= pos <= self.end:
            raise ValueError("Read outside staged task byte range")
        self.pos = pos
        self.raw.seek(pos - self.base)
        return pos

    def read(self, size=-1):
        size = self.end - self.pos if size < 0 else min(size, self.end - self.pos)
        value = self.raw.read(size)
        self.pos += len(value)
        return value

    def close(self):
        self.raw.close()
        super().close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--staging-root", required=True)
    p.add_argument("--census", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--segment-mib", type=int, default=1024)
    a = p.parse_args()
    matches = [
        (s["dataset"], g)
        for s in json.loads(Path(a.census).read_text())
        if s["dataset"] != "sharpa"
        for g in s["groups"]
        if g["group"] == a.group
    ]
    if len(matches) != 1:
        raise ValueError("Group must uniquely match the non-Sharpa release census")
    source, group = matches[0]
    result = stage_group(a.input, a.staging_root, source, group, a.segment_mib * 1024**2)
    print(json.dumps({"event": "staging_complete", "manifest": str(result)}), flush=True)


if __name__ == "__main__":
    main()
