#!/usr/bin/env python3
"""Stage one contiguous FTP-1 task byte range to PFS for a CPU cluster worker.

Source archives remain untouched. Each <=1 GiB segment has a SHA256 receipt;
partial copies resume at a completed segment boundary. The final manifest binds
archive file identities, exact original offsets, the census and the range hash.
Staged segments are retained after successful conversion; no automatic deletion.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import io
import json
from pathlib import Path

from convert import Parts, atomic_json, index_group, log, resolve_selection

SCHEMA = "touchwam.ftp1.task_range.v1"
SPARSE_SCHEMA = "touchwam.ftp1.episode_chunks.v1"


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
    if value.get("schema") == SPARSE_SCHEMA:
        return load_sparse_manifest(path, value, verify)
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


def selected_index(index, group):
    original, ranges, _ = resolve_selection(group)
    if "source_census" not in group:
        raise ValueError("Sparse staging requires an explicit pilot episode census")
    result = {"archive_fingerprint": index["archive_fingerprint"], "arrays": {}}
    for key, spec in index["arrays"].items():
        if not key.startswith("data/"):
            continue
        config = spec["config"]
        if config["shape"][0] != original["frames"] or config["chunks"][1:] != config["shape"][1:]:
            raise ValueError("Sparse staging requires regular time-only source chunks")
        size = config["chunks"][0]
        chunk_ids = {
            ".".join([str(i)] + ["0"] * (len(config["shape"]) - 1))
            for start, stop in ranges
            for i in range(start // size, (stop - 1) // size + 1)
        }
        result["arrays"][key] = {
            "config": copy.deepcopy(config),
            "chunks": {k: v for k, v in spec["chunks"].items() if k in chunk_ids},
        }
    return result


def stage_pilot_group(
    source_root, staging_root, source, group, segment_bytes=256 * 1024**2, index_path=None, max_bytes=None
):
    """Pack only present Zarr chunks intersecting the selected complete episodes."""
    if segment_bytes <= 0:
        raise ValueError("segment_bytes must be positive")
    files = sorted(p for p in (Path(source_root) / source).iterdir() if ".tar" in p.name)
    identities, fingerprint = archive_identity(files)
    original, _, _ = resolve_selection(group)
    selected_hash = hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()
    folder = Path(staging_root) / source / selected_hash[:20]
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        value = load_manifest(manifest_path, verify=True)
        if value["source_fingerprint"] != fingerprint or value["group"] != group:
            raise ValueError("Staged pilot source/census identity changed")
        if max_bytes is not None and value["packed_bytes"] > max_bytes:
            raise ValueError("Existing pilot exceeds byte budget")
        return manifest_path
    index = index_group(files, original, Path(index_path) if index_path else folder / "source_index.json")
    sparse = selected_index(index, group)
    payloads, total = [], 0
    for offset, count in sorted({tuple(x) for a in sparse["arrays"].values() for x in a["chunks"].values()}):
        payloads.append({"source_offset": offset, "bytes": count, "packed_offset": total})
        total += count
    if max_bytes is not None and total > max_bytes:
        raise ValueError(f"Selected chunks require {total} bytes, budget {max_bytes}; no payload was staged")
    if not payloads:
        raise ValueError("Selected episodes contain no present source chunks")
    packed_ends = [r["packed_offset"] + r["bytes"] for r in payloads]
    parts, aggregate = [], hashlib.sha256()
    with Parts(files) as original_stream:
        for i, start in enumerate(range(0, total, segment_bytes)):
            count = min(segment_bytes, total - start)
            path = folder / f"segment-{i:05d}.bin"
            receipt_path = path.with_suffix(".json")
            expected = {
                "source_fingerprint": fingerprint,
                "selection_sha256": selected_hash,
                "packed_offset": start,
                "bytes": count,
            }
            if path.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if any(receipt.get(k) != v for k, v in expected.items()) or path.stat().st_size != count:
                    raise ValueError("Incompatible sparse segment receipt")
                if sha256_file(path) != receipt["sha256"]:
                    raise ValueError("Sparse segment checksum mismatch")
            else:
                tmp = path.with_suffix(".partial")
                digest, pos = hashlib.sha256(), start
                with tmp.open("wb") as handle:
                    while pos < start + count:
                        record = payloads[bisect.bisect_right(packed_ends, pos)]
                        relative = pos - record["packed_offset"]
                        n = min(8 * 1024**2, record["bytes"] - relative, start + count - pos)
                        original_stream.seek(record["source_offset"] + relative)
                        block = original_stream.read(n)
                        if len(block) != n:
                            raise EOFError("Source archive ended within selected Zarr payload")
                        handle.write(block)
                        digest.update(block)
                        pos += n
                tmp.replace(path)
                receipt = dict(expected, sha256=digest.hexdigest())
                atomic_json(receipt_path, receipt)
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b""):
                    aggregate.update(block)
            parts.append({"file": path.name, "bytes": count, "sha256": receipt["sha256"]})
            log(
                "pilot_staging_segment_complete",
                source=source,
                group=group["group"],
                copied_bytes=start + count,
                total_bytes=total,
            )
    if archive_identity(files) != (identities, fingerprint):
        raise ValueError("Source archive changed during pilot staging")
    atomic_json(
        manifest_path,
        {
            "schema": SPARSE_SCHEMA,
            "complete": True,
            "source": source,
            "source_files": identities,
            "source_root": str(source_root),
            "source_fingerprint": fingerprint,
            "group": group,
            "selection_sha256": selected_hash,
            "archive_index": sparse,
            "payloads": payloads,
            "packed_bytes": total,
            "packed_sha256": aggregate.hexdigest(),
            "segments": parts,
            "retention": "retain_until_verified_bos_publication",
        },
    )
    return manifest_path


def load_sparse_manifest(path, value, verify=True):
    if value.get("complete") is not True:
        raise ValueError("Incomplete sparse manifest")
    original, _, _ = resolve_selection(value["group"])
    fingerprint = hashlib.sha256(json.dumps(value["source_files"]).encode()).hexdigest()
    if fingerprint != value["source_fingerprint"] or fingerprint != value["archive_index"]["archive_fingerprint"]:
        raise ValueError("Sparse source fingerprint mismatch")
    selection_hash = hashlib.sha256(json.dumps(value["group"], sort_keys=True).encode()).hexdigest()
    if selection_hash != value["selection_sha256"]:
        raise ValueError("Sparse selection checksum mismatch")
    expected_payloads = sorted(
        {tuple(x) for a in value["archive_index"]["arrays"].values() for x in a["chunks"].values()}
    )
    if expected_payloads != [(x["source_offset"], x["bytes"]) for x in value["payloads"]]:
        raise ValueError("Sparse index/payload mismatch")
    expected_index = selected_index(value["archive_index"], value["group"])
    if expected_index != value["archive_index"]:
        raise ValueError("Sparse index includes unselected chunks")
    count, prior_end = 0, original["offset"]
    for row in value["payloads"]:
        if (
            row["packed_offset"] != count
            or row["bytes"] <= 0
            or row["source_offset"] < prior_end
            or row["source_offset"] + row["bytes"] > original["end_offset"]
        ):
            raise ValueError("Sparse payload offsets invalid")
        count += row["bytes"]
        prior_end = row["source_offset"] + row["bytes"]
    if count != value["packed_bytes"]:
        raise ValueError("Sparse payload total mismatch")
    count, digest = 0, hashlib.sha256()
    for row in value["segments"]:
        segment = (path.parent / row["file"]).resolve(strict=True)
        if not segment.is_relative_to(path.parent) or segment.stat().st_size != row["bytes"]:
            raise ValueError("Sparse segment path/size mismatch")
        if verify:
            local = hashlib.sha256()
            with segment.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024**2), b""):
                    local.update(block)
                    digest.update(block)
            if local.hexdigest() != row["sha256"]:
                raise ValueError("Sparse segment checksum mismatch")
        count += row["bytes"]
    if count != value["packed_bytes"] or (verify and digest.hexdigest() != value["packed_sha256"]):
        raise ValueError("Sparse packed checksum/size mismatch")
    return value


class SparseArchive(io.RawIOBase):
    """Address packed Zarr payloads by original archive offsets; reject gaps."""

    def __init__(self, path, manifest=None):
        path = Path(path)
        self.meta = manifest or load_manifest(path)
        self.payloads = self.meta["payloads"]
        self.offsets = [r["source_offset"] for r in self.payloads]
        self.raw = Parts([path.parent / r["file"] for r in self.meta["segments"]])
        self.pos = self.offsets[0]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1):
            raise ValueError("Sparse archive supports absolute/current seek only")
        self.pos = offset if whence == 0 else self.pos + offset
        return self.pos

    def read(self, size=-1):
        i = bisect.bisect_right(self.offsets, self.pos) - 1
        if i < 0:
            raise ValueError("Read outside staged sparse payloads")
        row = self.payloads[i]
        relative = self.pos - row["source_offset"]
        if relative < 0 or relative >= row["bytes"]:
            raise ValueError("Read outside staged sparse payloads")
        available = row["bytes"] - relative
        if size < 0 or size > available:
            raise ValueError("Sparse read must stay within one indexed Zarr chunk")
        self.raw.seek(row["packed_offset"] + relative)
        data = self.raw.read(size)
        self.pos += len(data)
        return data

    def close(self):
        self.raw.close()
        super().close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--staging-root", required=True)
    p.add_argument("--census", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--segment-mib", type=int, default=256)
    p.add_argument("--index-path", help="Reuse immutable original archive index if available")
    p.add_argument("--max-bytes", type=int, help="Fail before copying if selected chunk payload exceeds budget")
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
    if "source_census" in group:
        result = stage_pilot_group(
            a.input, a.staging_root, source, group, a.segment_mib * 1024**2, a.index_path, a.max_bytes
        )
    else:
        result = stage_group(a.input, a.staging_root, source, group, a.segment_mib * 1024**2)
    print(json.dumps({"event": "staging_complete", "manifest": str(result)}), flush=True)


if __name__ == "__main__":
    main()
