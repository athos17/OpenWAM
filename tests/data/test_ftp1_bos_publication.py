"""Tiny local fixtures exercise the BOS publisher without a BOS connection."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/data/ftp1/publish_bos.py"
SPEC = importlib.util.spec_from_file_location("ftp1_publish_bos", SCRIPT)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
BUDGET = 1024**2


@pytest.fixture
def tree(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    files = {
        "data/chunk-000/file-000.parquet": b"tiny-numeric-fixture",
        "meta/conversion_complete.json": b'{"signature":"frozen","episodes":5}\n',
        "meta/source_definition.json": b'{"fps":60,"time_status":"nominal","arm":"right"}\n',
        "preview/report.json": b'{"status":"unverified_preview_only","units":"relative_not_pressure"}\n',
        "tactile/manifest.json": b'{"calibration_status":"unverified","tactile_enabled":false}\n',
        "videos/camera/file-000.mp4": b"tiny-video-fixture" * 10,
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return source, destination, files


def publish(tree, **kwargs):
    return publisher.publish(tree[0], tree[1], max_bytes=BUDGET, **kwargs)


def test_byte_exact_manifest_last_preserves_scientific_identity(tree, monkeypatch):
    source, destination, files = tree
    write_record = publisher.write_record
    order = []

    def observe(path, data):
        order.append(path)
        if path.name == publisher.MANIFEST:
            assert not path.exists()
            for relative, content in files.items():
                assert (destination / relative).read_bytes() == content
        else:
            assert not (destination / publisher.MANIFEST).exists()
        return write_record(path, data)

    def forbidden(*args, **kwargs):
        raise AssertionError("Object publication must not use rename")

    monkeypatch.setattr(publisher, "write_record", observe)
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(os, "replace", forbidden)
    summary = publish(tree)
    assert summary["complete"]
    assert summary["copied_files"] == len(files)
    assert summary["scientific_metadata"] == "preserved_verbatim_without_reclassification"
    assert order[-1] == destination / publisher.MANIFEST
    manifest = json.loads(order[-1].read_text())
    assert manifest["complete"] and manifest["integrity_scope"] == "byte_exact_transport_only"
    for entry in manifest["files"]:
        assert entry["bytes"] == len(files[entry["path"]])
        assert entry["sha256"] == hashlib.sha256(files[entry["path"]]).hexdigest()
        assert (source / entry["path"]).read_bytes() == files[entry["path"]]
    assert sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) == summary["publication_bytes"]


def test_interruption_recovers_owned_partial_and_skips_verified_file(tree, monkeypatch):
    source, destination, files = tree
    original_copy = publisher.copy_stream
    calls = []

    def interrupt(src, dst, expected, *, overwrite):
        assert not (destination / publisher.MANIFEST).exists()
        calls.append(dst)
        if len(calls) == 2:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes()[:4])
            raise KeyboardInterrupt
        original_copy(src, dst, expected, overwrite=overwrite)

    monkeypatch.setattr(publisher, "copy_stream", interrupt)
    with pytest.raises(KeyboardInterrupt):
        publish(tree)
    assert not (destination / publisher.MANIFEST).exists()
    completed = calls[0]
    completed_mtime = completed.stat().st_mtime_ns
    monkeypatch.setattr(publisher, "copy_stream", original_copy)
    summary = publish(tree)
    assert summary["recovered_files"] == 1
    assert summary["skipped_files"] == 1
    assert completed.stat().st_mtime_ns == completed_mtime
    for relative, content in files.items():
        assert (destination / relative).read_bytes() == content
    summary = publish(tree)
    assert summary["copied_files"] == 0 and summary["skipped_files"] == len(files)


@pytest.mark.parametrize("committed", [False, True])
def test_same_size_corruption_of_verified_object_fails(tree, monkeypatch, committed):
    source, destination, files = tree
    if committed:
        publish(tree)
    else:
        original_copy = publisher.copy_stream
        count = 0

        def interrupt(src, dst, expected, *, overwrite):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt
            original_copy(src, dst, expected, overwrite=overwrite)

        monkeypatch.setattr(publisher, "copy_stream", interrupt)
        with pytest.raises(KeyboardInterrupt):
            publish(tree)
        monkeypatch.setattr(publisher, "copy_stream", original_copy)
    relative = sorted(files)[0]
    target = destination / relative
    old_stat = target.stat()
    damaged = b"!" * len(files[relative])
    target.write_bytes(damaged)
    os.utime(target, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="checksum conflict"):
        publish(tree)
    assert target.read_bytes() == damaged
    assert (destination / publisher.MANIFEST).exists() == committed


def test_existing_unowned_conflict_never_gets_intent(tree):
    source, destination, files = tree
    relative = sorted(files)[-1]
    target = destination / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"!" * len(files[relative]))
    with pytest.raises(ValueError, match="checksum conflict"):
        publish(tree)
    assert not (destination / publisher.STATE).exists()
    assert not (destination / publisher.MANIFEST).exists()


def test_close_is_not_enough_readback_detects_same_size_corruption(tree, monkeypatch):
    original_copy = publisher.copy_stream

    def corrupt(src, dst, expected, *, overwrite):
        original_copy(src, dst, expected, overwrite=overwrite)
        dst.write_bytes(b"!" * expected["bytes"])

    monkeypatch.setattr(publisher, "copy_stream", corrupt)
    with pytest.raises(ValueError, match="readback checksum mismatch"):
        publish(tree)
    assert not (tree[1] / publisher.MANIFEST).exists()
    assert not list((tree[1] / publisher.STATE).glob("*.verified.json"))


def test_source_change_prevents_manifest(tree, monkeypatch):
    original_copy = publisher.copy_stream

    def change_source(src, dst, expected, *, overwrite):
        original_copy(src, dst, expected, overwrite=overwrite)
        src.write_bytes(b"!" * expected["bytes"])

    monkeypatch.setattr(publisher, "copy_stream", change_source)
    with pytest.raises(ValueError, match="Source changed during publication"):
        publish(tree)
    assert not (tree[1] / publisher.MANIFEST).exists()


def test_budget_payload_and_receipts_rejected_before_destination_writes(tree):
    payload = sum(len(value) for value in tree[2].values())
    for budget in (payload - 1, payload):
        with pytest.raises(ValueError, match="exceeds byte budget"):
            publisher.publish(tree[0], tree[1], max_bytes=budget)
        assert not tree[1].exists()
    for budget in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            publisher.publish(tree[0], tree[1], max_bytes=budget)


def test_temporary_files_and_directories_are_excluded_and_dry_run_does_not_write(tree):
    source, destination, files = tree
    for name in ("unfinished.mp4.partial", "cache.tmp.123", "nested.partial/not-a-payload"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * BUDGET)
    summary = publish(tree, dry_run=True)
    assert not summary["complete"] and not destination.exists()
    assert len(summary["excluded_temporary_paths"]) == 3
    assert summary["files"] == len(files)
    publish(tree)
    assert not (destination / "unfinished.mp4.partial").exists()


@pytest.mark.parametrize("relative", ["", ".", "..", "../escape", "/absolute", "a/../b", "a//b", "a/./b", "a\\b", "C:escape", "a\nb"])
def test_unsafe_relative_paths_rejected(relative):
    with pytest.raises(ValueError, match="Unsafe relative path"):
        publisher.safe_relative(relative)


def test_overlap_and_alias_overlap_rejected(tree, tmp_path):
    source = tree[0]
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    for destination in (source, source / "nested", source.parent, alias, alias / "nested"):
        with pytest.raises(ValueError, match="overlap or nest"):
            publisher.publish(source, destination, max_bytes=BUDGET)


def test_source_and_destination_symlinks_rejected(tree, tmp_path):
    source, destination, _ = tree
    outside = tmp_path / "outside"
    outside.mkdir()
    (source / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="source symlink"):
        publish(tree)
    (source / "linked").unlink()
    destination.mkdir()
    (destination / "meta").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        publish(tree)
    assert not list(outside.iterdir())


def test_recovery_of_interrupted_private_record_and_last_manifest(tree, monkeypatch):
    original_write = publisher.write_record
    manifest_path = tree[1] / publisher.MANIFEST

    def interrupt_manifest(path, content):
        if path == manifest_path:
            path.write_bytes(content[:len(content) // 2])
            raise KeyboardInterrupt
        original_write(path, content)

    monkeypatch.setattr(publisher, "write_record", interrupt_manifest)
    with pytest.raises(KeyboardInterrupt):
        publish(tree)
    with pytest.raises(json.JSONDecodeError):
        json.loads(manifest_path.read_text())
    monkeypatch.setattr(publisher, "write_record", original_write)
    summary = publish(tree)
    assert summary["copied_files"] == 0
    assert json.loads(manifest_path.read_text())["complete"]
    plan = tree[1] / publisher.STATE / "plan.json"
    plan.write_text("damaged")
    with pytest.raises(ValueError, match="Conflicting publication record"):
        publish(tree)


def test_interrupted_intent_record_recovers_without_uploading_early(tree, monkeypatch):
    original_write = publisher.write_record

    def interrupt_intent(path, content):
        if path.name.endswith(".intent.json"):
            path.write_bytes(content[:12])
            raise KeyboardInterrupt
        original_write(path, content)

    monkeypatch.setattr(publisher, "write_record", interrupt_intent)
    with pytest.raises(KeyboardInterrupt):
        publish(tree)
    assert not (tree[1] / sorted(tree[2])[0]).exists()
    assert not (tree[1] / publisher.MANIFEST).exists()
    monkeypatch.setattr(publisher, "write_record", original_write)
    assert publish(tree)["copied_files"] == len(tree[2])


def test_walk_errors_are_not_silently_published_as_a_smaller_dataset(tree, monkeypatch):
    def failing_walk(root, *, followlinks, onerror):
        onerror(PermissionError("unreadable source subtree"))
        return iter(())

    monkeypatch.setattr(os, "walk", failing_walk)
    with pytest.raises(PermissionError, match="unreadable source subtree"):
        publish(tree)
    assert not tree[1].exists()


def test_cli_outputs_machine_readable_success_and_failure(tree):
    command = [sys.executable, str(SCRIPT), "--source", str(tree[0]), "--destination", str(tree[1])]
    result = subprocess.run(command + ["--max-bytes", str(BUDGET)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["event"] == "bos_publication_complete"
    result = subprocess.run(command + ["--max-bytes", "1"], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["event"] == "bos_publication_failed"


def test_verify_after_local_source_cleanup_uses_trusted_manifest_hash(tree):
    summary = publish(tree)
    shutil.rmtree(tree[0])
    verified = publisher.verify_published(tree[1], expected_manifest_sha256=summary["manifest_sha256"], max_bytes=BUDGET)
    assert verified["complete"]
    assert not verified["verification"]["independent_bos_api_verified"]
    with pytest.raises(ValueError, match="trusted receipt"):
        publisher.verify_published(tree[1], expected_manifest_sha256="0" * 64, max_bytes=BUDGET)
    target = tree[1] / sorted(tree[2])[0]
    target.write_bytes(b"!" * target.stat().st_size)
    with pytest.raises(ValueError, match="checksum conflict"):
        publisher.verify_published(tree[1], expected_manifest_sha256=summary["manifest_sha256"], max_bytes=BUDGET)
