"""Range staging must preserve absolute tar offsets and source identities."""

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("numcodecs")
ROOT = Path(__file__).resolve().parents[2]


def test_split_archive_stage_keeps_offsets_and_detects_corruption(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/data/ftp1"))
    import convert
    import stage

    source = tmp_path / "source" / "Tiny"
    source.mkdir(parents=True)
    archive = tmp_path / "complete.tar"
    with tarfile.open(archive, "w") as tar:
        for name, data in [
            ("Tiny/prefix.bin", b"x" * 1100),
            (
                "Tiny/task.zarr/data/state/.zarray",
                json.dumps(
                    {
                        "shape": [4, 2],
                        "chunks": [4, 2],
                        "dtype": "<f4",
                        "compressor": None,
                        "fill_value": 0,
                        "order": "C",
                        "filters": None,
                    }
                ).encode(),
            ),
            ("Tiny/task.zarr/data/state/0.0", np.arange(8, dtype=np.float32).tobytes()),
            ("Tiny/next.zarr/dummy", b"next"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with tarfile.open(archive) as tar:
        members = list(tar)
    begin = next(m.offset for m in members if m.name.endswith("/.zarray"))
    end = next(m.offset for m in members if "next.zarr" in m.name)
    assert begin > 0
    raw = archive.read_bytes()
    split = begin + 777
    files = [source / "tiny.tar.part-aa", source / "tiny.tar.part-ab"]
    files[0].write_bytes(raw[:split])
    files[1].write_bytes(raw[split:])
    group = {
        "group": "Tiny/task.zarr",
        "offset": begin,
        "end_offset": end,
        "frames": 4,
        "episode_ends": [4],
        "episodes": 1,
    }
    manifest_path = stage.stage_group(source.parent, tmp_path / "stage", "Tiny", group, segment_bytes=512)
    manifest = stage.load_manifest(manifest_path, verify=True)
    with stage.RangeArchive(manifest_path, manifest) as stream:
        assert stream.tell() == begin
        assert stream.read() == raw[begin:end]
        stream.seek(begin + 600)
        assert stream.read(1000) == raw[begin + 600 : min(begin + 1600, end)]
        with pytest.raises(ValueError, match="outside"):
            stream.seek(begin - 1)
    direct = convert.index_group(files, group, tmp_path / "direct.json")
    staged = convert.index_group(
        [],
        group,
        tmp_path / "staged.json",
        lambda: stage.RangeArchive(manifest_path, manifest),
        manifest["source_fingerprint"],
    )
    assert direct == staged
    segment = manifest_path.parent / manifest["segments"][0]["file"]
    mtime = segment.stat().st_mtime_ns
    manifest_path.unlink()
    resumed = stage.stage_group(source.parent, tmp_path / "stage", "Tiny", group, segment_bytes=512)
    assert segment.stat().st_mtime_ns == mtime
    assert stage.load_manifest(resumed)["range_sha256"] == manifest["range_sha256"]
    with segment.open("r+b") as f:
        f.write(b"!")
    with pytest.raises(ValueError, match="checksum"):
        stage.load_manifest(resumed, verify=True)


def test_shared_queue_converts_and_reuses_complete_output(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/data/ftp1"))
    import stage_queue

    source = tmp_path / "input" / "Tiny"
    source.mkdir(parents=True)
    archive = source / "tiny.tar"
    value = np.arange(8, dtype=np.float32).reshape(4, 2)
    cfg = {
        "shape": [4, 2],
        "chunks": [4, 2],
        "dtype": "<f4",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
    }
    with tarfile.open(archive, "w") as tar:
        for name, data in [
            ("Tiny/task.zarr/data/state/.zarray", json.dumps(cfg).encode()),
            ("Tiny/task.zarr/data/state/0.0", value.tobytes()),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    group = {
        "group": "Tiny/task.zarr",
        "offset": 0,
        "end_offset": archive.stat().st_size,
        "frames": 4,
        "episode_ends": [4],
        "episodes": 1,
    }
    census = tmp_path / "census.json"
    census.write_text(json.dumps([{"dataset": "Tiny", "groups": [group]}]))
    defs = tmp_path / "defs"
    defs.mkdir()
    (defs / "Tiny.json").write_text(
        json.dumps({"source": "Tiny", "nominal_fps": 10, "embodiment": "test", "intensity_status": "unverified"})
    )
    mounted = tmp_path / "portable-mount"
    mounted.symlink_to(tmp_path / "actual-mount", target_is_directory=True)
    (tmp_path / "actual-mount").mkdir()
    args = SimpleNamespace(
        input=str(source.parent),
        staging_root=str(mounted / "stage"),
        queue=str(tmp_path / "queue"),
        output=str(mounted / "out"),
        definitions=str(defs),
        census=str(census),
        sources=["Tiny"],
        max_pending=2,
        segment_mib=1,
        worker_count=1,
        run_id="test-run",
    )
    stage_queue.producer(args)
    queue_state = json.loads((Path(args.queue) / "state.json").read_text())
    assert "/portable-mount/" in queue_state["output"]
    ready = json.loads(next((Path(args.queue) / "ready").glob("*.json")).read_text())
    assert "/portable-mount/" in ready["manifest"]
    worker_args = SimpleNamespace(
        queue=args.queue, wait_timeout=1, max_attempts=1, encoder_threads=1, rank=0, world_size=1, run_id="test-run"
    )
    stage_queue.worker(worker_args)
    complete = list((Path(args.queue) / "completed").glob("*.json"))
    assert len(complete) == 1
    assert json.loads(complete[0].read_text())["validation"]["frames"] == 4
    # A new queue pointing at the same source/output must not recopy its range.
    args.queue = str(tmp_path / "queue2")
    args.staging_root = str(tmp_path / "must-not-stage")
    stage_queue.producer(args)
    assert not Path(args.staging_root).exists()
    complete2 = list((Path(args.queue) / "completed").glob("*.json"))
    assert json.loads(complete2[0].read_text())["reused_without_staging"]
    worker_args.queue = args.queue
    stage_queue.worker(worker_args)


def test_fixed_shards_do_not_overlap_even_without_distributed_flock(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    monkeypatch.syspath_prepend(str(ROOT / "scripts/data/ftp1"))
    import stage_queue

    jobs = [{"source": "Tiny", "group": {"group": f"Tiny/task-{i}.zarr", "frames": 1}} for i in range(40)]
    state = {
        "schema": stage_queue.SCHEMA,
        "fingerprint": "immutable-test-queue",
        "run_id": "run1",
        "worker_count": 2,
        "staging_complete": True,
        "jobs": jobs,
        "definitions": str(tmp_path),
        "output": str(tmp_path / "out"),
    }
    for sub in ["ready", "claims", "completed", "errors"]:
        (tmp_path / sub).mkdir()
    (tmp_path / "state.json").write_text(json.dumps(state))
    for j in jobs:
        key = stage_queue.key_for(j["group"])
        (tmp_path / "ready" / f"{key}.json").write_text(
            json.dumps(dict(j, fingerprint=state["fingerprint"], manifest="unused"))
        )
    written, lock = set(), Lock()

    def convert(job):
        group = job[3]["group"]
        with lock:
            assert group not in written, "two nodes attempted to write the same bucket"
            written.add(group)
        return {"bucket": str(tmp_path / "out" / group), "frames": 1}

    monkeypatch.setattr(stage_queue.fcntl, "flock", lambda *args: None)
    monkeypatch.setattr(stage_queue, "load_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(stage_queue, "convert_group", convert)
    monkeypatch.setattr(stage_queue, "validate_bucket", lambda *args: {"frames": 1})
    arguments = [
        SimpleNamespace(
            queue=str(tmp_path), rank=r, world_size=2, run_id="run1", encoder_threads=1, wait_timeout=1, max_attempts=1
        )
        for r in range(2)
    ]
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(stage_queue.worker, arguments))
    assert written == {j["group"]["group"] for j in jobs}
    assert len(list((tmp_path / "completed").glob("*.json"))) == len(jobs)
    for p in (tmp_path / "claims").glob("*.json"):
        claim = json.loads(p.read_text())
        assert claim["rank"] == stage_queue.assigned_rank(p.stem, 2)
    # Resuming every rank must be a pure skip over completed task receipts.
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(stage_queue.worker, arguments))


def test_queue_rejects_wrong_rank_count_run_or_legacy_schema(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/data/ftp1"))
    import stage_queue

    state = {"schema": stage_queue.SCHEMA, "worker_count": 2, "run_id": "stable"}
    args = SimpleNamespace(rank=None, world_size=None, run_id="stable")
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match="Explicit"):
        stage_queue.worker_identity(args, state)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert stage_queue.worker_identity(args, state) == (1, 2)
    for rank, size, run in [(0, 1, "stable"), (2, 2, "stable"), (0, 2, "different")]:
        with pytest.raises(ValueError):
            stage_queue.worker_identity(SimpleNamespace(rank=rank, world_size=size, run_id=run), state)
    with pytest.raises(ValueError, match="legacy"):
        stage_queue.worker_identity(args, dict(state, schema="touchwam.ftp1.stage_queue.v1"))
