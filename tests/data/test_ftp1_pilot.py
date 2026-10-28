"""Pilot episodes preserve exact source frames while avoiding whole-task copies."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("numcodecs")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def tiny_pilot(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/data/ftp1"))
    import pilot

    source = tmp_path / "input" / "Tiny"
    source.mkdir(parents=True)
    archive = source / "tiny.tar"
    arrays = {
        "data/camera_ego_rgb": np.arange(30 * 8 * 8 * 3, dtype=np.uint8).reshape(30, 8, 8, 3),
        "data/right_tactile_data_gripper": np.arange(30 * 2 * 8 * 8 * 3, dtype=np.uint8).reshape(30, 2, 8, 8, 3),
        "data/state": np.arange(60, dtype=np.float32).reshape(30, 2),
        "data/sub_task_instruction": np.array(["task"] * 30, dtype="<U4"),
    }
    with tarfile.open(archive, "w") as tar:
        for key, value in arrays.items():
            shape = (4, *value.shape[1:])
            cfg = {
                "shape": list(value.shape),
                "chunks": list(shape),
                "dtype": value.dtype.str,
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
            }
            entries = [(f"Tiny/task.zarr/{key}/.zarray", json.dumps(cfg).encode())]
            for n, begin in enumerate(range(0, len(value), 4)):
                if key.endswith("tactile_data_gripper") and n == 1:
                    continue
                chunk = np.zeros(shape, dtype=value.dtype)
                chunk[: min(4, len(value) - begin)] = value[begin : begin + 4]
                suffix = ".".join([str(n)] + ["0"] * (value.ndim - 1))
                entries.append((f"Tiny/task.zarr/{key}/{suffix}", chunk.tobytes()))
            for name, raw in entries:
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
    original = {
        "group": "Tiny/task.zarr",
        "offset": 0,
        "end_offset": archive.stat().st_size,
        "frames": 30,
        "episodes": 10,
        "episode_ends": list(range(3, 31, 3)),
    }
    selected = pilot.select_episodes(original, [1, 7, 9])
    definition = tmp_path / "Tiny.json"
    definition.write_text(
        json.dumps(
            {
                "source": "Tiny",
                "nominal_fps": 10,
                "embodiment": "test",
                "intensity_status": "unverified",
                "tactile_representation": "optical_rgb",
            }
        )
    )
    return source.parent, definition, original, selected, arrays


def test_sparse_direct_and_reuse_are_frame_exact(tmp_path, tiny_pilot):
    import av
    import convert
    import pyarrow.parquet as pq
    import stage
    import validate

    source, definition, original, selected, arrays = tiny_pilot
    full = convert.convert_group((str(source), str(tmp_path / "full"), str(definition), original, 1))
    validate.validate_bucket(full["bucket"])
    direct = convert.convert_group((str(source), str(tmp_path / "direct"), str(definition), selected, 1))
    manifest = stage.stage_pilot_group(
        source,
        tmp_path / "sparse",
        "Tiny",
        selected,
        segment_bytes=512,
        index_path=Path(full["bucket"]) / "meta/archive_index.json",
    )
    meta = stage.load_manifest(manifest)
    assert meta["packed_bytes"] < original["end_offset"] / 2
    sparse = convert.convert_group(
        ("/not/mounted", str(tmp_path / "staged"), str(definition), selected, 1, str(manifest))
    )
    reused = convert.convert_group(
        ("/not/mounted", str(tmp_path / "reused"), str(definition), selected, 1, None, {"reuse_bucket": full["bucket"]})
    )
    source_frames = [3, 4, 5, 21, 22, 23, 27, 28, 29]
    expected = arrays["data/camera_ego_rgb"][source_frames]
    tables = []
    for result in [direct, sparse, reused]:
        bucket = Path(result["bucket"])
        validation = validate.validate_bucket(bucket)
        assert validation["frames"] == 9 and validation["episodes"] == 3
        table = pq.read_table(bucket / "data/chunk-000/file-000.parquet")
        tables.append(table)
        assert table["observation.ftp1.source_frame_index"].to_pylist() == source_frames
        assert table["observation.ftp1.source_episode_index"].to_pylist() == [1] * 3 + [7] * 3 + [9] * 3
        assert (
            table["observation.tactile_rgb.right_tactile_data_gripper.0.valid"].to_pylist()
            == [True, False, False] + [True] * 6
        )
        eps = pq.read_table(bucket / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        assert [x["source/episode_index"] for x in eps] == [1, 7, 9]
        assert [x["source/from_frame"] for x in eps] == [3, 21, 27]
        with av.open(str(bucket / "videos/observation.images.camera_ego_rgb/chunk-000/file-000.mp4")) as container:
            decoded = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
        np.testing.assert_array_equal(decoded, expected)
    assert direct["signature"] == sparse["signature"] == reused["signature"]
    assert tables[0].equals(tables[1]) and tables[0].equals(tables[2])
    assert stage.stage_pilot_group(source, tmp_path / "sparse", "Tiny", selected, segment_bytes=512) == manifest
    segment = manifest.parent / meta["segments"][0]["file"]
    with segment.open("r+b") as handle:
        handle.write(b"!")
    with pytest.raises(ValueError, match="checksum"):
        stage.load_manifest(manifest)


def test_pilot_budget_checked_before_payload_copy(tmp_path, tiny_pilot):
    import stage

    source, _, _, selected, _ = tiny_pilot
    with pytest.raises(ValueError, match="no payload was staged"):
        stage.stage_pilot_group(source, tmp_path / "staged", "Tiny", selected, max_bytes=1)
    assert not list((tmp_path / "staged").rglob("segment-*.bin"))


def test_pilot_selection_is_explicit_and_independent(tmp_path, tiny_pilot):
    import convert
    import pilot

    _, definition, original, selected, _ = tiny_pilot
    before = json.dumps(original, sort_keys=True)
    census, plan = pilot.make_plan([{"dataset": "Tiny", "groups": [original]}], definition.parent, 5)
    assert plan["total_episodes"] == 5
    assert census[0]["groups"][0]["source_census"] == original
    assert json.dumps(original, sort_keys=True) == before
    bad = dict(selected, selected_source_episode_indices=[1, 1, 7])
    with pytest.raises(ValueError, match="unique"):
        convert.resolve_selection(bad)
    bad = dict(selected, frames=100)
    with pytest.raises(ValueError, match="frame counts"):
        convert.resolve_selection(bad)
    with pytest.raises(ValueError, match="excluded"):
        pilot.make_plan([{"dataset": "Tiny", "groups": [original]}], definition.parent, sources=["sharpa"])


def test_reuse_rejects_modified_assets(tmp_path, tiny_pilot):
    import convert
    import validate

    source, definition, original, selected, _ = tiny_pilot
    full = convert.convert_group((str(source), str(tmp_path / "full"), str(definition), original, 1))
    validate.validate_bucket(full["bucket"])
    info = Path(full["bucket"]) / "meta/info.json"
    info.write_text(info.read_text() + " ")
    with pytest.raises(ValueError, match="changed since validation"):
        convert.convert_group(
            (
                "/not/mounted",
                str(tmp_path / "pilot"),
                str(definition),
                selected,
                1,
                None,
                {"reuse_bucket": full["bucket"]},
            )
        )


def test_partial_pilot_resume_rejects_same_size_different_episodes(tmp_path, tiny_pilot):
    import convert
    import pilot

    source, definition, original, selected, _ = tiny_pilot
    result = convert.convert_group((str(source), str(tmp_path / "out"), str(definition), selected, 1))
    (Path(result["bucket"]) / "meta/conversion_complete.json").unlink()
    changed = pilot.select_episodes(original, [0, 2, 4])
    assert changed["frames"] == selected["frames"]
    with pytest.raises(ValueError, match="different immutable selection"):
        convert.convert_group((str(source), str(tmp_path / "out"), str(definition), changed, 1))


def test_explicit_smaller_source_override_is_reproducible(tmp_path, tiny_pilot):
    import pilot

    _, definition, original, _, _ = tiny_pilot
    override = {"Tiny": {"group": original["group"], "episode_indices": [1, 7], "reason": "bounded cache"}}
    census, plan = pilot.make_plan(
        [{"dataset": "Tiny", "groups": [original]}], definition.parent, 5, episode_overrides=override
    )
    assert census[0]["groups"][0]["selected_source_episode_indices"] == [1, 7]
    assert plan["total_episodes"] == 2 and plan["total_frames"] == 6
    assert plan["episode_overrides"] == override


def test_real_sparse_worker_publication_matches_training_contract(tmp_path, tiny_pilot, monkeypatch):
    import hashlib
    import importlib.util
    from types import SimpleNamespace

    import coordinate_pilot
    import run_pilot_worker
    import stage

    source, definition, _, selected, _ = tiny_pilot
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    (definitions / "Tiny.json").write_bytes(definition.read_bytes())
    control, work = tmp_path / "control", tmp_path / "work"
    output = work / "output"
    (control / "ready").mkdir(parents=True)
    plan = {
        "run_id": "real-pilot-worker-fixture",
        "worker_count": 1,
        "jobs": [{"source": "Tiny", "group": selected}],
        "output": str(output),
        "definitions": str(definitions),
        "weights": str(tmp_path / "weights-unused-for-unsupported-sensor"),
    }
    ownership_args = SimpleNamespace(
        control=str(control), raw=str(source), bos_output=str(tmp_path / "bos"), max_work_bytes=2 * 1024**2
    )
    owner = coordinate_pilot.initialize_ownership(ownership_args, plan)
    assert coordinate_pilot.load_owned_work(control, plan) == owner
    manifest = stage.stage_pilot_group(source, work / "staging", "Tiny", selected, segment_bytes=512)
    assert 0 < coordinate_pilot.check_work_budget(control, plan) <= ownership_args.max_work_bytes
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    (control / "worker_plan.json").write_text(json.dumps(plan))
    (control / "ready/Tiny.json").write_text(
        json.dumps({"source": "Tiny", "plan_sha256": fingerprint, "staged_manifest": str(manifest)})
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    run_pilot_worker.worker(SimpleNamespace(control=str(control), wait_seconds=1, encoder_threads=1))
    spec = importlib.util.spec_from_file_location(
        "ftp_pilot_training_publication", ROOT / "openwam/dataloader/publication.py"
    )
    publication = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publication)
    contract = publication.publication_contract(output / "Tiny")
    assert contract["frames"] == 9 and contract["episodes"] == 3
    assert len(contract["buckets"]) == 1
    record = json.loads((output / "Tiny/publication.json").read_text())
    assert record["status"] == "complete" and record["training_ready"]
    assert not record["tactile_enabled"] and not record["deform_training_enabled"]
    assert record["deform_status"] == "unsupported_optical_calibration"
    assert (control / "completed/Tiny.json").is_file()
    assert (control / "computed/Tiny.json").is_file()
    assert 0 < coordinate_pilot.check_work_budget(control, plan) <= ownership_args.max_work_bytes
    assert not list(output.rglob("tactile/manifest.json"))


def test_sparse_shared_chunk_is_staged_and_read_once_with_missing_mask(tmp_path, tiny_pilot):
    import convert
    import pilot
    import stage

    source, _, original, _, arrays = tiny_pilot
    # Episode 0 and 1 overlap chunk 0. The following tactile chunk is absent.
    selected = pilot.select_episodes(original, [0, 1])
    path = stage.stage_pilot_group(source, tmp_path / "stage", "Tiny", selected, segment_bytes=512)
    manifest = stage.load_manifest(path)
    entries = [(row["source_offset"], row["bytes"]) for row in manifest["payloads"]]
    assert len(entries) == len(set(entries))
    spec = manifest["archive_index"]["arrays"]["data/right_tactile_data_gripper"]
    assert set(spec["chunks"]) == {"0.0.0.0.0"}
    with stage.SparseArchive(path, manifest) as stream:
        calls = []
        original_read = stream.read

        def counted_read(size):
            calls.append((stream.tell(), size))
            return original_read(size)

        stream.read = counted_read
        data, valid = convert.Array(stream, spec, [[0, 3], [3, 6]]).read()
    assert len(calls) == 1
    np.testing.assert_array_equal(data[:4], arrays["data/right_tactile_data_gripper"][:4])
    np.testing.assert_array_equal(data[4:], 0)
    np.testing.assert_array_equal(valid, [True] * 4 + [False] * 2)
