"""Regression checks for archival conversion and modality safety."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("numcodecs")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ftp1_convert", ROOT / "scripts/data/ftp1/convert.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_split_archive_boundary(tmp_path):
    files = []
    for i, data in enumerate([b"abc", b"defg", b"hi"]):
        p = tmp_path / str(i)
        p.write_bytes(data)
        files.append(p)
    with module.Parts(files) as stream:
        stream.seek(2)
        assert stream.read(5) == b"cdefg"
        stream.seek(-3, 2)
        assert stream.read() == b"ghi"


def test_missing_chunk_is_not_observed_zero(tmp_path):
    file = tmp_path / "data"
    file.write_bytes(np.array([[1, 2], [3, 4]], dtype=np.float32).tobytes())
    cfg = {"shape": [5, 2], "chunks": [2, 2], "dtype": "<f4", "fill_value": 0, "order": "C", "compressor": None}
    with module.Parts([file]) as stream:
        arr = module.Array(stream, {"config": cfg, "chunks": {"0.0": [0, 16]}})
        data, valid = arr.read()
    np.testing.assert_equal(data[:2], [[1, 2], [3, 4]])
    np.testing.assert_equal(data[2:], 0)
    np.testing.assert_equal(valid, [True, True, False, False, False])


def test_definitions_do_not_invent_pressure_or_commands():
    paths = sorted((ROOT / "configs/data_sources/ftp1").glob("*.json"))
    definitions = [json.loads(p.read_text()) for p in paths if p.name != "archive_census.json"]
    assert len(definitions) == 14
    assert "sharpa" not in {x["source"] for x in definitions}
    for definition in definitions:
        assert definition["action_enabled"] is False
        assert definition["intensity_enabled"] is False
        if definition["nominal_fps"] is None:
            assert definition["time_status"] == "unresolved_export_fps"


def test_real_v3_conversion_keeps_episode_and_validity(tmp_path, monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("imageio_ffmpeg")
    import pyarrow.parquet as pq

    source = tmp_path / "input" / "Tiny"
    source.mkdir(parents=True)
    arrays = {
        "data/camera_ego_rgb": np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3),
        "data/sub_task_instruction": np.array(["one", "one", "two", "two"], dtype="<U3"),
        "data/right_tactile_data_gripperforce": np.array([1, 2, 3, 4], dtype=np.float32).reshape(4, 1, 1),
    }
    archive = source / "tiny.tar"
    with tarfile.open(archive, "w") as tar:
        for key, value in arrays.items():
            cfg = {
                "shape": list(value.shape),
                "chunks": list(value.shape),
                "dtype": value.dtype.str,
                "compressor": None,
                "filters": None,
                "fill_value": 0,
                "order": "C",
                "zarr_format": 2,
            }
            for name, raw in [
                (f"Tiny/test.zarr/{key}/.zarray", json.dumps(cfg).encode()),
                (f"Tiny/test.zarr/{key}/" + ".".join(["0"] * value.ndim), value.tobytes()),
            ]:
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
    definition = tmp_path / "definition.json"
    definition.write_text(
        json.dumps(
            {
                "source": "Tiny",
                "nominal_fps": 10,
                "embodiment": "single_gripper",
                "intensity_status": "requires_calibration",
            }
        )
    )
    result = module.convert_group(
        (
            str(source.parent),
            str(tmp_path / "out"),
            str(definition),
            {
                "group": "Tiny/test.zarr",
                "offset": 0,
                "end_offset": archive.stat().st_size,
                "frames": 4,
                "episode_ends": [2, 4],
            },
            1,
        )
    )
    bucket = Path(result["bucket"])
    info = json.loads((bucket / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 2
    table = pq.read_table(bucket / "data/chunk-000/file-000.parquet").to_pydict()
    assert table["episode_index"] == [0, 0, 1, 1]
    assert table["timestamp"] == [0, 0.1, 0, 0.1]
    assert "action" not in table
    eps = pq.read_table(bucket / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert eps[1]["dataset_from_index"] == 2
    assert eps[1]["videos/observation.images.camera_ego_rgb/from_timestamp"] == 0.2
    assert not (bucket / "tactile/manifest.json").exists()
    assert info["features"]["observation.ftp1.right_tactile_data_gripperforce"]["shape"] == [1]
    assert info["features"]["observation.ftp1.right_tactile_data_gripperforce"]["native_shape"] == [1, 1]
    monkeypatch.syspath_prepend(str(ROOT / "scripts/data/ftp1"))
    import validate

    checked = validate.validate_bucket(bucket)
    assert checked["frames"] == 4 and checked["episodes"] == 2
    assert validate.validate_bucket(bucket) == checked
    # Frame-exact RGB roundtrip and resumability are checked on actual encoding.
    import av

    with av.open(str(bucket / "videos/observation.images.camera_ego_rgb/chunk-000/file-000.mp4")) as container:
        decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    np.testing.assert_equal(decoded, arrays["data/camera_ego_rgb"])
    assert (
        module.convert_group(
            (
                str(source.parent),
                str(tmp_path / "out"),
                str(definition),
                {
                    "group": "Tiny/test.zarr",
                    "offset": 0,
                    "end_offset": archive.stat().st_size,
                    "frames": 4,
                    "episode_ends": [2, 4],
                },
                1,
            )
        )["signature"]
        == result["signature"]
    )
    # A cached receipt must not hide a later malformed feature definition.
    info["features"]["observation.ftp1.right_tactile_data_gripperforce"]["shape"] = [7]
    (bucket / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(AssertionError):
        validate.validate_bucket(bucket)


def test_publication_rejects_wrong_task_with_matching_counts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/data/ftp1"))
    import validate

    census = [
        {
            "dataset": "Tiny",
            "frames": 4,
            "episodes": 1,
            "groups": [{"group": "Tiny/task.zarr", "frames": 4, "episode_ends": [4]}],
        }
    ]
    wrong = str(tmp_path / "Tiny" / "wrong-task")
    result = validate.publish_sources(
        tmp_path,
        census,
        {
            wrong: {
                "bucket": wrong,
                "frames": 4,
                "episodes": 1,
                "physical_time_verified": True,
            }
        },
        {},
    )
    assert result["Tiny"]["ready"] is False
    assert "absent" in result["Tiny"]["errors"][wrong]
