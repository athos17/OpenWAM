"""A tactile adapter must not quietly replace missing data or sensor physics."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("pyarrow")
pytest.importorskip("cv2")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ftp1_intensity", ROOT / "scripts/data/ftp1/build_intensity.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
REGION = "right.gripper.jaw_a.surface"


def definition():
    return {
        "schema": "touchwam.ftp1.intensity_definition.v1",
        "source": "RH20TCfg7Tactile",
        "semantics": "normal_contact_intensity",
        "verification": {"status": "validated", "evidence": ["synthetic test calibration"]},
        "regions": [
            {
                "region": REGION,
                "sensor_name": "test_uskin",
                "feature": "observation.ftp1.right_tactile_data_gripper",
                "mode": "spatial_axis",
                "spatial_shape": [4, 4],
                "normal_axis": 2,
                "sensor_index": 0,
                "sign": -1,
                "baseline": {
                    "kind": "constant",
                    "value": 100,
                    "verified_no_contact": True,
                    "evidence": "test zero load",
                },
                "normalization": {"bounds": [0, 20]},
            }
        ],
    }


def test_fixed_spatial_normalization_and_direction():
    d = definition()
    module.validate_definition(d, [REGION])
    row = d["regions"][0]
    values = np.full((4, 4), 90, dtype=float)
    out = module.transform_frame(values, row, np.float32(100), size=4)
    assert out.dtype == np.uint8 and np.all(out == 128)
    assert np.all(module.transform_frame(np.full((4, 4), 110), row, 100) == 0)
    assert np.all(module.transform_frame(np.full((4, 4), 70), row, 100) == 255)


def test_never_broadcast_wrench_into_map():
    d = definition()
    d["regions"][0]["spatial_shape"] = [1, 6]
    with pytest.raises(ValueError, match="spatial grid"):
        module.validate_definition(d, [REGION])
    row = definition()["regions"][0]
    with pytest.raises(ValueError, match="shape mismatch"):
        module.transform_frame(np.zeros(6), row, 100)


def test_rgb_proxy_cannot_claim_force_or_bypass_verification():
    d = definition()
    row = d["regions"][0]
    row["mode"] = "optical_change_proxy"
    with pytest.raises(ValueError, match="optical proxy"):
        module.validate_definition(d, [REGION])
    d["semantics"] = "optical_change_proxy"
    with pytest.raises(ValueError, match="authorization"):
        module.validate_definition(d, [REGION])
    d["verification"]["proxy_use_explicitly_approved"] = True
    module.validate_definition(d, [REGION])
    d["verification"]["status"] = "unverified"
    with pytest.raises(ValueError, match="uncalibrated RGB"):
        module.validate_definition(d, [REGION])


def test_baseline_fingerprint_and_roi_are_strict(tmp_path):
    a = tmp_path / "baseline.npy"
    np.save(a, np.ones((2, 3)))
    with pytest.raises(ValueError, match="fingerprint"):
        module.load_asset({"path": str(a), "sha256": "bad"}, tmp_path)
    np.testing.assert_equal(module.load_asset({"path": str(a), "sha256": module.digest(a)}, tmp_path), np.ones((2, 3)))
    row = definition()["regions"][0]
    row["roi_xywh"] = [1, 1, 4, 4]
    with pytest.raises(ValueError, match="ROI"):
        module.transform_frame(np.zeros((4, 4)), row, 100)


def test_duplicate_regions_and_unverified_zero_rejected():
    d = definition()
    d["regions"].append(copy.deepcopy(d["regions"][0]))
    with pytest.raises(ValueError, match="unique"):
        module.validate_definition(d, [REGION])
    d = definition()
    d["regions"][0]["baseline"]["verified_no_contact"] = False
    with pytest.raises(ValueError, match="no-contact"):
        module.validate_definition(d, [REGION])


def test_depth_plugin_is_fingerprinted_and_spatially_registered(tmp_path):
    plugin = tmp_path / "reconstruction.py"
    plugin.write_text(
        "import numpy as np\ndef reconstruct(*, image, baseline, calibration, options):\n    return np.maximum(0, baseline-image[...,0]) * calibration['gain']\n"
    )
    gain = tmp_path / "gain.npy"
    np.save(gain, np.array(0.1, dtype=np.float32))
    d = definition()
    row = d["regions"][0]
    row.update(
        mode="depth_reconstruction",
        reconstruction_output="indentation_depth",
        reconstruction_unit="mm",
        monotonic_response_evidence="synthetic known response",
        reconstruction_plugin={"path": str(plugin), "sha256": module.digest(plugin)},
        calibration_assets={"gain": {"path": str(gain), "sha256": module.digest(gain)}},
        normalization={"bounds": [0, 2]},
    )
    module.validate_definition(d, [REGION])
    baseline, backend = module.assets_for(row, tmp_path)
    out = module.transform_frame(np.full((4, 4, 3), 90), row, baseline, backend, size=4)
    assert np.all(out == 128)
    row["reconstruction_plugin"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="fingerprint"):
        module.assets_for(row, tmp_path)


def test_intensity_sidecar_episode_boundaries_and_missing_frames(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    bucket = tmp_path / "bucket"
    meta = bucket / "meta"
    data = bucket / "data/chunk-000"
    (meta / "episodes/chunk-000").mkdir(parents=True)
    data.mkdir(parents=True)
    feature = "observation.ftp1.right_tactile_data_gripper"
    values = np.full((4, 2, 4, 4, 3), 90, dtype=np.float32)
    pq.write_table(
        pa.table({feature: values.reshape(4, -1).tolist(), feature + ".valid": [True, False, True, True]}),
        data / "file-000.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"episode_index": 0, "length": 2, "dataset_from_index": 0, "dataset_to_index": 2},
                {"episode_index": 1, "length": 2, "dataset_from_index": 2, "dataset_to_index": 4},
            ]
        ),
        meta / "episodes/chunk-000/file-000.parquet",
    )
    info = {
        "fps": 10,
        "total_frames": 4,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {feature: {"dtype": "float32", "shape": [2, 4, 4, 3]}},
    }
    (meta / "info.json").write_text(json.dumps(info))
    (meta / "source_definition.json").write_text(
        json.dumps(
            {
                "source": "RH20TCfg7Tactile",
                "source_id": "ftp1/RH20TCfg7Tactile",
                "physical_time_verified": True,
                "training_modalities": ["video"],
            }
        )
    )
    (meta / "validation.json").write_text("{}")
    (meta / "conversion_complete.json").write_text('{"signature":"synthetic"}')
    config = tmp_path / "adapter.json"
    config.write_text(json.dumps(definition()))
    manifest = module.build_bucket(bucket, config, [REGION])
    assert set(manifest["episodes"]) == {"0", "1"}
    first = manifest["episodes"]["0"]
    np.testing.assert_equal(np.load(bucket / "tactile" / first["valid"]), [[True], [False]])
    maps = np.load(bucket / "tactile" / first["maps"])
    assert np.all(maps[0] == 128) and np.all(maps[1] == 0)
    np.testing.assert_allclose(np.load(bucket / "tactile" / manifest["episodes"]["1"]["timestamps"]), [0, 0.1])
    assert module.build_bucket(bucket, config, [REGION])["adapter_signature"] == manifest["adapter_signature"]


def test_dtact_reference_index_not_lut_zero():
    spec = importlib.util.spec_from_file_location("dtact_lut", ROOT / "scripts/data/ftp1/depth_plugins/dtact_lut.py")
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    image = np.full((4, 4, 3), 96, dtype=np.float32)
    base = np.full((4, 4), 100, dtype=np.float32)
    calibration = {"response_lut": np.array([-0.5, 0.05, 0.07, 0.1, 0.2], np.float32)}
    options = {"gray_channel": 2, "reference_index": 2, "blur_kernel": 1}
    np.testing.assert_allclose(
        backend.reconstruct(image=image, baseline=base, calibration=calibration, options=options), 0.13, atol=1e-7
    )
    assert np.all(
        backend.reconstruct(image=np.full_like(image, 101), baseline=base, calibration=calibration, options=options)
        == 0
    )
    with pytest.raises(ValueError, match="reference_index"):
        backend.reconstruct(image=image, baseline=base, calibration=calibration, options={"gray_channel": 2})
