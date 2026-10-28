"""Held-out metrics score decoded intensities on measured pixels, not flow loss."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts.validation.touchwam_tactile_eval import aggregate, aggregate_by_source, evaluate_sample, metric, score_maps
from tests.test_touchwam_regions import models as regional_models


def sample(size=2):
    return dict(
        training_modality="video_tactile", prompt="hold object", tactile_codec="tactile_intensity_v1",
        tactile_region_names=["right.thumb.tip", "left.thumb.tip"],
        tactile_sensor_names=["sharpa_real", "sharpa_real"],
        tactile_maps=np.full((2, 2, 1, size, size), .2, np.float32),
        tactile_valid=np.array([[True, False], [False, False]]),
        future_tactile_maps=np.full((2, 2, 1, size, size), .8, np.float32),
        future_tactile_valid=np.ones((2, 2), bool), tactile_layout_present=np.ones(2, bool),
        tactile_history_times_s=np.array([-1 / 30, 0]), future_tactile_times_s=np.array([1 / 30, 2 / 30]),
        first_frame_image=Image.new("RGB", (16, 16)),
    )


class EvaluationModel:
    tactile_spec = dict(region_names=["right.thumb.tip", "left.thumb.tip"], sensor_names=["sharpa_real"],
                        history=2, future_frames=2, future_stride=1, fps=30, map_size=2)

    def generate(self, schedule, prompt, **kwargs):
        self.kwargs = kwargs
        return dict(tactile_region_names=["left.thumb.tip", "right.thumb.tip"],
                    future_tactile_maps=np.full((2, 2, 1, 2, 2), .5, np.float32),
                    future_tactile_valid=np.ones((2, 2), bool))


def test_metrics_mask_nan_and_hold_last_has_own_and_common_support():
    target = np.ones((1, 2, 1, 1, 2), np.float32)
    pred = np.zeros_like(target)
    valid = np.ones_like(target, dtype=bool)
    valid[0, 1, 0, 0, 1] = False
    target[~valid] = np.nan
    history = np.full((2, 2, 1, 1, 2), .25, np.float32)
    history[1] = np.nan  # both packets absent; do not substitute the last time slot
    history_valid = np.array([[True, False], [False, False]])
    report, held, common = score_maps(pred, target, valid, history, history_valid, ["a", "b"])
    assert report["all_target_support"]["model"] == dict(
        pixels=3, absolute_error_sum=3., squared_error_sum=3., mae=1., rmse=1.)
    assert report["all_target_support"]["hold_last"]["pixels"] == 2
    assert report["common_history_support"]["model"]["pixels"] == 2
    assert report["common_history_support"]["hold_last"]["mae"] == .75
    assert report["by_region"]["b"]["hold_last"]["mae"] is None
    assert held[0, 0, 0, 0, 0] == .25 and common.sum() == 2
    pred[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        metric(pred, target, valid)


def test_evaluate_never_passes_future_labels_or_fake_robot_state_and_writes_panels(tmp_path):
    model = EvaluationModel()
    row = sample()
    report = evaluate_sample(model, row, schedule=[(1000, 1000), (0, 0)], output_dir=tmp_path)
    assert not any(key in model.kwargs for key in ("future_tactile_maps", "future_tactile_valid", "proprio", "actions"))
    assert model.kwargs["generation_modality"] == "video_tactile"
    assert report["metrics"]["all_target_support"]["model"]["mae"] == pytest.approx(.3)
    assert report["metrics"]["all_target_support"]["zero"]["mae"] == pytest.approx(.8)
    assert report["metrics"]["common_history_support"]["hold_last"]["mae"] == pytest.approx(.6)
    assert (tmp_path / "predictions.npz").exists()
    for panel in report["panels"]:
        with Image.open(tmp_path / panel["path"]) as image:
            assert image.size == (317, 512)
    assert json.loads((tmp_path / "metrics.json").read_text())["schema"] == "touchwam_tactile_eval_v1"
    summary = aggregate([report, report])
    assert summary["all_target_support"]["model"]["pixels"] == 32
    assert summary["all_target_support"]["model"]["mae"] == pytest.approx(.3)


def test_vat_requires_real_robot_fields_and_vt_rejects_them():
    model = EvaluationModel()
    row = sample()
    row["training_modality"] = "video_action_tactile"
    with pytest.raises(ValueError, match="real action"):
        evaluate_sample(model, row, schedule=[(1000, 1000), (0, 0)])
    row.update(action=np.zeros((4, 80)), proprio=np.zeros(80), action_mask=np.ones((4, 80), bool))
    evaluate_sample(model, row, schedule=[(1000, 1000), (0, 0)])
    assert model.kwargs["action_num_frames"] == 5 and model.kwargs["active_action_mask"].shape == (80,)
    row["training_modality"] = "video_tactile"
    with pytest.raises(ValueError, match="cannot carry robot"):
        evaluate_sample(model, row, schedule=[(1000, 1000), (0, 0)])


def test_roi_holes_do_not_change_metrics_or_turn_into_zero_targets():
    model = EvaluationModel()
    row = sample()
    row["tactile_spatial_valid"] = np.ones((2, 1, 2, 2), bool)
    row["tactile_spatial_valid"][:, :, 0, 0] = False
    row["future_tactile_maps"][:, :, :, 0, 0] = np.nan
    row["tactile_maps"][:, :, :, 0, 0] = np.nan
    result = evaluate_sample(model, row, schedule=[(1000, 1000), (0, 0)])
    assert result["metrics"]["all_target_support"]["model"]["pixels"] == 12
    assert result["metrics"]["all_target_support"]["model"]["mae"] == pytest.approx(.3)


def test_generated_validity_cannot_hide_missing_predictions():
    model = EvaluationModel()
    generate = model.generate

    def missing_prediction(*args, **kwargs):
        result = generate(*args, **kwargs)
        result["future_tactile_valid"][0, 0] = False
        return result

    model.generate = missing_prediction
    with pytest.raises(ValueError, match="do not hide missing predictions"):
        evaluate_sample(model, sample(), schedule=[(1000, 1000), (0, 0)])


def test_real_model_vt_generation_reaches_map_metrics_without_action(monkeypatch, tmp_path):
    model = regional_models.__wrapped__(monkeypatch, tmp_path)[2]
    model.tactile_spec["allow_missing_history"] = True
    row = sample(size=8)
    row["first_frame_image"] = [row["first_frame_image"]]  # exact native reader contract
    report = evaluate_sample(model, row, schedule=[(1000, 1000), (0, 0)], output_dir=tmp_path / "eval")
    assert report["metrics"]["all_target_support"]["model"]["pixels"] == 256
    assert np.isfinite(report["metrics"]["all_target_support"]["model"]["rmse"])
    assert model.training  # an inline evaluation must not leave a training model in eval mode


def test_cli_heldout_archive_runs_without_action_normalizer(monkeypatch, tmp_path):
    from scripts.validation.touchwam_tactile_eval import main

    row = sample()
    arrays = {key: value for key, value in row.items() if isinstance(value, (np.ndarray, list))}
    np.savez_compressed(tmp_path / "row.npz", **arrays)
    row["first_frame_image"].save(tmp_path / "frame.png")
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"fake")
    manifest = dict(schema="touchwam_tactile_eval_samples_v1", split="validation", samples=[dict(
        arrays="row.npz", first_frame_image="frame.png", source_id="ego_touch", episode_index=10,
        reference_time_s=1., training_modality="video_tactile", tactile_codec="tactile_intensity_v1", prompt="hold")])
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(manifest))
    model = EvaluationModel()
    model.model_version = "touchwam_regions_v1"
    model.set_dtype_device = lambda *args: None
    model.move_frozen_to_device = lambda *args: None
    model.eval = lambda: None
    model.video_scheduler = model.tactile_scheduler = object()
    monkeypatch.setattr("openwam.train.utils.ckpt_model_loader.build_architecture_from_ckpt_dir",
                        lambda *args, **kw: (None, model, SimpleNamespace()))
    monkeypatch.setattr("openwam.deploy.denoise_schedule.schedule_sync", lambda *args, **kw: [(1000, 1000), (0, 0)])
    main(["--checkpoint", str(checkpoint), "--manifest", str(path), "--output", str(tmp_path / "out"),
          "--device", "cpu", "--dtype", "fp32"])
    report = json.loads((tmp_path / "out/summary.json").read_text())
    assert report["split"] == "validation" and report["samples"][0]["sample"]["source_id"] == "ego_touch"
    manifest["split"] = "train"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="held-out"):
        main(["--checkpoint", str(checkpoint), "--manifest", str(path), "--output", str(tmp_path / "bad")])


def test_rgb_eval_retains_all_color_channels_and_labels_units(tmp_path):
    row = sample()
    row["tactile_codec"] = "tactile_rgb_v1"
    row["tactile_maps"] = np.repeat(row["tactile_maps"], 3, axis=2)
    row["future_tactile_maps"] = np.repeat(row["future_tactile_maps"], 3, axis=2)
    row["future_tactile_maps"][:, :, 0] = 0.1
    model = EvaluationModel()
    model.model_version = "touchwam_regions_v1"
    model.tactile_spec = dict(model.tactile_spec, rgb_enabled=True, rgb_map_size=2, codec="tactile_intensity_v1")
    old_generate = model.generate

    def generate(*args, **kwargs):
        result = old_generate(*args, **kwargs)
        result["tactile_codec"] = "tactile_rgb_v1"
        result["future_tactile_maps"] = np.repeat(result["future_tactile_maps"], 3, axis=2)
        return result

    model.generate = generate
    report = evaluate_sample(model, row, schedule=[(1000, 1000), (0, 0)], output_dir=tmp_path)
    assert model.kwargs["tactile_codec"] == "tactile_rgb_v1"
    assert model.kwargs["tactile_spatial_valid"].shape == (2, 1, 2, 2)
    assert report["units"] == "RGB channel values [0,1]"
    assert report["metrics"]["all_target_support"]["model"]["pixels"] == 48
    with np.load(tmp_path / "predictions.npz") as archive:
        assert archive["prediction"].shape == (2, 2, 3, 2, 2)
        assert archive["tactile_codec"].item() == "tactile_rgb_v1"
    with Image.open(tmp_path / "region_00.png") as panel:
        assert panel.getpixel((140, 48)) == (26, 204, 204)
    scalar_report = evaluate_sample(EvaluationModel(), sample(), schedule=[(1000, 1000), (0, 0)])
    with pytest.raises(ValueError, match="separately"):
        aggregate([report, scalar_report])
    report["sample"] = scalar_report["sample"] = {"source_id": "same_sensor"}
    grouped = aggregate_by_source([report, scalar_report])["same_sensor"]
    assert grouped["metrics"] is None
    assert grouped["by_representation"]["tactile_rgb_v1"]["all_target_support"]["model"]["pixels"] == 48
    assert grouped["by_representation"]["tactile_intensity_v1"]["all_target_support"]["model"]["pixels"] == 16


def test_rgb_eval_rejects_older_nonregional_generation_before_call():
    row = sample()
    row["tactile_codec"] = "tactile_rgb_v1"
    model = EvaluationModel()
    model.tactile_spec = dict(model.tactile_spec, rgb_enabled=True, rgb_map_size=2)
    with pytest.raises(ValueError, match="regional checkpoint"):
        evaluate_sample(model, row, schedule=[(1000, 1000), (0, 0)])
    assert not hasattr(model, "kwargs")
