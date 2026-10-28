"""Preflight must reject incompatible data before constructing the large model."""

from contextlib import nullcontext
import json

from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch

from scripts import preflight_touchwam as preflight
from openwam.tactile.registry import REGION_NAMES


@pytest.fixture
def tactile_spec():
    # Deliberately different from the production 4/8/32/30 settings: the live
    # model contract, rather than another hardcoded dataset shape, is binding.
    return dict(map_size=8, history=2, future_frames=3, future_stride=2, fps=20,
                region_names=list(REGION_NAMES), sensor_names=["unit", "sharpa_real"],
                codec="tactile_intensity_v1")


def sample_for(mode, spec):
    sample = dict(training_modality=mode, video=[Image.new("RGB", (32, 48))] * 3,
                  video_mask=torch.ones(3, dtype=torch.bool), prompt="Pick up the object.")
    if mode in preflight.MODALITIES["robot"]:
        sample.update(action=torch.zeros(32, 80), action_mask=torch.zeros(32, 80, dtype=torch.bool),
                      proprio=torch.zeros(1, 80), proprio_mask=torch.zeros(1, 80, dtype=torch.bool))
        sample["action_mask"][:, :7] = True
        sample["proprio_mask"][:, :7] = True
    if mode in ("video_tactile", "video_action_tactile"):
        h, f, size = spec["history"], spec["future_frames"], spec["map_size"]
        sample.update(
            tactile_region_names=["right.gripper.jaw_a.surface"], tactile_sensor_names=["unit"],
            tactile_codec="tactile_intensity_v1", tactile_layout_present=torch.ones(1, dtype=torch.bool),
            tactile_maps=torch.zeros(h, 1, 1, size, size), tactile_valid=torch.ones(h, 1, dtype=torch.bool),
            future_tactile_maps=torch.zeros(f, 1, 1, size, size),
            future_tactile_valid=torch.ones(f, 1, dtype=torch.bool),
            tactile_spatial_valid=torch.ones(1, 1, size, size, dtype=torch.bool),
            tactile_query_mask=torch.ones(f, 1, dtype=torch.bool),
            tactile_history_times_s=torch.arange(1 - h, 1) / spec["fps"],
            future_tactile_times_s=torch.arange(1, f + 1) * spec["future_stride"] / spec["fps"],
        )
    return sample


@pytest.mark.parametrize("kind,mode", [(kind, mode) for kind, modes in preflight.MODALITIES.items() for mode in sorted(modes)])
def test_all_four_modes_follow_source_and_live_model_contract(tactile_spec, kind, mode):
    summary = preflight.validate_sample(sample_for(mode, tactile_spec), mode,
                                       tactile_spec=tactile_spec, source_kind=kind, expected_modalities=[mode])
    assert summary["modality"] == mode
    if "tactile" in mode:
        assert summary["collator_validated"]
        assert summary["history_shape"] == [1, 2, 1, 1, 8, 8]
        assert summary["future_shape"] == [1, 3, 1, 1, 8, 8]


@pytest.mark.parametrize("key,value,match", [
    ("tactile_sensor_names", ["not_registered"], "sensor names"),
    ("tactile_maps", torch.zeros(4, 1, 1, 8, 8), "tactile_maps: expected"),
    ("future_tactile_maps", torch.zeros(8, 1, 1, 8, 8), "future_tactile_maps: expected"),
    ("tactile_query_mask", torch.ones(2, 1, dtype=torch.bool), "tactile_query_mask: expected"),
    ("tactile_query_mask", torch.ones(3, 1), "requires torch.bool"),
    ("tactile_spatial_valid", torch.ones(1, 8, 8, dtype=torch.bool), "tactile_spatial_valid: expected"),
    ("tactile_spatial_valid", torch.zeros(1, 1, 8, 8, dtype=torch.bool), "nonempty spatial ROI"),
    ("tactile_layout_present", torch.zeros(1, dtype=torch.bool), "present region"),
    ("tactile_history_times_s", torch.tensor([-.1, 0]), "history_times_s differ"),
    ("future_tactile_times_s", torch.tensor([.1, .2, .4]), "future_tactile_times_s differ"),
])
def test_real_collation_rejects_registry_shape_time_and_mask_errors(tactile_spec, key, value, match):
    sample = sample_for("video_tactile", tactile_spec)
    sample[key] = value
    with pytest.raises(ValueError, match=match):
        preflight.validate_sample(sample, "video_tactile", tactile_spec=tactile_spec)


def test_only_measured_roi_pixels_are_range_checked(tactile_spec):
    sample = sample_for("video_tactile", tactile_spec)
    sample["tactile_spatial_valid"][..., 0, 0] = False
    sample["tactile_maps"][..., 0, 0] = float("nan")
    sample["future_tactile_maps"][..., 0, 0] = 500
    assert preflight.validate_sample(sample, "video_tactile", tactile_spec=tactile_spec)["future_labels"] == 3
    sample["future_tactile_maps"][..., 1, 1] = 1.01
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        preflight.validate_sample(sample, "video_tactile", tactile_spec=tactile_spec)


def test_query_is_independent_of_future_labels_and_only_effective_labels_count(tactile_spec):
    sample = sample_for("video_tactile", tactile_spec)
    sample["future_tactile_valid"].zero_()
    summary = preflight.validate_sample(sample, "video_tactile", tactile_spec=tactile_spec)
    assert summary["supervised_future_regions"] == 0  # video still supervises the sample
    sample["video_mask"][1:] = False
    with pytest.raises(ValueError, match="no future"):
        preflight.validate_sample(sample, "video_tactile", tactile_spec=tactile_spec)
    sample["future_tactile_valid"].fill_(True)
    sample["tactile_query_mask"].zero_()
    with pytest.raises(ValueError, match="no future"):
        preflight.validate_sample(sample, "video_tactile", tactile_spec=tactile_spec)


@pytest.mark.parametrize("key,values,match", [
    ("tactile_history_source_times_s", torch.tensor([0.0, 0.0]), "must not borrow future"),
    ("future_tactile_source_times_s", torch.tensor([0.0, .2, .3]), "strictly future"),
    ("future_tactile_source_times_s", torch.tensor([float("nan"), .2, .3]), "Invalid measured"),
])
def test_measured_timestamps_cannot_leak_future_or_relabel_history(tactile_spec, key, values, match):
    sample = sample_for("video_tactile", tactile_spec)
    sample[key] = values
    with pytest.raises(ValueError, match=match):
        preflight.validate_sample(sample, "video_tactile", tactile_spec=tactile_spec)


@pytest.mark.parametrize("mode", ["video_only", "video_tactile"])
def test_robot_source_cannot_fall_back_to_ego_mode(tactile_spec, mode):
    with pytest.raises(ValueError, match="robot source unexpectedly"):
        preflight.validate_sample(sample_for(mode, tactile_spec), mode, source_kind="robot")


@pytest.mark.parametrize("key", ["action", "action_mask", "proprio", "proprio_mask"])
def test_robot_source_requires_real_action_and_proprio_fields(tactile_spec, key):
    sample = sample_for("video_action", tactile_spec)
    del sample[key]
    with pytest.raises(ValueError, match="require measured"):
        preflight.validate_sample(sample, "video_action", source_kind="robot")


@pytest.mark.parametrize("key", ["action_mask", "proprio_mask"])
def test_all_masked_robot_placeholders_are_rejected(tactile_spec, key):
    sample = sample_for("video_action", tactile_spec)
    sample[key].zero_()
    with pytest.raises(ValueError, match="zero placeholders"):
        preflight.validate_sample(sample, "video_action", source_kind="robot")


def test_ftp_identity_cannot_be_declared_ego():
    source = {"source_kind": "ego", "source_family": "ftp1", "expected_modalities": ["video_tactile"]}
    with pytest.raises(ValueError, match="FTP-1 is a robot"):
        preflight.validate_source_contract("renamed_source", source)
    source.update(source_kind="robot", expected_modalities=["video_action_tactile"], type="lerobot_video_intensity")
    with pytest.raises(ValueError, match="action reader"):
        preflight.validate_source_contract("renamed_source", source)


@pytest.mark.parametrize("keep", [0, -2, 1.5, True])
def test_invalid_checkpoint_retention_is_rejected(keep):
    with pytest.raises(ValueError, match="keep_last_k_ckpts"):
        preflight.validate_training_schedule({"num_epochs": 1, "keep_last_k_ckpts": keep})


def test_keep_all_and_deferred_save_stop_are_valid():
    summary = preflight.validate_training_schedule(dict(
        num_epochs=1, keep_last_k_ckpts=-1, gradient_accumulation_steps=4, save_steps=10, max_steps=17))
    assert summary["keep_last_k_ckpts"] == -1
    assert summary["deferred_boundary_settings"] == ["save_steps", "max_steps"]


class FakeDataset:
    def __init__(self, sample):
        self.sample = sample
        self.training_modality = sample["training_modality"]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.sample


def fake_config(spec):
    return OmegaConf.create(dict(
        training=dict(num_epochs=1, action_backbone_init="pretrained", resume_ckpt_path="/already_verified_resume"),
        model=dict(architecture=dict(variant="touchwam_regions", training_mode="joint_finetune", tactile=spec)),
        dataloader=dict(type="mixture", datasets=dict(ftp_unit=dict(
            type="lerobot_robot_intensity", source_kind="robot", source_family="ftp1",
            expected_modalities=["video_action_tactile"], dataset_dir="/ftp1/Unit"))),
        project=dict(seed=42),
    ))


def stub_run(monkeypatch, cfg, dataset):
    monkeypatch.setattr(preflight, "initialize_config_dir", lambda **kwargs: nullcontext())
    monkeypatch.setattr(preflight, "compose", lambda **kwargs: cfg)
    monkeypatch.setattr(preflight, "source_publication_contract", lambda source: {"status": "complete"})
    monkeypatch.setattr(preflight, "build_dataset", lambda *args, **kwargs: dataset)


def test_full_preflight_report_blocks_unregistered_sensor(tmp_path, monkeypatch, tactile_spec):
    sample = sample_for("video_action_tactile", tactile_spec)
    sample["tactile_sensor_names"] = ["unknown"]
    stub_run(monkeypatch, fake_config(tactile_spec), FakeDataset(sample))
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="One or more selected"):
        preflight.run("unused", [], path)
    report = json.loads(path.read_text())
    assert report["status"] == "failed" and not report["training_launch_allowed"]
    assert "sensor names" in report["sources"]["ftp_unit"]["error"]


def test_every_bucket_mode_is_checked_before_decoding_selected_buckets(tmp_path, monkeypatch, tactile_spec):
    dataset = FakeDataset(sample_for("video_action_tactile", tactile_spec))
    dataset.buckets = [dataset, FakeDataset(sample_for("video_only", tactile_spec)), dataset, dataset, dataset]
    stub_run(monkeypatch, fake_config(tactile_spec), dataset)
    path = tmp_path / "report.json"
    with pytest.raises(ValueError, match="One or more selected"):
        preflight.run("unused", [], path)
    report = json.loads(path.read_text())
    assert "Bucket emits video_only" in report["sources"]["ftp_unit"]["error"]


def test_metadata_only_checks_source_identity_without_allowing_launch(tmp_path, monkeypatch, tactile_spec):
    cfg = fake_config(tactile_spec)
    stub_run(monkeypatch, cfg, None)
    report = preflight.run("unused", [], tmp_path / "report.json", metadata_only=True)
    assert report["status"] == "metadata_validated" and not report["training_launch_allowed"]
    cfg.dataloader.datasets.ftp_unit.expected_modalities = ["video_tactile"]
    with pytest.raises(ValueError, match="One or more selected"):
        preflight.run("unused", [], tmp_path / "invalid.json", metadata_only=True)
