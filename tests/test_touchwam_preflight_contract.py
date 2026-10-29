"""Preflight must reject incompatible data before constructing the large model."""

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch

from scripts import preflight_touchwam as preflight
from openwam.tactile.registry import REGION_NAMES


def test_six_source_recipe_preserves_ftp_action_only_mode_after_yaml_composition():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        config = compose(config_name="train_touchwam_six_source_20261029")
    sources = OmegaConf.to_container(config.dataloader.datasets, resolve=True)
    enabled = {name: source for name, source in sources.items() if source.get("enabled", True)}
    assert len(enabled) == 14
    assert enabled["robocoin"]["weight"] > 0
    action_only_ftp = {name: source for name, source in enabled.items()
                       if name.startswith("ftp_") and source["expected_modalities"] == ["video_action"]}
    assert len(action_only_ftp) == 4
    for name, source in action_only_ftp.items():
        # YAML 1.1 silently parses an unquoted `off` as False. Check the actual
        # inherited Hydra configuration that reaches the production reader.
        assert source["tactile_mode"] == "off", name
        assert source["source_kind"] == "robot", name
        preflight.validate_source_contract(name, source)


def test_preflight_normalization_rejects_missing_robot_statistics_but_accepts_ego():
    robot = SimpleNamespace(training_modality="video_action", normalization_stats_path=None)
    with pytest.raises(ValueError, match="has no statistics artifact"):
        preflight.validate_source_normalization("ftp", robot, {"normalize_mode": "min-max", "action_mode": "eef"})
    ego = SimpleNamespace(training_modality="video_tactile", normalization_stats_path=None)
    result = preflight.validate_source_normalization("touchscale", ego, {})
    assert result["touchscale"]["stats_sha256"] is None


def test_preflight_normalization_matches_checkpoint_per_robot_expansion(tmp_path):
    import numpy as np
    from openwam.dataloader.utils.normalization_package import save_normalization_package, read_normalization_package

    entries = {name: {"config": {"normalize_mode": "min-max", "action_mode": "joint"},
                     "training_modality": "video_action",
                     "stats_payload": {"joint": {"action_min": np.zeros(size), "action_max": np.ones(size)}}}
               for name, size in (("arm", 7), ("bimanual", 14))}
    reader = SimpleNamespace(normalization_sources_for_config=lambda config: entries)
    result = preflight.validate_source_normalization("robot", reader, {})
    save_normalization_package(tmp_path, {"robot/" + key: value for key, value in entries.items()})
    saved = read_normalization_package(tmp_path)["sources"]
    assert set(result) == {"robot/arm", "robot/bimanual"}
    assert all(result[key]["stats_sha256"] == saved[key]["stats_sha256"] for key in result)


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


def rgb_sample(spec):
    spec.update(rgb_enabled=True, rgb_map_size=16, rgb_patch_size=8)
    sample = sample_for("video_action_tactile", spec)
    sample["tactile_codec"] = "tactile_rgb_v1"
    sample["tactile_maps"] = torch.full((2, 1, 3, 16, 16), .25)
    sample["future_tactile_maps"] = torch.full((3, 1, 3, 16, 16), .75)
    sample["tactile_spatial_valid"] = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    return sample


def test_rgb_preflight_uses_live_rgb_size_and_one_channel_roi(tactile_spec):
    sample = rgb_sample(tactile_spec)
    sample["tactile_spatial_valid"][..., 0, 0] = False
    sample["tactile_maps"][..., 0, 0] = float("nan")
    report = preflight.validate_sample(sample, "video_action_tactile", tactile_spec=tactile_spec)
    assert report["history_shape"] == [1, 2, 1, 3, 16, 16]
    assert report["tactile_representation"] == "optical_rgb"
    assert report["tactile_codec"] == "tactile_rgb_v1"
    sample["future_tactile_maps"][0, 0, 2, 1, 1] = 2
    with pytest.raises(ValueError, match="outside"):
        preflight.validate_sample(sample, "video_action_tactile", tactile_spec=tactile_spec)


def test_rgb_preflight_rejects_disabled_adapter_or_wrong_source_representation(tactile_spec):
    sample = rgb_sample(tactile_spec)
    tactile_spec["rgb_enabled"] = False
    with pytest.raises(ValueError, match="rgb_enabled"):
        preflight.validate_sample(sample, "video_action_tactile", tactile_spec=tactile_spec)
    tactile_spec["rgb_enabled"] = True
    with pytest.raises(ValueError, match="selected source representation"):
        preflight.validate_sample(sample, "video_action_tactile", tactile_spec=tactile_spec,
                                  expected_tactile_codec="tactile_intensity_v1")


def test_raw_optical_source_config_does_not_reclassify_array_or_wrench_sources():
    from pathlib import Path
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/dataloader/touchwam_pretrain.yaml")
    for name in ("ftp_unit", "ftp_freetacman", "ftp_rdp", "ftp_exumi", "ftp_visuotactile_d_wheel"):
        assert cfg.datasets[name].tactile_representation == "rgb"
        assert cfg.datasets[name].expected_modalities == ["video_action_tactile"]
    assert cfg.datasets.ftp_rh20tcfg7tactile.tactile_representation == "intensity"
    assert cfg.datasets.ftp_rh20tcfg7tactile.tactile_mode == "required"
    for name in ("ftp_motiontrans", "ftp_rh20tcfg5franka", "ftp_rh20tcfg6atiaxia"):
        assert cfg.datasets[name].enabled is False
