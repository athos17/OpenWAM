"""Native observation resolution, strict frontend identity, and inactive optimizer state."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from openwam.tactile.intensity_store import IntensityStore
from openwam.tactile.registry import REGION_NAMES
from openwam.tactile.regions import collate_regions, observation_batch_shape
from openwam.tactile.rgb_store import RGBTactileStore
from openwam.train.utils.modality_optimizer import (
    ModalityAwareAdamW, build_modality_groups, global_observation_encoder_ids,
)
from scripts.preflight_touchwam import validate_tactile_sample
from tests.test_touchwam_preflight_contract import sample_for
from tests.test_touchwam_rgb_reader import rgb_bucket, rgb_reader


def intensity_publication(root, hw=(24, 16)):
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text("{}")
    data = np.broadcast_to((np.indices(hw).sum(0) % 2).astype(np.float32), (5, 1, *hw)).copy()
    np.save(root / "maps.npy", data)
    np.save(root / "valid.npy", np.ones((5, 1), dtype=bool))
    np.save(root / "timestamps.npy", np.arange(5) / 10)
    manifest = dict(schema="tactile_intensity_v1", codec="tactile_intensity_v1", fps=10,
                    dataset_info_sha256=hashlib.sha256(b"{}").hexdigest(),
                    region_names=["right.thumb.tip"], sensor_names=["unit"],
                    episodes={"0": dict(maps="maps.npy", valid="valid.npy", timestamps="timestamps.npy",
                                        layout_present=[True])})
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, data


def test_intensity_observation_uses_native_matrix_not_resized_targets(tmp_path):
    path, original = intensity_publication(tmp_path)
    store = IntensityStore(path, tmp_path, size=2, observation_size=[24, 16], encoder_name="pressure",
                           history_times=[0], future_times=[.1])
    sample = store.window(0, .1)
    assert sample["tactile_maps"].shape == (1, 1, 1, 2, 2)
    assert sample["tactile_observation_maps"].shape == (1, 1, 1, 24, 16)
    torch.testing.assert_close(sample["tactile_observation_maps"][0, :, 0], torch.from_numpy(original[1]))
    torch.testing.assert_close(sample["tactile_maps"], torch.full((1, 1, 1, 2, 2), .5))
    assert sample["tactile_encoder_names"] == ["pressure"]
    assert sample["tactile_observation_spatial_valid"].all()


def test_low_resolution_sidecar_cannot_masquerade_as_native_deform(tmp_path):
    path, _ = intensity_publication(tmp_path, (32, 32))
    store = IntensityStore(path, tmp_path, observation_size=224, encoder_name="sharpa")
    with pytest.raises(ValueError, match="native-resolution.*not upsampled"):
        store.window(0, .1)


def test_rgb_observation_is_resized_from_decoded_native_frame(tmp_path, monkeypatch):
    root, _, _ = rgb_bucket(tmp_path / "rgb")
    pixels = np.broadcast_to(((np.indices((8, 12)).sum(0) % 2) * 255)[..., None], (8, 12, 3)).astype(np.uint8)
    from openwam.tactile import rgb_store

    monkeypatch.setattr(rgb_store, "decode_video_frames", lambda path, indices, *shape: [pixels] * len(indices))
    store = RGBTactileStore(root / "meta/tactile_rgb.json", root, size=2,
                            observation_size=[8, 12], encoder_name="gelsight",
                            history_times=[0], future_times=[.1])
    sample = store.window(1, .1)
    torch.testing.assert_close(sample["tactile_maps"], torch.full((1, 2, 3, 2, 2), .5))
    torch.testing.assert_close(sample["tactile_observation_maps"][0, 0], torch.from_numpy(pixels).permute(2, 0, 1) / 255)
    assert sample["tactile_encoder_names"] == ["gelsight", "gelsight"]


def test_robot_reader_passes_observation_config_and_preserves_contract(tmp_path):
    root, _, _ = rgb_bucket(tmp_path / "robot")
    reader = rgb_reader(root, tactile_observation_map_size=224, tactile_encoder_name="gelsight")
    sample = reader[0]
    assert sample["tactile_observation_maps"].shape == (4, 2, 3, 224, 224)
    assert sample["future_tactile_maps"].shape == (8, 2, 3, 128, 128)
    assert reader.training_data_contract["tactile_observation_map_size"] == [224, 224]
    assert reader.training_data_contract["tactile_encoder_name"] == "gelsight"
    assert sample["action_mask"].any()


def example_spec():
    return dict(map_size=8, history=2, future_frames=3, future_stride=2, fps=20,
                region_names=list(REGION_NAMES), sensor_names=["unit"],
                codec="tactile_intensity_v1",
                observation_encoder=dict(kind="ftp1_heterogeneous_v1", image_size=224,
                    sensors=[dict(name="pressure", kind="matrix", input_representation="pressure", input_shape=[24, 16])],
                    source_to_encoder={"unit": "pressure"}))


def matrix_sample(spec):
    sample = sample_for("video_tactile", spec)
    sample.update(tactile_observation_maps=torch.zeros(2, 1, 1, 24, 16),
                  tactile_observation_spatial_valid=torch.ones(1, 1, 24, 16, dtype=torch.bool),
                  tactile_encoder_names=["pressure"])
    return sample


def test_collator_requires_observations_and_distinct_encoder_identity():
    spec = example_spec()
    sample = matrix_sample(spec)
    kwargs = dict(history=2, future_frames=3, map_size=8, sensor_names=["unit"],
                  codec="tactile_intensity_v1", observation_map_size=[24, 16], encoder_names=["pressure"])
    batch = collate_regions([sample, {}], **kwargs)
    assert batch["tactile_encoder_ids"].tolist() == [[0], [-1]]
    assert batch["tactile_observation_maps"].shape == (2, 2, 1, 1, 24, 16)
    assert not batch["tactile_observation_spatial_valid"][1].any()
    for key in ("tactile_observation_maps", "tactile_observation_spatial_valid", "tactile_encoder_names"):
        missing = {k: v for k, v in sample.items() if k != key}
        with pytest.raises(ValueError, match="requires"):
            collate_regions([missing], **kwargs)
    empty = collate_regions([{}], **kwargs)
    assert empty["tactile_observation_maps"].shape == (1, 2, 0, 1, 24, 16)


def test_preflight_verifies_native_shape_source_mapping_and_observed_range():
    spec = example_spec()
    sample = matrix_sample(spec)
    assert observation_batch_shape([sample], spec["observation_encoder"]) == (24, 16)
    summary, _ = validate_tactile_sample(sample, spec)
    assert summary["observation_shape"] == [1, 2, 1, 1, 24, 16]
    sample["tactile_observation_maps"][0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="heterogeneous tactile values"):
        validate_tactile_sample(sample, spec)
    sample["tactile_observation_spatial_valid"][0, 0, 0, 0] = False
    assert validate_tactile_sample(sample, spec)[0]["collator_validated"]
    spec["observation_encoder"]["source_to_encoder"] = {}
    with pytest.raises(ValueError, match="not mapped"):
        validate_tactile_sample(sample, spec)


class OptimizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tactile_spec = dict(rgb_enabled=True, observation_encoder=dict(kind="ftp1_heterogeneous_v1", sensors=[
            dict(name="sharpa", kind="image"), dict(name="gelsight", kind="image"), dict(name="pressure", kind="matrix")]))
        self.video_backbone, self.action_backbone, self.proprio_encoder = [nn.Linear(2, 2) for _ in range(3)]
        self.touch_v2 = nn.Module()
        self.touch_v2.shared = nn.Linear(2, 2)
        self.touch_v2.input_projection = nn.Linear(2, 2)
        self.touch_v2.rgb_input_projection = nn.Linear(2, 2)
        observation = self.touch_v2.observation_encoder = nn.Module()
        observation.image_frontends = nn.ModuleDict({name: nn.Linear(2, 2) for name in ("sharpa", "gelsight")})
        observation.image_trunk, observation.image_projection = nn.Linear(2, 2), nn.Linear(2, 2)
        observation.matrix_encoders = nn.ModuleDict({"pressure": nn.Linear(2, 2)})

    def get_trainable_modules(self):
        return {name: getattr(self, name) for name in ("video_backbone", "action_backbone", "proprio_encoder", "touch_v2")}


def test_observation_frontends_and_trunk_activate_only_for_valid_history():
    model = OptimizerModel()
    opt = ModalityAwareAdamW(build_modality_groups(model), lr=.1, weight_decay=.1)
    opt.activate("video_action_tactile", "tactile_rgb_v1", observation_encoder_ids=[0, 1, 2])
    for group in opt.param_groups:
        for p in group["params"]:
            p.grad = torch.ones_like(p)
    opt.step()
    before = {id(p): (p.detach().clone(), copy.deepcopy(opt.state[p])) for g in opt.param_groups for p in g["params"]
              if p in opt.state}
    opt.activate("video_tactile", "tactile_rgb_v1", observation_encoder_ids=[1])
    for group in opt.param_groups:
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    opt.step()
    held = {"tactile_observation_image:sharpa", "tactile_observation_matrix:pressure"}
    for group in opt.param_groups:
        if group["touchwam_stream"] in held:
            for p in group["params"]:
                torch.testing.assert_close(p, before[id(p)][0], rtol=0, atol=0)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(opt.state[p][key], before[id(p)][1][key], rtol=0, atol=0)
    shared = next(g for g in opt.param_groups if g["touchwam_stream"] == "tactile_observation_image_shared")
    assert all(opt.state[p]["step"] == 2 for p in shared["params"])
    opt.activate("video_tactile", "tactile_rgb_v1", observation_encoder_ids=[])
    assert not any(g["touchwam_active_in_window"] for g in opt.param_groups if "touchwam_observation_encoder_ids" in g)


def test_optimizer_activation_unions_history_across_ranks_and_accumulation(monkeypatch):
    from openwam.train.utils import modality_optimizer as module

    inputs = dict(tactile_encoder_ids=torch.tensor([[0, 1, 2]]),
                  tactile_valid=torch.tensor([[[False, True, True]]]),
                  tactile_layout_present=torch.tensor([[True, True, True]]),
                  tactile_observation_spatial_valid=torch.tensor([[[[[True]]], [[[True]]], [[[False]]]]]))
    assert global_observation_encoder_ids(inputs, 3, "cpu") == [1]
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "all_reduce", lambda value, op: value.__setitem__(0, 1))
    assert global_observation_encoder_ids(inputs, 3, "cpu") == [0, 1]
    opt = ModalityAwareAdamW(build_modality_groups(OptimizerModel()), lr=.1)
    opt.activate("video_tactile", "tactile_intensity_v1", observation_encoder_ids=[0])
    opt.activate("video_tactile", "tactile_rgb_v1", observation_encoder_ids=[1])
    groups = {g["touchwam_stream"] for g in opt.param_groups if g["touchwam_active_in_window"]}
    assert "tactile_observation_image:sharpa" in groups and "tactile_observation_image:gelsight" in groups
    assert "tactile_observation_matrix:pressure" not in groups


def test_region_specific_rgb_encoder_names_are_not_overridden_by_source_config(tmp_path):
    root, manifest, _ = rgb_bucket(tmp_path / "mixed")
    manifest["regions"][0]["encoder_name"] = "gelsight"
    manifest["regions"][1]["encoder_name"] = "mctac"
    path = root / "meta/tactile_rgb.json"
    path.write_text(json.dumps(manifest))
    sample = RGBTactileStore(path, root, observation_size=224).window(1, .1)
    assert sample["tactile_encoder_names"] == ["gelsight", "mctac"]
    with pytest.raises(ValueError, match="conflicts"):
        RGBTactileStore(path, root, observation_size=224, encoder_name="gelsight")


def test_native_matrix_shapes_cannot_silently_mix_in_one_batch():
    spec = example_spec()
    obs = spec["observation_encoder"]
    obs["sensors"].append(dict(name="small", kind="matrix", input_representation="pressure", input_shape=[4, 4]))
    sample = matrix_sample(spec)
    other = dict(sample, tactile_encoder_names=["small"])
    with pytest.raises(ValueError, match="one observation spatial shape"):
        observation_batch_shape([sample, other], obs)


def test_observation_padding_has_no_encoder_activation():
    inputs = dict(tactile_encoder_ids=torch.tensor([[0, -1]]),
                  tactile_valid=torch.zeros(1, 2, 2, dtype=torch.bool),
                  tactile_layout_present=torch.tensor([[True, False]]),
                  tactile_observation_spatial_valid=torch.ones(1, 2, 1, 2, 2, dtype=torch.bool))
    assert global_observation_encoder_ids(inputs, 1, "cpu") == []
    inputs["tactile_valid"][0, 0, 0] = True
    assert global_observation_encoder_ids(inputs, 1, "cpu") == [0]
