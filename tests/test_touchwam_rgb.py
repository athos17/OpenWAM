"""Original optical RGB uses independent adapters and the shared tactile flow."""

from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from openwam.model import build_architecture
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.touchwam.regional_modules import RGBPatchCodec, regional_flow_loss
from openwam.tactile.regions import CODEC, RGB_CODEC, REGION_NAMES
from tests.test_touchwam import VideoBackbone, tiny_config, open_gates
from tests.test_touchwam_regions import regional_loss_inputs


def rgb_config():
    cfg = tiny_config()
    cfg["training_mode"] = "joint_finetune"
    cfg["tactile"].update(rgb_enabled=True, rgb_map_size=16, rgb_patch_size=8)
    return cfg


def rgb_inputs(modality="video_action_tactile"):
    data = regional_loss_inputs(2)
    data.update(tactile_codec=RGB_CODEC, tactile_maps=torch.rand(2, 2, 2, 3, 16, 16),
                future_tactile_maps=torch.rand(2, 2, 2, 3, 16, 16),
                tactile_spatial_valid=torch.ones(2, 2, 1, 16, 16, dtype=torch.bool))
    if modality == "video_tactile":
        data.pop("actions")
        data.pop("proprio")
        data["video_only_mask"] = torch.ones(2, dtype=torch.bool)
    return data


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", lambda self, cfg: setattr(
        self, "video_backbone", VideoBackbone(dim=32, num_layers=3, num_heads=4)))
    model = build_architecture("touchwam_regions", rgb_config())
    model.set_dtype_device(torch.float32, torch.device("cpu"))
    model.init_training_schedulers()
    return model


def test_rgb_codec_roundtrip_preserves_channels_and_expands_roi():
    codec = RGBPatchCodec(128, 64)
    raw = torch.randint(0, 256, (1, 2, 1, 3, 128, 128), dtype=torch.uint8)
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    spatial = torch.ones(1, 1, 1, 128, 128, dtype=torch.bool)
    tokens, enabled = codec.encode(raw, valid, spatial)
    assert tokens.shape == (1, 8, 3 * 64**2) and enabled.all()
    torch.testing.assert_close(codec.decode(tokens, 1), raw.float() / 255)
    spatial[..., :64, :64] = False
    pixels = codec.pixel_mask(valid, spatial)
    assert pixels.shape == tokens.shape and pixels.sum() == 2 * 3 * (128**2 - 64**2)
    floats = raw.float() / 255
    floats[..., :64, :64] = float("nan")
    encoded, enabled = codec.encode(floats, valid, spatial)
    assert enabled.all() and torch.isfinite(encoded).all()


def test_rgb_flow_loss_does_not_use_brightness_as_force_weight():
    codec = RGBPatchCodec(2, 1)
    target = torch.zeros(1, 4, 3)
    clean = torch.tensor([[[-1., 0., 1.]]]).expand_as(target)
    prediction = torch.tensor([[[1., 2., 3.]]]).expand_as(target)
    pixels = torch.ones_like(prediction, dtype=torch.bool)
    one = regional_flow_loss(codec, prediction, target, clean, pixels, 1, foreground_weight=1)
    large = regional_flow_loss(codec, prediction, target, clean, pixels, 1, foreground_weight=99)
    torch.testing.assert_close(one, torch.tensor(14 / 3))
    torch.testing.assert_close(one, large, rtol=0, atol=0)


@pytest.mark.parametrize("modality", ["video_tactile", "video_action_tactile"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_wan_rgb_backward_updates_rgb_and_shared_only(monkeypatch, modality, dtype):
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan21

    def init_video(self, cfg):
        dit = WanModel(dim=96, in_dim=4, out_dim=4, ffn_dim=192, text_dim=32, freq_dim=32,
                       eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
                       has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True)
        self.video_backbone = Wan21(SimpleNamespace(dit=dit, scheduler=ActionScheduler()))

    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", init_video)
    cfg = rgb_config()
    cfg.update(bridge_layers=[0, 1], attn_head_dim=24, video_dim=96)
    cfg["tactile"]["bridge_layers"] = [0, 1]
    model = build_architecture("touchwam_regions", cfg)
    model.set_dtype_device(dtype, torch.device("cpu"))
    model.init_training_schedulers()
    open_gates(model)
    data = rgb_inputs(modality)
    data["input_latents"] = torch.randn(2, 4, 3, 4, 4)
    data["first_frame_latents"] = data["input_latents"][:, :, :1].clone()
    data["fuse_vae_embedding_in_latents"] = True
    data = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in data.items()}
    losses = model.compute_loss(**data, use_gradient_checkpointing=True)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"]) and losses["loss_tactile_rgb"] > 0
    assert losses["loss_tactile_intensity"] == 0 and losses["loss_tactile_deform"] == 0
    for name in ("rgb_input_projection.0.weight", "rgb_output.1.weight", "rgb_spatial_projection.weight",
                 "history_encoder.layers.0.linear1.weight", "blocks.0.read_va.q.weight", "region_embedding.weight"):
        gradient = model.touch_v2.get_parameter(name).grad
        assert gradient is not None and gradient.abs().sum() > 0, name
    assert model.touch_v2.input_projection[0].weight.grad is None
    assert model.touch_v2.output[1].weight.grad is None
    assert (model.action_backbone.action_encoder.weight.grad is not None) == (modality == "video_action_tactile")


def test_rgb_roi_and_unlabeled_future_values_cannot_leak(model):
    data = rgb_inputs()
    data["tactile_spatial_valid"][..., :8, :8] = False
    data["tactile_maps"][..., :8, :8] = float("nan")
    data["future_tactile_maps"][..., :8, :8] = float("nan")
    data["future_tactile_valid"][:, 1, 1] = False
    data["future_tactile_maps"][:, 1, 1] = float("nan")
    torch.manual_seed(5)
    before = model.compute_loss(**data)
    data["tactile_maps"][..., :8, :8] = 1000
    data["future_tactile_maps"][..., :8, :8] = 1000
    data["future_tactile_maps"][:, 1, 1] = 1000
    torch.manual_seed(5)
    after = model.compute_loss(**data)
    torch.testing.assert_close(before["loss"], after["loss"], rtol=0, atol=0)


def test_prepare_inputs_selects_codec_and_rejects_mixed_representation(model, monkeypatch):
    data = rgb_inputs("video_tactile")
    sample = {k: v[0] for k, v in data.items() if k.startswith(("tactile_", "future_tactile_"))
              and k not in ("tactile_codec", "tactile_region_ids")}
    sample.update(tactile_codec=RGB_CODEC, tactile_region_names=list(REGION_NAMES[:2]),
                  tactile_sensor_names=["sharpa_real"] * 2, training_modality="video_tactile",
                  tactile_history_times_s=[-1 / 30, 0], future_tactile_times_s=[1 / 30, 2 / 30])
    base = {k: v for k, v in data.items() if not k.startswith(("tactile_", "future_tactile_"))}
    monkeypatch.setattr(BaseWAMArchitecture, "prepare_inputs", lambda self, batch: base.copy())
    prepared = model.prepare_inputs([sample, sample])
    assert prepared["tactile_codec"] == RGB_CODEC
    assert prepared["tactile_maps"].shape == (2, 2, 2, 3, 16, 16)
    assert torch.isfinite(model.compute_loss(**prepared)["loss"])
    intensity = dict(sample, tactile_codec=CODEC)
    with pytest.raises(ValueError, match="homogeneous tactile codecs"):
        model.prepare_inputs([sample, intensity])


@pytest.mark.parametrize("modality", ["video_tactile", "video_action_tactile"])
def test_rgb_intensity_generation_routing_is_local_and_cfg_supported(model, modality):
    model.eval()
    open_gates(model)
    data = rgb_inputs(modality)
    kwargs = {k: v[0] for k, v in data.items() if k.startswith("tactile_") and k != "tactile_codec"}
    kwargs.update(tactile_codec=RGB_CODEC, generation_modality=modality, decode_video=False)
    if modality == "video_action_tactile":
        kwargs.update(proprio=data["proprio"][0], action_num_frames=5)
    guard = patch.object(model.action_backbone, "prepare_state", side_effect=AssertionError("ActionDiT entered"))
    with guard if modality == "video_tactile" else nullcontext():
        rgb = [model.generate([(1000, 1000), (500, 500), (0, 0)], "hold", cfg_scale=2, cfg_merge=merge,
                              **kwargs) for merge in (False, True)]
    assert rgb[0]["tactile_codec"] == RGB_CODEC and rgb[0]["tactile_representation"] == "optical_rgb"
    assert rgb[0]["future_tactile_maps"].shape == (2, 2, 3, 16, 16)
    np.testing.assert_allclose(rgb[0]["future_tactile_maps"], rgb[1]["future_tactile_maps"], rtol=2e-5, atol=2e-6)
    assert model.touch_v2.codec.channels == 1
    intensity = regional_loss_inputs(2)
    gray_kwargs = {k: v[0] for k, v in intensity.items() if k.startswith("tactile_")}
    gray = model.generate([(1000, 1000), (0, 0)], "hold", generation_modality="video_tactile",
                          decode_video=False, **gray_kwargs)
    assert gray["future_tactile_maps"].shape == (2, 2, 1, 8, 8) and gray["tactile_codec"] == CODEC
    rgb_again = model.generate([(1000, 1000), (500, 500), (0, 0)], "hold", cfg_scale=2, **kwargs)
    np.testing.assert_array_equal(rgb[0]["future_tactile_maps"], rgb_again["future_tactile_maps"])
    absent = model.generate([(1000, 1000), (0, 0)], "hold", generation_modality="video_only", decode_video=False)
    assert absent["future_tactile_maps"].shape[1] == 0 and absent["actions"] is None


def test_rgb_enabled_model_still_trains_intensity_and_bypasses_touch_when_absent(model):
    open_gates(model)
    inputs = regional_loss_inputs(2)
    losses = model.compute_loss(**inputs)
    losses["loss"].backward()
    assert losses["loss_tactile_rgb"] == 0 and losses["loss_tactile_intensity"] > 0
    assert model.touch_v2.input_projection[0].weight.grad.abs().sum() > 0
    assert model.touch_v2.rgb_input_projection[0].weight.grad is None
    assert model.touch_v2.rgb_output[1].weight.grad is None
    model.zero_grad(set_to_none=True)
    inputs = {k: v for k, v in inputs.items() if not k.startswith(("tactile_", "future_tactile_"))}
    with patch.object(model.touch_v2.history_encoder, "forward", side_effect=AssertionError("touch entered")):
        losses = model.compute_loss(**inputs)
        losses["loss"].backward()
    assert losses["loss_tactile"] == 0
    assert all(p.grad is None for p in model.touch_v2.parameters())


def test_disabled_rgb_preserves_old_checkpoint_and_alpha_initializes_new_rgb(monkeypatch, tmp_path):
    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", lambda self, cfg: setattr(
        self, "video_backbone", VideoBackbone(dim=32, num_layers=3, num_heads=4)))
    legacy = build_architecture("touchwam_regions", tiny_config())
    disabled = tiny_config()
    disabled["tactile"].update(rgb_enabled=False, rgb_map_size=128, rgb_patch_size=64)
    disabled = build_architecture("touchwam_regions", disabled)
    assert legacy.checkpoint_metadata == disabled.checkpoint_metadata
    assert legacy.state_dict().keys() == disabled.state_dict().keys()
    path = tmp_path / "old.safetensors"
    legacy.save_checkpoint(str(path))
    disabled.load_checkpoint(str(path))
    base = build_architecture("dual_system_self_attn", tiny_config())
    alpha = tmp_path / "alpha.safetensors"
    base.save_checkpoint(str(alpha))
    rgb = build_architecture("touchwam_regions", rgb_config())
    rgb.load_openwam_checkpoint(str(alpha))
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, rgb.state_dict()[name], rtol=0, atol=0)
    assert hasattr(rgb.touch_v2, "rgb_output") and not hasattr(disabled.touch_v2, "rgb_output")
    bad = rgb_config()
    bad["tactile"]["rgb_patch_size"] = 4
    with pytest.raises(ValueError, match="equal patches"):
        build_architecture("touchwam_regions", bad)
