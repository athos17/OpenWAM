"""Spatial FTP observation routing, strict reuse and generative integration."""
import copy
import hashlib
import json
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from omegaconf import OmegaConf

from openwam.model import build_architecture
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.touchwam.heterogeneous import HeterogeneousObservationEncoder, normalize_encoder_spec
from openwam.tactile.regions import RGB_CODEC
from tests.test_touchwam import VideoBackbone, tiny_config, open_gates
from tests.test_touchwam_regions import regional_loss_inputs


def encoder_spec(frozen=False):
    norm = dict(mean=[.1, .2, .3], std=[.4, .5, .6], pre_scale=2, pre_bias=-1)
    return normalize_encoder_spec(dict(initialization="random", image_size=16, image_patch_size=4,
        image_width=32, image_front_layers=1, image_shared_layers=1, image_heads=4,
        matrix_width=16, matrix_fourier_dim=8, freeze_pretrained=frozen,
        sensors=[dict(name="Sharpa", kind="image", input_representation="deform"),
                 dict(name="Optical", kind="image", input_representation="rgb"),
                 dict(name="Pressure", kind="matrix", input_shape=[4, 6])],
        source_to_encoder={"sharpa_real":["Sharpa"], "bench2dex_sim":["Pressure"], "optical":["Optical"]},
        normalizations={"sharpa_real":{"Sharpa":norm}, "optical":{"Optical":norm}}),
        ["sharpa_real", "bench2dex_sim", "optical"])


def config(frozen=False):
    cfg = tiny_config()
    cfg["tactile"].update(rgb_enabled=True, rgb_map_size=8, rgb_patch_size=4,
                         sensor_names=["sharpa_real", "bench2dex_sim", "optical"],
                         observation_encoder=encoder_spec(frozen))
    return cfg


def test_hydra_encoder_spec_resolves_nested_values_without_mutating_input():
    cfg = OmegaConf.create(dict(shared_mean=.1, encoder=encoder_spec()))
    cfg.encoder.normalizations.sharpa_real.Sharpa.mean[0] = "${shared_mean}"
    cfg.encoder.source_to_encoder.sharpa_real = "Sharpa"
    before = OmegaConf.to_container(cfg, resolve=False)
    normalized = normalize_encoder_spec(cfg.encoder, ["sharpa_real", "bench2dex_sim", "optical"])
    assert json.loads(json.dumps(normalized)) == normalized
    assert normalized["normalizations"]["sharpa_real"]["Sharpa"]["mean"][0] == .1
    assert normalized["source_to_encoder"]["sharpa_real"] == ["Sharpa"]
    assert OmegaConf.to_container(cfg, resolve=False) == before


def test_hydra_heterogeneous_checkpoint_roundtrip(monkeypatch, tmp_path):
    from safetensors import safe_open

    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", lambda self, cfg: setattr(
        self, "video_backbone", VideoBackbone(dim=32, num_layers=3, num_heads=4)))
    cfg = OmegaConf.create(config())
    model = build_architecture("touchwam_regions", cfg)
    target = tmp_path / "heterogeneous.safetensors"
    model.save_checkpoint(str(target))
    with safe_open(str(target), framework="pt") as saved:
        assert json.loads(saved.metadata()["tactile_spec"]) == model.tactile_spec
    restored = build_architecture("touchwam_regions", cfg)
    restored.load_checkpoint(str(target))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", lambda self, cfg: setattr(
        self, "video_backbone", VideoBackbone(dim=32, num_layers=3, num_heads=4)))
    model = build_architecture("touchwam_regions", config())
    model.set_dtype_device(torch.float32, torch.device("cpu"))
    model.init_training_schedulers()
    open_gates(model)
    return model


def inputs(kind="deform"):
    data = regional_loss_inputs(2)
    encoder, source, channels, shape = {"deform":(0, 0, 1, (16, 16)),
        "rgb":(1, 2, 3, (16, 16)), "pressure":(2, 1, 1, (4, 6))}[kind]
    if kind == "rgb":
        data.update(tactile_codec=RGB_CODEC, tactile_maps=torch.rand(2, 2, 2, 3, 8, 8),
                    future_tactile_maps=torch.rand(2, 2, 2, 3, 8, 8))
    data.update(tactile_observation_maps=torch.rand(2, 2, 2, channels, *shape),
                tactile_observation_spatial_valid=torch.ones(2, 2, 1, *shape, dtype=torch.bool),
                tactile_encoder_ids=torch.full((2, 2), encoder, dtype=torch.long),
                tactile_sensor_ids=torch.full((2, 2), source, dtype=torch.long))
    return data


@pytest.mark.parametrize("kind", ["deform", "rgb", "pressure"])
def test_real_flow_backward_routes_observation_and_future_separately(model, kind):
    data = inputs(kind)
    result = model.compute_loss(**data)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    enc = model.touch_v2.observation_encoder
    if kind == "pressure":
        assert enc.matrix_encoders["Pressure"].cnn[0].weight.grad.abs().sum() > 0
        assert enc.image_projection.weight.grad is None
    else:
        active, inactive = ("Sharpa", "Optical") if kind == "deform" else ("Optical", "Sharpa")
        assert enc.image_frontends[active].patch_embed.proj.weight.grad.abs().sum() > 0
        assert enc.image_frontends[inactive].patch_embed.proj.weight.grad is None
        assert enc.image_projection.weight.grad.abs().sum() > 0
    output = model.touch_v2.rgb_output if kind == "rgb" else model.touch_v2.output
    assert output[1].weight.grad.abs().sum() > 0


def condition(model, data):
    return model._condition(data["tactile_maps"], data["tactile_valid"], 2,
        region_ids=data["tactile_region_ids"], layout_present=data["tactile_layout_present"],
        sensor_ids=data["tactile_sensor_ids"], observation_maps=data["tactile_observation_maps"],
        observation_spatial_valid=data["tactile_observation_spatial_valid"], encoder_ids=data["tactile_encoder_ids"])


def test_history_is_causal_and_prediction_cache_cannot_replace_observation(model):
    data = inputs()
    before = condition(model, data)
    data["tactile_observation_maps"][:, 1] = torch.rand_like(data["tactile_observation_maps"][:, 1])
    after = condition(model, data)
    torch.testing.assert_close(before[0][:, :8], after[0][:, :8], rtol=0, atol=0)
    assert not torch.equal(before[0][:, 8:], after[0][:, 8:])
    data["tactile_maps"].fill_(float("nan"))
    torch.testing.assert_close(condition(model, data)[0], after[0], rtol=0, atol=0)


def test_missing_packets_and_invalid_roi_have_no_information_path(model):
    data = inputs()
    data["tactile_valid"][:, :, 1] = False
    data["tactile_observation_spatial_valid"][:, 0, :, :8, :8] = False
    before = condition(model, data)
    data["tactile_observation_maps"][:, :, 1].fill_(float("nan"))
    data["tactile_observation_maps"][:, :, 0, :, :8, :8].fill_(float("nan"))
    after = condition(model, data)
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert not after[1].reshape(2, 2, 2, 4)[:, :, 1].any()
    assert not after[1].reshape(2, 2, 2, 4)[:, :, 0, 0].any()


def test_strict_observation_contract_and_source_routing(model):
    data = inputs()
    del data["tactile_observation_maps"]
    with pytest.raises(ValueError, match="requires observation"):
        model.compute_loss(**data)
    data = inputs()
    data["tactile_sensor_ids"].fill_(2)
    with pytest.raises(ValueError, match="Unregistered source/encoder"):
        model.compute_loss(**data)


def test_pretrained_freeze_survives_training_policy(model):
    model.touch_v2.observation_encoder.spec["freeze_pretrained"] = True
    model._apply_training_policy()
    enc = model.touch_v2.observation_encoder
    assert not any(p.requires_grad for p in enc.image_frontends.parameters())
    assert not any(p.requires_grad for p in enc.image_trunk.parameters())
    assert enc.image_projection.weight.requires_grad
    assert enc.matrix_encoders["Pressure"].projection.weight.requires_grad


def test_generation_caches_observation_encoder_across_denoising_and_cfg(model):
    data = inputs()
    args = {key:value[:1] for key, value in data.items() if key.startswith("tactile_")}
    with patch.object(model.touch_v2.observation_encoder, "forward", wraps=model.touch_v2.observation_encoder.forward) as spy:
        result = model.generate([(1000, 1000), (500, 500), (0, 0)], "touch", generation_modality="video_action_tactile",
            proprio=torch.zeros(1, 7), action_num_frames=3, decode_video=False,
            cfg_scale=2, cfg_merge=True, **args)
    assert spy.call_count == 1
    assert result["future_tactile_maps"].shape == (2, 2, 1, 8, 8)


def test_pinned_loading_rejects_missing_keys_shapes_and_wrong_sha(tmp_path):
    spec = encoder_spec()
    encoder = HeterogeneousObservationEncoder(spec, 16, ["sharpa_real", "bench2dex_sim", "optical"])
    files = [(name, mod, "vit_encoder.") for name, mod in encoder.image_frontends.items()]
    files += [("shared", encoder.image_trunk, "shared_chunk_encoder.")]
    weights = {}
    for name, module, prefix in files:
        state = {prefix+k:v for k,v in module.state_dict().items()}
        if name == "shared":
            state.update({"image_proj.weight":torch.zeros(512, 32), "image_proj.bias":torch.zeros(512)})
        path = tmp_path / f"{name}.safetensors"
        save_file(state, str(path))
        weights[name] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    spec.update(initialization="pretrained", weights_root=str(tmp_path), weights=weights)
    target = HeterogeneousObservationEncoder(spec, 16, encoder.source_names)
    assert not target.weights_initialized
    report = target.load_pretrained()
    assert target.weights_initialized and len(report["files"]) == 3
    for key in encoder.image_trunk.state_dict():
        torch.testing.assert_close(target.image_trunk.state_dict()[key], encoder.image_trunk.state_dict()[key], rtol=0, atol=0)
    spec["weights"]["Sharpa"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        target.load_pretrained()
    path = tmp_path / "Sharpa.safetensors"
    save_file({"vit_encoder.cls_token":torch.zeros(1, 1, 32)}, str(path))
    spec["weights"]["Sharpa"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="key mismatch"):
        target.load_pretrained()


def test_uninitialized_pretrained_image_never_falls_back_to_random(model):
    model.touch_v2.observation_encoder.weights_initialized.fill_(False)
    with pytest.raises(RuntimeError, match="not initialized"):
        model.compute_loss(**inputs())


def test_optimizer_activation_uses_history_after_dropout_and_nonfinite_masking(model):
    model.tactile_spec["history_dropout"] = 1.0
    result = model.compute_loss(**inputs())
    assert result["observation_encoder_active_ids"].numel() == 0
    model.tactile_spec["history_dropout"] = 0.0
    data = inputs()
    data["tactile_observation_maps"].fill_(float("nan"))
    result = model.compute_loss(**data)
    assert result["observation_encoder_active_ids"].numel() == 0


@pytest.mark.parametrize("kind", ["deform", "pressure"])
def test_padded_variable_region_batch_backward(model, kind):
    data = inputs(kind)
    data["tactile_region_ids"][1, 1] = -1
    data["tactile_encoder_ids"][1, 1] = -1
    data["tactile_layout_present"][1, 1] = False
    data["tactile_valid"][1, :, 1] = False
    data["future_tactile_valid"][1, :, 1] = False
    data["tactile_observation_spatial_valid"][1, 1] = False
    data["tactile_observation_maps"][1, :, 1] = float("nan")
    result = model.compute_loss(**data)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])


@pytest.mark.parametrize("kind", ["deform", "rgb", "pressure"])
def test_bfloat16_observation_spatial_pool_preserves_projection_dtype(model, kind):
    model.set_dtype_device(torch.bfloat16, torch.device("cpu"))
    # The synthetic video backbone's set_dtype_device hook intentionally does
    # not move its small Linear module (the real Wan hook does).
    model.to(dtype=torch.bfloat16)
    data = inputs(kind)
    # CPU oneDNN does not provide BF16 backwards on all supported x86 hosts.
    with torch.backends.mkldnn.flags(enabled=False):
        result = model.compute_loss(**data)
        result["loss"].backward()
    assert torch.isfinite(result["loss"])
