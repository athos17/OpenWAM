"""Selective initialization keeps Wan/proprio and the entire constructor ActionDiT."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir


class TinyArchitecture(torch.nn.Module):
    def __init__(self, meta=False):
        super().__init__()
        self.video_backbone = torch.nn.Linear(8, 8, device="meta" if meta else "cpu")
        self.proprio_encoder = torch.nn.Linear(8, 8)
        self.action_backbone = ActionDiT(
            action_dim=8, dim=16, ffn_dim=32, num_heads=2, num_layers=2,
            video_dim=16, bridge_layers=[0, 1], variant="joint_self_attn",
            attn_head_dim=8, text_dim=8, shift_action=5.0,
        )
        self.driver_action_reference = SimpleNamespace(module=self.action_backbone)


@pytest.fixture
def source(tmp_path, monkeypatch):
    cfg = OmegaConf.create({"model": {"video_backbone": {}}})
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    original = TinyArchitecture().state_dict()
    original = {k: torch.full_like(v, 0.123) for k, v in original.items()}
    weights = tmp_path / "checkpoint_step_1.safetensors"
    save_file(original, weights)
    monkeypatch.setattr("openwam.model.resolve_architecture_config", lambda _: SimpleNamespace(registry_name="tiny", params={}))
    monkeypatch.setattr("openwam.model.build_architecture", lambda _, params: TinyArchitecture(meta=True))
    return weights, original


def build(source, seed=42, **kwargs):
    return build_architecture_from_ckpt_dir(
        str(source), weights_required=True, reset_action_backbone=True,
        action_init_seed=seed, **kwargs,
    )[1]


def test_all_action_tensors_remain_constructor_values(source):
    weights, original = source
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        expected = TinyArchitecture(meta=True).action_backbone.state_dict()
    torch.manual_seed(974)
    rng_before = torch.get_rng_state().clone()
    model = build(weights)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.driver_action_reference.module is model.action_backbone
    assert any("modulation" in name for name in expected)
    for name, value in model.state_dict().items():
        assert not value.is_meta
        if name.startswith("action_backbone."):
            assert torch.equal(value, expected[name.removeprefix("action_backbone.")]), name
        else:
            assert torch.equal(value, original[name]), name
    assert len(model.initialization_provenance["random_tensor_names"]) == len(expected)


def test_action_seed_is_rank_independent_and_configurable(source):
    weights, _ = source
    torch.manual_seed(100)
    first = build(weights).action_backbone.state_dict()
    torch.manual_seed(200)
    second = build(weights).action_backbone.state_dict()
    different = build(weights, seed=43).action_backbone.state_dict()
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not torch.equal(first["action_encoder.weight"], different["action_encoder.weight"])
    for k in first:
        if "modulation" in k:
            assert not torch.equal(first[k], different[k]), k


@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape"])
def test_retained_weights_are_strict(source, defect):
    weights, original = source
    if defect == "missing":
        del original["proprio_encoder.weight"]
    elif defect == "unexpected":
        original["surprise.weight"] = torch.ones(1)
    else:
        original["video_backbone.weight"] = torch.ones(1)
    save_file(original, weights)
    with pytest.raises(RuntimeError):
        build(weights)


@pytest.mark.parametrize("seed", [None, -1, True, 1.5, 2**63])
def test_invalid_seed_rejected(source, seed):
    with pytest.raises(ValueError, match="action_init_seed"):
        build(source[0], seed=seed)


def test_resume_never_allows_reset(source):
    with pytest.raises(ValueError, match="never resume"):
        build_architecture_from_ckpt_dir(str(source[0]), weights_required=False,
                                         reset_action_backbone=True, action_init_seed=42)
