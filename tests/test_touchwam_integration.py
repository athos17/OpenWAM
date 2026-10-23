"""TouchWAM integration with the current trainer, loader and server contracts."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.server import PolicyServer
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
from tests.test_touchwam import inputs, loss_inputs, tiny_config
from tests.test_touchwam import pair as touchwam_pair


@pytest.fixture
def pair(monkeypatch, tmp_path):
    return touchwam_pair.__wrapped__(monkeypatch, tmp_path)


def test_alpha_loader_preserves_base_weights_and_initializes_touch_identically_across_ranks(pair, monkeypatch):
    base, _, weights = pair
    source = OmegaConf.create(
        {
            "model": {
                "architecture": {"framework": "dual_system", "variant": "joint_self_attn"},
                "video_backbone": {},
            }
        }
    )
    OmegaConf.save(source, weights.parent / "config.yaml")
    live = OmegaConf.create(
        {
            "model": {
                "architecture": {
                    "framework": "dual_system",
                    "variant": "touchwam_v2",
                    "training_mode": "joint_finetune",
                },
                "video_backbone": {},
            },
            "training": {"initialize_touch_from_openwam": True},
            "project": {"seed": 42},
        }
    )
    monkeypatch.setattr(
        "openwam.model.resolve_architecture_config",
        lambda cfg: SimpleNamespace(
            registry_name="touchwam_v2",
            params=dict(tiny_config(), training_mode=cfg.architecture.training_mode),
        ),
    )

    def build(rank_seed):
        torch.manual_seed(rank_seed)
        before = torch.get_rng_state().clone()
        model = build_architecture_from_ckpt_dir(str(weights), weights_required=True, override_cfg=live)[1]
        assert torch.equal(torch.get_rng_state(), before)
        return model

    first, second = build(100), build(200)
    assert first.training_mode == "joint_finetune"
    for name, value in base.state_dict().items():
        torch.testing.assert_close(first.state_dict()[name], value, rtol=0, atol=0)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(second.state_dict()[name], value, rtol=0, atol=0)
    live.project.seed = 43
    third = build(300)
    assert not torch.equal(first.touch_v2.input_projection[0].weight, third.touch_v2.input_projection[0].weight)
    with pytest.raises(ValueError, match="Alpha -> TouchWAM"):
        build_architecture_from_ckpt_dir(str(weights), weights_required=False, override_cfg=live)
    with pytest.raises(ValueError, match="Alpha -> TouchWAM"):
        build_architecture_from_ckpt_dir(
            str(weights), weights_required=True, override_cfg=live, reset_action_backbone=True, action_init_seed=42
        )


def test_checkpoint_loader_keeps_live_training_policy_when_continuing_v2(pair, monkeypatch):
    _, model, alpha = pair
    weights = alpha.with_name("checkpoint_step_1.safetensors")
    model.save_checkpoint(str(weights))
    source = OmegaConf.create(
        {
            "model": {
                "architecture": {"framework": "dual_system", "variant": "touchwam_v2", "training_mode": "new_only"},
                "video_backbone": {},
            }
        }
    )
    OmegaConf.save(source, weights.parent / "config.yaml")
    live = OmegaConf.create(OmegaConf.to_container(source))
    live.model.architecture.training_mode = "joint_finetune"
    monkeypatch.setattr(
        "openwam.model.resolve_architecture_config",
        lambda cfg: SimpleNamespace(
            registry_name="touchwam_v2",
            params=dict(tiny_config(), training_mode=cfg.architecture.training_mode),
        ),
    )
    loaded = build_architecture_from_ckpt_dir(str(weights), weights_required=True, override_cfg=live)[1]
    assert loaded.action_backbone.action_encoder.weight.requires_grad
    assert loaded.video_backbone.dit.weight.requires_grad
    assert not next(loaded.video_backbone.vae.parameters()).requires_grad


def test_text_padding_from_lengths_matches_existing_alpha_path(pair):
    base, touch, _ = pair
    args = inputs()
    del args["context_mask"]
    args["seq_lens"] = torch.tensor([2, 3])
    expected = base(**{key: value for key, value in args.items() if not key.startswith(("tactile", "noisy_tactile"))})
    actual = touch(**args)
    for a, b in zip(actual[:2], expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_video_only_samples_fail_before_preprocessing_or_loss(pair):
    _, touch, _ = pair
    assert touch.supports_video_only_training is False
    with pytest.raises(ValueError, match="video-action"):
        touch.prepare_inputs([{"training_modality": "video_only"}])
    with pytest.raises(ValueError, match="video-action"):
        touch.compute_loss(**loss_inputs(), video_only_mask=torch.tensor([True, False]))


def test_engine_forwards_observed_touch_and_server_advertises_contract(pair):
    from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler

    _, touch, _ = pair
    touch.video_backbone._scheduler = FlowMatchScheduler("Wan")
    cfg = OmegaConf.create(
        {
            "inference": {"denoise_mode": "sync", "denoise_steps": 2, "num_frames": 5, "video_num_frames": 3},
            "optimization": {"decode_video": False, "dit_cache": {"enabled": False}, "compile": {"enabled": False}},
        }
    )
    engine = JointInferenceEngine(cfg, architecture=touch)
    args = inputs(1)
    touch.normalize_deploy_proprio = lambda value: value
    result = engine.generate(
        {
            "prompt": "pick",
            "observation": {
                "state": args["proprio"],
                "tactile_maps": args["tactile_maps"],
                "tactile_valid": args["tactile_valid"],
            },
        }
    )
    assert result["future_tactile_valid"].all()
    server = PolicyServer(engine, cfg)
    assert server._ckpt_contract()["tactile"] == touch.tactile_input_contract
    engine.architecture = SimpleNamespace()
    assert server._ckpt_contract() == {}


def test_online_history_retains_no_contact_and_resets_on_gaps():
    import numpy as np

    from openwam.tactile.sharpa import SITE_NAMES, SharpaTactileHistory

    history = SharpaTactileHistory(history=4, size=8)
    distance = {name: np.zeros((16, 16), np.float32) for name in SITE_NAMES}
    contact = {name: np.zeros((16, 16), bool) for name in SITE_NAMES}
    for t in range(4):
        result = history.append(t / 20, distance, contact)
    assert result["tactile_valid"].all() and not result["tactile_maps"].any()
    result = history.append(1.0, distance, contact)
    assert not result["tactile_valid"][:3].any() and result["tactile_valid"][3].all()
    history.reset()
    result = history.append(0.0, {}, {})
    assert not result["tactile_valid"].any()
