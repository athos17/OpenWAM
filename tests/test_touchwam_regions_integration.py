import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.policy import WAMPolicy
from openwam.deploy.server import PolicyServer
from openwam.model import resolve_architecture_config
from openwam.tactile.registry import REGION_NAMES
from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
from tests.test_touchwam import open_gates, tiny_config
from tests.test_touchwam_regions import models as regional_models
from tests.test_touchwam_regions import regional_inputs, regional_loss_inputs


@pytest.fixture
def models(monkeypatch, tmp_path):
    return regional_models.__wrapped__(monkeypatch, tmp_path)


def test_new_hydra_training_configs_keep_legacy_geometry_separate():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        real = compose(config_name="train_touchwam_regions")
        sim = compose(
            config_name="train_touchwam_regions",
            overrides=[
                "dataloader=bench2dex_deform",
                "model.architecture.tactile.fps=20",
                "model.architecture.tactile.future_stride=2",
            ],
        )
    assert resolve_architecture_config(real.model).registry_name == "touchwam_regions"
    assert tuple(real.model.architecture.tactile.region_names) == REGION_NAMES
    assert real.dataloader.tactile_fps == 30 and real.dataloader.tactile_future_stride == 3
    assert sim.dataloader.type == "bench2dex_deform" and sim.dataloader.tactile_fps == 20


def test_legacy_migration_through_real_checkpoint_loader_is_reproducible(models, tmp_path, monkeypatch):
    _, legacy, _ = models
    weights = tmp_path / "legacy.safetensors"
    legacy.save_checkpoint(str(weights))
    source = OmegaConf.create(
        {
            "model": {
                "architecture": {"framework": "dual_system", "variant": "touchwam_v2", "tactile": legacy.tactile_spec},
                "video_backbone": {},
            }
        }
    )
    OmegaConf.save(source, tmp_path / "config.yaml")
    cfg = OmegaConf.create(
        {
            "model": {
                "architecture": {
                    "framework": "dual_system",
                    "variant": "touchwam_regions",
                    "tactile": tiny_config()["tactile"],
                },
                "video_backbone": {},
            },
            "training": {"initialize_touch_from_legacy": True},
            "project": {"seed": 42},
        }
    )
    monkeypatch.setattr(
        "openwam.model.resolve_architecture_config",
        lambda config: SimpleNamespace(
            registry_name="touchwam_regions", params=dict(tiny_config(), tactile=dict(config.architecture.tactile))
        ),
    )
    output = []
    for seed in [100, 200]:
        torch.manual_seed(seed)
        before = torch.get_rng_state().clone()
        loaded = build_architecture_from_ckpt_dir(str(weights), weights_required=True, override_cfg=copy.deepcopy(cfg))[
            1
        ]
        assert torch.equal(before, torch.get_rng_state())
        assert not loaded.initialization_provenance["optimizer_resumed"]
        output.append(loaded)
    for name, value in output[0].state_dict().items():
        torch.testing.assert_close(value, output[1].state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="finetune"):
        build_architecture_from_ckpt_dir(str(weights), weights_required=False, override_cfg=cfg)


def test_engine_forwards_layout_queries_and_palm_disable(models):
    from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler

    _, _, model = models
    model.video_backbone._scheduler = FlowMatchScheduler("Wan")
    cfg = OmegaConf.create(
        {
            "inference": {"denoise_mode": "sync", "denoise_steps": 2, "num_frames": 5, "video_num_frames": 3},
            "optimization": {"decode_video": False, "dit_cache": {"enabled": False}, "compile": {"enabled": False}},
        }
    )
    engine = JointInferenceEngine(cfg, architecture=model)
    args = regional_inputs(batch=1)
    observation = {
        key: value
        for key, value in args.items()
        if key in ("tactile_maps", "tactile_valid", "tactile_region_ids", "tactile_layout_present")
    }
    observation.update(state=args["proprio"], tactile_palm_mode="off")
    model.normalize_deploy_proprio = lambda x: x
    result = engine.generate(dict(prompt="pick", observation=observation))
    assert result["future_tactile_maps"].shape == (2, 10, 1, 8, 8)
    assert not any("palm" in name for name in result["tactile_region_names"])
    assert "tactile_surface_ids" not in result
    observation["tactile_surface_ids"] = torch.zeros(1, 12, dtype=torch.long)
    with pytest.raises(ValueError, match="removed"):
        engine.generate(dict(prompt="pick", observation=observation))


def test_sync_server_returns_maps_for_the_action_plan_without_resampling():
    class Engine:
        calls = 0
        architecture = SimpleNamespace()

        def generate(self, conditions):
            self.calls += 1
            return dict(
                actions=np.full((3, 7), self.calls),
                future_tactile_maps=np.ones((2, 1, 1, 8, 8)),
                tactile_region_names=["left.palm"],
                tactile_representation="deform_gray",
            )

    engine = Engine()
    policy = WAMPolicy(engine, {})
    server = object.__new__(PolicyServer)
    server._init_policy = lambda: None
    server._policy = policy
    server._obs_preprocessor = SimpleNamespace(preprocess=lambda x: x)
    server._request_count, server._total_latency = 0, 0
    first = server.predict({"include_tactile": True, "timestamp_s": 12.3})
    second = server.predict({"include_tactile": True, "timestamp_s": 12.4})
    assert engine.calls == 1
    assert first["tactile"]["prediction_id"] == second["tactile"]["prediction_id"]
    assert [first["tactile"]["action_index"], second["tactile"]["action_index"]] == [0, 1]
    assert second["tactile"]["observation_timestamp_s"] == 12.3
    json.dumps(second)  # Wire payload contains ordinary arrays/scalars only.
    server.reset()
    assert policy.get_tactile_prediction() is None
    assert "tactile" not in server.predict({})


def test_same_seed_keeps_finger_noise_when_palms_are_removed(models):
    _, _, model = models
    conditions = []
    for n in [10, 12]:
        args = regional_inputs(n, batch=1)
        condition = model._condition(
            args["tactile_maps"],
            args["tactile_valid"],
            1,
            region_ids=args["tactile_region_ids"],
            layout_present=args["tactile_layout_present"],
        )
        _, pixels = model._query(condition)
        noise = model.sample_tactile_noise(condition, pixels.any(-1), 19)
        conditions.append(noise.reshape(1, 2, n, 4, 16)[:, :, :10])
    torch.testing.assert_close(*conditions, rtol=0, atol=0)


def test_mixed_layout_preparation_reaches_loss_and_backward(models, monkeypatch):
    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.tactile.regions import CODEC, REGION_NAMES

    _, _, model = models
    open_gates(model)
    samples = []
    for n in [10, 12]:
        args = regional_loss_inputs(n)
        sample = {
            k: v[0]
            for k, v in args.items()
            if k.startswith(("tactile_", "future_tactile_")) and k != "tactile_region_ids"
        }
        sample.update(
            tactile_region_names=list(REGION_NAMES[:n]),
            tactile_codec=CODEC,
            tactile_fps=30,
            future_tactile_offsets=torch.tensor([1, 2]),
        )
        samples.append(sample)
    # Replace expensive visual preprocessing only; use real collation, model and loss.
    va = {k: v for k, v in regional_loss_inputs().items() if not k.startswith(("tactile_", "future_tactile_"))}
    monkeypatch.setattr(BaseWAMArchitecture, "prepare_inputs", lambda self, batch: va.copy())
    prepared = model.prepare_inputs(samples)
    assert prepared["tactile_region_ids"][0, 10:].tolist() == [-1, -1]
    assert not prepared["future_tactile_valid"][0, :, 10:].any()
    losses = model.compute_loss(**prepared, use_gradient_checkpointing=True)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.touch_v2.region_embedding.weight.grad[10:].abs().sum() > 0


def test_merged_cfg_matches_sequential_with_regional_metadata(models):
    _, _, model = models
    open_gates(model)
    args = regional_inputs(batch=1)
    kwargs = {
        k: v
        for k, v in args.items()
        if k in ("tactile_maps", "tactile_valid", "tactile_region_ids", "tactile_layout_present", "proprio")
    }
    result = [
        model.generate(
            schedule=[(1000, 1000), (500, 500), (0, 0)],
            prompt="pick",
            action_num_frames=5,
            decode_video=False,
            cfg_scale=2,
            cfg_merge=merge,
            tactile_palm_mode="off",
            **kwargs,
        )
        for merge in [False, True]
    ]
    for key in ["actions", "future_tactile_maps"]:
        np.testing.assert_allclose(result[0][key], result[1][key], rtol=2e-5, atol=2e-6)
