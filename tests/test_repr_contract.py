"""Representation contract must advertise checkpoint training truth."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.deploy.model_loader import repr_contract_from_cfg
from openwam.deploy.server import PolicyServer

TRAINING_DL = {
    "action_mode": "robocasa365",
}


def test_repr_contract_from_cfg_reads_compact_values():
    contract = repr_contract_from_cfg(OmegaConf.create({"dataloader": TRAINING_DL}))
    assert contract == {"representation": "robocasa365"}


def test_repr_contract_defaults():
    contract = repr_contract_from_cfg(OmegaConf.create({"dataloader": {}}))
    assert contract == {"representation": "joint"}


def test_repr_contract_preserves_optional_legacy_markers():
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "action_mode": "legacy",
                "binary_action_dims": [3],
                "gripper_convention": "minus1_closed_plus1_open",
            }
        }
    )
    assert repr_contract_from_cfg(cfg) == {
        "representation": "legacy",
        "binary_action_dims": [3],
        "gripper_convention": "minus1_closed_plus1_open",
    }


def test_pong_contract_uses_architecture_not_deploy_override():
    training_cfg = OmegaConf.create({"dataloader": TRAINING_DL})
    contract = repr_contract_from_cfg(training_cfg)
    engine = SimpleNamespace(architecture=SimpleNamespace(repr_contract=contract))
    merged = OmegaConf.merge(
        training_cfg,
        OmegaConf.create({"dataloader": {"action_mode": "joint"}}),
    )
    pong = PolicyServer(engine=engine, cfg=merged)._ckpt_contract()
    assert pong == contract


def test_pong_contract_empty_without_architecture():
    server = PolicyServer(engine=SimpleNamespace(architecture=None), cfg=OmegaConf.create({}))
    assert server._ckpt_contract() == {}


def test_marvin_contract_infers_fixed_wire_layout_from_training_dataset():
    # Existing checkpoints record the dataset, not redundant wire-format fields.
    cfg = OmegaConf.create({"dataloader": {"type": "marvin", "action_mode": "eef"}})
    assert repr_contract_from_cfg(cfg) == {
        "representation": "eef",
        "raw_action_dim": 62,
        "rot6d_convention": "column",
        "action_layout": "left_eef9,left_hand22,right_eef9,right_hand22",
    }


def test_other_eef_datasets_do_not_advertise_marvin_layout():
    cfg = OmegaConf.create({"dataloader": {"type": "robotwin", "action_mode": "eef"}})
    assert repr_contract_from_cfg(cfg) == {"representation": "eef"}


@pytest.mark.parametrize("prefix_length", [0, 10])
def test_marvin_rtc_uses_raw_62d_contract_with_unified_80d_model(prefix_length):
    training_cfg = OmegaConf.create(
        {"dataloader": {"type": "marvin", "action_mode": "eef", "num_frames": 33}}
    )
    arch = SimpleNamespace(
        repr_contract=repr_contract_from_cfg(training_cfg),
        training_rtc=SimpleNamespace(max_delay_steps=16),
        action_dim=80,
    )
    prefix = np.arange(prefix_length * 62, dtype=np.float32).reshape(prefix_length, 62)

    def generate(conditions):
        np.testing.assert_array_equal(conditions["action_prefix"], prefix)
        return {"actions": np.zeros((32, 62), dtype=np.float32)}

    # Deployment overrides must not change the saved training representation.
    merged_cfg = OmegaConf.merge(
        training_cfg, {"dataloader": {"type": "robotwin", "action_mode": "joint"}}
    )
    server = PolicyServer(SimpleNamespace(architecture=arch, generate=generate), merged_cfg)
    server._policy = SimpleNamespace(_build_conditions=lambda obs: {})
    server._obs_preprocessor = SimpleNamespace(preprocess=lambda obs: obs)
    server._execution_config = SimpleNamespace(mode="training_rtc", inference_horizon=16)
    pong = {**server._ckpt_contract(), **server._protocol_capabilities()}
    assert pong["raw_action_dim"] == 62
    assert pong["rot6d_convention"] == "column"
    assert pong["inference_mode"] == "training_rtc"
    result = server.predict_rtc(
        dict(episode_id="test", request_id=1, observation_step=0,
             commit_step=prefix_length, prefix_length=prefix_length,
             action_prefix=prefix.tolist())
    )
    assert np.asarray(result["actions"]).shape == (32, 62)
    np.testing.assert_array_equal(np.asarray(result["actions"])[:prefix_length], prefix)
