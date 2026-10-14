from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from openwam.model.training_rtc import RTCConfig, condition_training_actions, rtc_config


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_delay(value):
    with pytest.raises(ValueError):
        rtc_config({"enabled": True, "max_delay_steps": value})


def test_training_prefix_mask_padding_and_gradient(monkeypatch):
    from openwam.model.architectures.base import BaseWAMArchitecture

    monkeypatch.setattr(torch, "rand", lambda *a, **k: torch.tensor([0.9, 0.9, 0.9]))
    actions = torch.randn(3, 5, 4)
    noisy = torch.randn_like(actions)
    pad = torch.zeros_like(actions, dtype=torch.bool)
    pad[0, :, 3] = True  # inactive unified dimension
    pad[1, 2:] = True  # only two real steps
    pad[2] = True  # no action supervision
    mixed, times, mask = condition_training_actions(
        actions, noisy, torch.tensor([500.0, 600.0, 700.0]), pad, RTCConfig(True, 3)
    )
    assert times.tolist() == [[0, 0, 0, 500, 500], [0, 600, 600, 600, 600], [700] * 5]
    torch.testing.assert_close(mixed[0, :3], actions[0, :3])
    torch.testing.assert_close(mixed[0, 3:], noisy[0, 3:])
    pred = torch.randn_like(actions, requires_grad=True)
    scheduler = SimpleNamespace(training_weight=lambda ids: torch.ones(3))
    loss = BaseWAMArchitecture._compute_action_loss(
        None, pred, torch.zeros_like(pred), torch.arange(3), scheduler, {"action_is_pad": mask}, "cpu"
    )
    loss.backward()
    assert (pred.grad[mask] == 0).all()
    assert (pred.grad[~mask] != 0).all()
    assert torch.isfinite(loss)


def test_zero_delay_loss_matches_baseline():
    from tests.test_openwam_trainer import _make_fake_loss_inputs, _make_tiny_arch

    arch = _make_tiny_arch()
    arch.init_training_schedulers(100)
    inputs = _make_fake_loss_inputs(B=2)
    actions = torch.randn(2, 5, 7)
    torch.manual_seed(71)
    baseline = arch.compute_loss(actions=actions, **inputs)["loss"]
    arch.training_rtc = RTCConfig(True, 0)
    torch.manual_seed(71)
    rtc = arch.compute_loss(actions=actions, **inputs)["loss"]
    torch.testing.assert_close(rtc, baseline)
    rtc.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in arch.action_backbone.parameters())


def test_actual_mot_per_token_timestep_and_compile():
    from tests.test_mot_driver import _make_action_dit

    ab = _make_action_dit(num_layers=1).eval()
    actions = torch.randn(2, 5, 7)
    context = torch.randn(2, 3, 32)
    t = torch.tensor([400.0, 600.0])
    keys = tuple(ab.state_dict())

    def run(times):
        state = ab.prepare_state(actions, times, context=context)
        q, k, v, post = ab.pre_attn_at_layer_for_compile(0, state)
        state = ab.post_attn_at_layer_for_compile(0, state, v, post)
        return ab.extract_prediction(state)

    scalar = run(t)
    tokens = t[:, None].expand(2, 5).clone()
    torch.testing.assert_close(run(tokens), scalar)
    tokens[:, :2] = 0
    assert not torch.allclose(run(tokens), scalar)
    compiled = torch.compile(run, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(tokens), run(tokens))
    assert tuple(ab.state_dict()) == keys


@pytest.mark.parametrize("length", [0, 1, 3])
def test_generation_keeps_prefix_at_every_step_and_output(length):
    from tests.test_wam_architecture import _make_inactive_dim_probe_arch

    arch = _make_inactive_dim_probe_arch()
    prefix = torch.randn(1, 4, 8)
    prefix[:, :, [1, 3, 4, 6, 7]] = 0
    seen_times = []
    original = arch.forward

    def forward(actions, times, **kwargs):
        seen_times.append(times.clone())
        return original(actions, times, **kwargs)

    arch.forward = forward
    mask = torch.zeros(8, dtype=torch.bool)
    mask[[0, 2, 5]] = True
    result = arch.generate(
        schedule=[(1000, 1000), (500, 500), (0, 0)],
        prompt="",
        num_frames=5,
        decode_video=False,
        action_prefix=prefix,
        prefix_length=length,
        active_action_mask=mask,
    )
    for actions, times in zip(arch.seen_action_inputs, seen_times):
        torch.testing.assert_close(actions[:, :length], prefix[:, :length])
        assert times.shape == (1, 4)
        assert (times[:, :length] == 0).all()
    np.testing.assert_array_equal(result["actions"][:length], prefix[0, :length].numpy())
    assert (result["actions"][:, ~mask] == 0).all()


def test_action_normalization_does_not_use_state_statistics_or_map():
    from openwam.dataloader.transforms.normalize import Normalizer
    from openwam.deploy.model_loader import _UnifyAwareNormalizer

    stats = {
        "min": np.array([0.0, 10.0]),
        "max": np.array([2.0, 14.0]),
        "_normalize_stats": {"min": np.array([100.0, 100.0]), "max": np.array([200.0, 200.0])},
    }
    normalizer = _UnifyAwareNormalizer(Normalizer("min_max", stats), np.array([1, 3]), 5, np.array([0, 2]))
    raw = np.array([[2.0, 14.0]], dtype=np.float32)
    np.testing.assert_array_equal(normalizer.normalize_action(raw), [[0, 1, 0, 1, 0]])
    np.testing.assert_allclose(normalizer.unnormalize(normalizer.normalize_action(raw)), raw)
    assert not np.array_equal(normalizer.normalize(raw), normalizer.normalize_action(raw))


def _rtc_server():
    from openwam.deploy.server import PolicyServer

    arch = SimpleNamespace(training_rtc=RTCConfig(True, 3), action_dim=8, repr_contract={"raw_action_dim": 2})
    engine = SimpleNamespace(architecture=arch, generate=lambda conditions: {"actions": np.zeros((5, 2))})
    server = PolicyServer(engine, OmegaConf.create({"dataloader": {"num_frames": 6}}))
    server._policy = SimpleNamespace(_build_conditions=lambda obs: {"observation": obs})
    server._execution_config = SimpleNamespace(mode="training_rtc", inference_horizon=2)
    server._obs_preprocessor = SimpleNamespace(preprocess=lambda obs: obs)
    return server


def _request():
    return dict(
        episode_id="test",
        request_id=1,
        observation_step=3,
        commit_step=5,
        prefix_length=2,
        action_prefix=[[1.0, 2.0], [3.0, 4.0]],
    )


def test_server_returns_exact_prefix_and_capabilities():
    server = _rtc_server()
    result = server.predict_rtc(_request())
    assert result["actions"][:2] == _request()["action_prefix"]
    assert result["count"] == 5
    caps = server._protocol_capabilities()
    assert caps["supported_requests"] == ["obs_rtc", "reset", "ping"]
    assert caps["rtc_max_delay_steps"] == 3


@pytest.mark.parametrize(
    "patch",
    [
        dict(prefix_length=True),
        dict(commit_step=6),
        dict(action_prefix=[[1, 2]]),
        dict(action_prefix=[[float("nan"), 2], [3, 4]]),
        dict(episode_id=""),
    ],
)
def test_server_rejects_invalid_prefix(patch):
    from openwam.deploy.obs_preprocess import ObsValidationError

    request = {**_request(), **patch}
    with pytest.raises(ObsValidationError):
        _rtc_server().predict_rtc(request)


def test_self_attn_rtc_train_step_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
    from tests.test_openwam_trainer import _make_fake_loss_inputs, _MockVideoBackbone

    config = dict(
        action_dim=7,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        video_dim=32,
        bridge_layers=(0, 1),
        text_dim=32,
        attention_mask_mode="mutual",
    )
    arch = DualSystemSelfAttnArchitecture(config)
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    arch.build_mot_driver()
    arch._device, arch._dtype = torch.device("cpu"), torch.float32
    arch.init_training_schedulers(100)
    arch.training_rtc = RTCConfig(True, 3)
    monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.tensor([0.8]))
    seen = []
    hook = arch.register_forward_pre_hook(lambda module, args: seen.append((args[0].clone(), args[1].clone())))
    actions = torch.randn(1, 5, 7)
    optimizer = torch.optim.AdamW(arch.action_backbone.parameters(), lr=1e-3)
    before = arch.action_backbone.action_decoder.weight.detach().clone()
    loss = arch.compute_loss(actions=actions, context=torch.randn(1, 3, 32), **_make_fake_loss_inputs())["loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    hook.remove()
    assert torch.isfinite(loss)
    torch.testing.assert_close(seen[0][0][:, :3], actions[:, :3])
    assert seen[0][1].shape == (1, 5)
    assert (seen[0][1][:, :3] == 0).all()
    assert not torch.equal(before, arch.action_backbone.action_decoder.weight)
    path = str(tmp_path / "rtc.safetensors")
    arch.save_checkpoint(path)
    expected = arch.action_backbone.action_decoder.weight.detach().clone()
    with torch.no_grad():
        arch.action_backbone.action_decoder.weight.zero_()
    arch.load_checkpoint(path)
    torch.testing.assert_close(arch.action_backbone.action_decoder.weight, expected)
