import copy
from types import SimpleNamespace

import pytest
import torch

from openwam.train.utils.modality_optimizer import (
    ModalityAwareAdamW, build_modality_groups, validate_global_modality_layout,
)
from scripts.validation.touchwam_multisource_gpu import (
    check_inactive_groups, snapshot_groups, tiny_model,
)


def optimizer_fixture(rgb_enabled=False):
    streams = ("video", "action", "proprio", "tactile")
    if rgb_enabled:
        streams += ("tactile_intensity", "tactile_rgb")
    groups = [{"params": [torch.nn.Parameter(torch.ones(5))], "touchwam_stream": stream,
               "touchwam_active_in_window": False} for stream in streams]
    optimizer = ModalityAwareAdamW(groups, lr=0.1, weight_decay=0.2)
    return optimizer


def zero_filled_step(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = torch.zeros_like(p)
    optimizer.step()


def test_zero_filled_inactive_partitions_preserve_weights_moments_and_step():
    opt = optimizer_fixture()
    opt.activate("video_action_tactile")
    for group in opt.param_groups:
        group["params"][0].grad = torch.ones(5)
    opt.step()
    before = snapshot_groups(opt)
    opt.activate("video_only")
    zero_filled_step(opt)
    assert check_inactive_groups(opt, before, {"video"}) == ["action", "proprio", "tactile"]
    assert not torch.equal(opt.param_groups[0]["params"][0], before["video"][0][0])
    assert all(not group["touchwam_active_in_window"] for group in opt.param_groups)


def test_activation_or_survives_accumulation_and_resume():
    opt = optimizer_fixture()
    opt.activate("video_action_tactile")
    opt.activate("video_only")
    saved = copy.deepcopy(opt.state_dict())
    restored = optimizer_fixture()
    restored.load_state_dict(saved)
    assert all(g["touchwam_active_in_window"] for g in restored.param_groups)
    zero_filled_step(restored)
    assert all(g["params"][0] in restored.state for g in restored.param_groups)
    assert all(restored.state[g["params"][0]]["step"] == 1 for g in restored.param_groups)
    restored.activate("video_tactile")
    restored.reset_activation()  # overflow/skipped optimizer boundary
    assert all(not g["touchwam_active_in_window"] for g in restored.param_groups)


def test_groups_cover_parameters_once_and_keep_proprio_base_lr():
    torch.set_num_threads(1)
    model = tiny_model(torch.device("cpu"), torch.float32)
    groups = build_modality_groups(model, action_lr=1e-3, video_lr=1e-4)
    indexed = {g["touchwam_stream"]: g for g in groups}
    assert indexed["action"]["lr"] == 1e-3 and indexed["video"]["lr"] == 1e-4
    assert "lr" not in indexed["proprio"] and "lr" not in indexed["tactile"]
    assert sum(p.numel() for g in groups for p in g["params"]) == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


def test_trainer_builds_stream_optimizer_and_rejects_accelerate_cpu_replacement():
    from openwam.train.openwam_trainer import OpenWAMTrainer

    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.architecture = tiny_model(torch.device("cpu"), torch.float32)
    trainer.cfg = SimpleNamespace(training=SimpleNamespace(
        learning_rate=1e-4, weight_decay=0.1, offload_optimizer_device="none"
    ))
    assert trainer.build_optimizer() is trainer._modality_optimizer
    trainer.cfg.training.offload_optimizer_device = "cpu"
    with pytest.raises(ValueError, match="offload_optimizer_device=none"):
        trainer.build_optimizer()


def test_tactile_layout_contract_accepts_different_region_counts_but_rejects_missing_layout():
    assert validate_global_modality_layout([
        {"training_modality": "video_tactile", "tactile_region_names": ["one"]},
        {"training_modality": "video_tactile", "tactile_region_names": ["one", "two"]},
    ], "cpu") == "video_tactile"
    for samples in ([{"training_modality": "video_tactile"}],
                    [{"training_modality": "video_only", "tactile_region_names": ["one"]}],
                    [{"training_modality": "video_only"}, {"training_modality": "video_action"}]):
        with pytest.raises(ValueError, match="global modality"):
            validate_global_modality_layout(samples, "cpu")


def test_rgb_heads_grouped_separately_and_all_parameters_covered_once():
    model = tiny_model(torch.device("cpu"), torch.float32, rgb_enabled=True)
    groups = build_modality_groups(model)
    indexed = {group["touchwam_stream"]: {id(p) for p in group["params"]} for group in groups}
    assert set(indexed) == {"video", "action", "proprio", "tactile", "tactile_intensity", "tactile_rgb"}
    for name, module in model.touch_v2.named_children():
        owner = ("tactile_rgb" if name.startswith("rgb_") else "tactile_intensity"
                 if name in ("input_projection", "spatial_projection", "output") else "tactile")
        assert {id(p) for p in module.parameters() if p.requires_grad} <= indexed[owner]
    assert sum(len(ids) for ids in indexed.values()) == len({id(p) for p in model.parameters() if p.requires_grad})


@pytest.mark.parametrize("codec,head", [("tactile_intensity_v1", "tactile_intensity"),
                                       ("tactile_rgb_v1", "tactile_rgb")])
def test_representation_only_step_preserves_other_adapter_weights_and_moments(codec, head):
    opt = optimizer_fixture(rgb_enabled=True)
    # Seed both adapters' moments so the test catches momentum as well as decay.
    opt.activate("video_action_tactile", "tactile_intensity_v1")
    opt.activate("video_action_tactile", "tactile_rgb_v1")
    for group in opt.param_groups:
        group["params"][0].grad = torch.ones(5)
    opt.step()
    before = snapshot_groups(opt)
    opt.activate("video_tactile", codec)
    zero_filled_step(opt)
    inactive = check_inactive_groups(opt, before, {"video", "tactile", head})
    assert set(inactive) == {"action", "proprio", "tactile_rgb" if head == "tactile_intensity" else "tactile_intensity"}
    for group in opt.param_groups:
        if group["touchwam_stream"] in {"video", "tactile", head}:
            assert opt.state[group["params"][0]]["step"] == 2


def test_accumulation_can_update_both_representations_after_state_restore():
    opt = optimizer_fixture(rgb_enabled=True)
    opt.activate("video_tactile", "tactile_intensity_v1")
    opt.activate("video_tactile", "tactile_rgb_v1")
    restored = optimizer_fixture(rgb_enabled=True)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    assert {group["touchwam_stream"] for group in restored.param_groups if group["touchwam_active_in_window"]} == {
        "video", "tactile", "tactile_intensity", "tactile_rgb"}
    zero_filled_step(restored)
    assert {group["touchwam_stream"] for group in restored.param_groups if group["params"][0] in restored.state} == {
        "video", "tactile", "tactile_intensity", "tactile_rgb"}


def test_global_layout_rejects_different_codecs_on_same_rank_or_remote_rank(monkeypatch):
    import openwam.train.utils.modality_optimizer as module

    intensity = dict(training_modality="video_tactile", tactile_region_names=["right.thumb.tip"],
                     tactile_codec="tactile_intensity_v1")
    rgb = dict(intensity, tactile_codec="tactile_rgb_v1")
    for samples in ([intensity, rgb], [dict(intensity, tactile_codec=None)]):
        with pytest.raises(ValueError, match="global modality"):
            validate_global_modality_layout(samples, "cpu")
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)

    def remote_rgb(states, local):
        states[0].copy_(local)
        states[1].copy_(local)
        states[1][2] = 3

    monkeypatch.setattr(module.dist, "all_gather", remote_rgb)
    with pytest.raises(ValueError, match="tactile codec"):
        validate_global_modality_layout([intensity], "cpu")
    assert validate_global_modality_layout([rgb], "cpu") == "video_tactile"
