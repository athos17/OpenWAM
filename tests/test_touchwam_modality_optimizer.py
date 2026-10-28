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


def optimizer_fixture():
    groups = [{"params": [torch.nn.Parameter(torch.ones(5))], "touchwam_stream": stream,
               "touchwam_active_in_window": False} for stream in ("video", "action", "proprio", "tactile")]
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
