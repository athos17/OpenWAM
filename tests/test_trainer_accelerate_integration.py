"""Production trainer loop/checkpoint IO on a tiny CPU model and real Accelerate.

No dataset conversion, downloads, GPU, or full-model initialization is needed.
Only the expensive constructor is bypassed; optimizer, prepare, train, logging,
save, finish and resume paths are the production implementation.
"""
import json
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.checkpointing import find_latest_accel_state
from openwam.train.utils.training_utils import align_stop_step, optimizer_steps_through


@pytest.fixture(autouse=True)
def cpu_only_optional_backend_detection(monkeypatch):
    # Accelerate probes an installed DeepSpeed even on its plain CPU backend.
    # Hide that optional backend so these CPU tests need no CUDA compiler.
    monkeypatch.setattr("accelerate.utils.other.is_deepspeed_available", lambda: False)


class TinyDataset(torch.utils.data.Dataset):
    names = ["tiny"]

    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"training_modality": "video_only", "x": torch.ones(1), "_dataset_name": "tiny"}


class TinyArchitecture(torch.nn.Module):
    model_version = "touchwam_regions_v1"
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.video_backbone = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.video_backbone.weight)

    def get_trainable_modules(self):
        return {"video_backbone": self.video_backbone}

    def set_dtype_device(self, dtype, device):
        self.to(dtype=dtype, device=device)

    def move_frozen_to_device(self, device):
        pass

    def save_assets_for_deployment(self, path, cfg):
        pass

    def save_checkpoint(self, path, state_dict=None):
        save_file(state_dict or self.state_dict(), path)

    def prepare_inputs(self, samples):
        return {"x": torch.stack([sample["x"] for sample in samples])}

    def compute_loss(self, x, **kwargs):
        loss = self.video_backbone(x).square().mean()
        return {"loss": loss, "loss_video": loss}


def make_trainer(root, *, length=10, max_steps=None, resume=None, save_steps=3,
                 keep=-1, full_state=True, accum=4):
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({
        "training": {"num_epochs": 1, "max_steps": max_steps, "batch_size": 1,
            "gradient_accumulation_steps": accum, "dataset_num_workers": 0,
            "max_grad_norm": 1.0, "learning_rate": 0.1, "weight_decay": 0.01,
            "offload_optimizer_device": "none", "lr_scheduler": None,
            "save_steps": save_steps, "save_full_states_for_resume": full_state,
            "keep_last_k_ckpts": keep, "resume_ckpt_path": resume,
            "output_path": str(root)},
        "project": {"seed": 42, "wandb": None}, "model": {}})
    trainer.accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accum)
    trainer.architecture = TinyArchitecture()
    trainer.dataset = TinyDataset(length)
    trainer._run_seed = 42
    trainer._rank = 0
    trainer._ckpt_source_dir = None
    trainer.lambda_video = 1.0
    trainer.lambda_action = 1.0
    return trainer


def run_dir(root):
    return next(Path(root).iterdir())


def state_meta(run, step):
    return json.loads((run / f"accel_state_step_{step}" / "trainer_state.json").read_text())


def test_real_trainer_defers_save_and_stop_and_keeps_final_full_state(tmp_path):
    trainer = make_trainer(tmp_path, max_steps=1, save_steps=3)
    trainer.train()
    run = run_dir(tmp_path)
    assert (run / "checkpoint_step_4.safetensors").exists()
    assert not (run / "checkpoint_step_3.safetensors").exists()
    assert state_meta(run, 4)["opt_step"] == 1
    assert state_meta(run, 4)["optimizer_boundary"] is True
    assert trainer.architecture.video_backbone.weight.item() < 1
    rows = [json.loads(line) for line in (run / "training_metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and rows[-1]["opt_step"] == 1
    assert rows[-1]["samples_source/tiny"] == 1
    assert rows[-1]["tactile_labeled_regions"] == 0


def test_actual_checkpoint_resume_matches_uninterrupted_and_keeps_tail_counter(tmp_path):
    complete = make_trainer(tmp_path / "complete")
    complete.train()
    expected = complete.architecture.video_backbone.weight.detach().clone()
    full_run = run_dir(tmp_path / "complete")
    assert state_meta(full_run, 10)["opt_step"] == 3

    pilot = make_trainer(tmp_path / "split", max_steps=4)
    pilot.train()
    run = run_dir(tmp_path / "split")
    resumed = make_trainer(tmp_path / "ignored", resume=str(run))
    resumed.train()
    assert torch.equal(resumed.architecture.video_backbone.weight, expected)
    assert state_meta(run, 10)["opt_step"] == 3
    assert find_latest_accel_state(str(run)).endswith("accel_state_step_10")
    # A finished-run resume must not delete or rewrite its saved state.
    marker = run / "accel_state_step_10" / "trainer_state.json"
    mtime = marker.stat().st_mtime_ns
    make_trainer(tmp_path / "ignored", resume=str(run)).train()
    assert marker.stat().st_mtime_ns == mtime


def test_interruption_resumes_real_optimizer_and_cosine_scheduler(tmp_path, monkeypatch):
    import openwam.train.openwam_trainer as trainer_module

    def scheduled(root, resume=None):
        trainer = make_trainer(root, resume=resume)
        trainer.cfg.training.lr_scheduler = "cosine"
        trainer.cfg.training.warmup_ratio = 0.34
        trainer.cfg.training.lr_min_ratio = 0.01
        return trainer

    complete = scheduled(tmp_path / "complete")
    complete.train()
    expected_weights = complete.architecture.video_backbone.weight.detach().clone()
    expected_optimizer = complete._modality_optimizer.state_dict()
    expected_scheduler = complete.accelerator._schedulers[0].state_dict()
    real_save = trainer_module.save_full_state

    class Interrupted(Exception):
        pass

    def interrupt_after_saved(*args, **kwargs):
        real_save(*args, **kwargs)
        if args[2] == 4:
            raise Interrupted()

    monkeypatch.setattr(trainer_module, "save_full_state", interrupt_after_saved)
    with pytest.raises(Interrupted):
        scheduled(tmp_path / "split").train()
    monkeypatch.setattr(trainer_module, "save_full_state", real_save)
    resumed = scheduled(tmp_path / "ignored", str(run_dir(tmp_path / "split")))
    resumed.train()
    assert torch.equal(resumed.architecture.video_backbone.weight, expected_weights)
    assert resumed.accelerator._schedulers[0].state_dict() == expected_scheduler
    actual_optimizer = resumed._modality_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for param_id, fields in expected_optimizer["state"].items():
        for name, value in fields.items():
            assert torch.equal(actual_optimizer["state"][param_id][name], value)


def test_final_save_between_periodic_intervals_and_positive_retention(tmp_path):
    trainer = make_trainer(tmp_path, save_steps=3, keep=2)
    trainer.train()
    run = run_dir(tmp_path)
    assert sorted(p.name for p in run.glob("checkpoint_step_*.safetensors")) == [
        "checkpoint_step_10.safetensors", "checkpoint_step_8.safetensors"]
    assert sorted(p.name for p in run.glob("accel_state_step_*")) == [
        "accel_state_step_10", "accel_state_step_8"]
    assert torch.equal(load_file(str(run / "checkpoint_step_10.safetensors"))["video_backbone.weight"],
                       trainer.architecture.video_backbone.weight)


def test_save_disabled_does_not_write_or_delete_checkpoints(tmp_path):
    trainer = make_trainer(tmp_path, save_steps=0)
    trainer.train()
    run = run_dir(tmp_path)
    assert not list(run.glob("checkpoint_step_*"))
    assert not list(run.glob("accel_state_step_*"))


@pytest.mark.parametrize("full_state", [True, False])
def test_final_checkpoint_when_periodic_interval_is_never_reached(tmp_path, full_state):
    trainer = make_trainer(tmp_path, max_steps=4, save_steps=99, full_state=full_state)
    trainer.train()
    run = run_dir(tmp_path)
    assert (run / "checkpoint_step_4.safetensors").is_file()
    assert (run / "accel_state_step_4" / "trainer_state.json").exists() is full_state


def test_resume_rejects_partial_window_before_loading_state(tmp_path):
    trainer = make_trainer(tmp_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "trainer_state.json").write_text(json.dumps({"global_step": 3, "opt_step": 0, "epoch": 0}))
    with pytest.raises(ValueError, match="inside a gradient accumulation"):
        trainer.resume_if_configured(str(partial), [None] * 10, 4)


@pytest.mark.parametrize("requested,expected", [(1, 4), (4, 4), (9, 10), (10, 10), (11, 14), (19, 20)])
def test_stop_alignment_handles_epoch_tail(requested, expected):
    assert align_stop_step(requested, 10, 4) == expected
    assert optimizer_steps_through(20, 10, 4) == 6
