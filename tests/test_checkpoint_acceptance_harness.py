"""Guard the validation wrapper's interface to the real production trainer."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.utils.checkpoint

from openwam.train import openwam_trainer


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('TOUCHWAM_ACCEPTANCE_PHASE', 'fresh')
    monkeypatch.setenv('TOUCHWAM_ACCEPTANCE_REPORTS', str(tmp_path))
    # Register originals for restoration: importing the entry point installs
    # its validation-only instrumentation before scripts.train is imported.
    monkeypatch.setattr(openwam_trainer, 'OpenWAMTrainer', openwam_trainer.OpenWAMTrainer)
    monkeypatch.setattr(torch.utils.checkpoint, 'checkpoint', torch.utils.checkpoint.checkpoint)
    path = Path(__file__).resolve().parents[1]/'scripts/validation/touchwam_checkpoint_acceptance.py'
    spec = importlib.util.spec_from_file_location('checkpoint_acceptance_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_preserves_real_trainer_three_value_interface(harness, monkeypatch):
    cls = harness.AcceptanceTrainer
    optimizer, loader, scheduler = object(), object(), object()
    monkeypatch.setattr(cls.__bases__[0], 'prepare_accelerate', lambda self, *args: (optimizer, loader, scheduler))
    instance = cls.__new__(cls)
    assert instance.prepare_accelerate(None, None, None) == (optimizer, loader, scheduler)
    assert instance.acceptance_scheduler is scheduler


def test_snapshot_distinguishes_weights_moments_and_scheduler(harness):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 3)).sum().backward(); optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    instance = harness.AcceptanceTrainer.__new__(harness.AcceptanceTrainer)
    instance.architecture = model
    instance.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    instance._modality_optimizer = optimizer
    instance.acceptance_scheduler = scheduler
    before = instance.snapshot()
    assert before == instance.snapshot()
    with torch.no_grad(): model.weight[0, 0] += 1
    assert before['model'] != instance.snapshot()['model']
    optimizer.state[model.weight]['exp_avg'][0, 0] += 1
    assert before['optimizer'] != instance.snapshot()['optimizer']
    scheduler.step()
    assert before['scheduler'] != instance.snapshot()['scheduler']
