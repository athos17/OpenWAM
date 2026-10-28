"""Acceptance helper tests retain the production regional/base preparation."""

from types import MethodType

import pytest
import torch
from PIL import Image

from openwam.tactile.regions import INTENSITY_CODEC
from openwam.tactile.registry import FINGERTIP_NAMES
from scripts.validation.touchwam_sharpa_pipeline import (
    ACTION_SLOTS, bounded_encoders, prepared_backward, sample_receipt, tiny_model,
)


def sample(tail=False):
    mask = torch.zeros(32, 80, dtype=torch.bool)
    mask[:1 if tail else 32, ACTION_SLOTS] = True
    return dict(training_modality="video_action_tactile", prompt="Put the gear on the base.",
                video=[Image.new("RGB", (320, 384), (20 + i, 40, 90)) for i in range(9)],
                video_mask=torch.tensor([True] + [not tail] * 8),
                action=torch.rand(32, 80), action_mask=mask,
                proprio=torch.rand(1, 80), proprio_mask=mask[:1].clone(),
                tactile_maps=torch.rand(4, 10, 1, 32, 32), tactile_valid=torch.ones(4, 10, dtype=torch.bool),
                future_tactile_maps=torch.zeros(8, 10, 1, 32, 32) if tail else torch.rand(8, 10, 1, 32, 32),
                future_tactile_valid=torch.full((8, 10), not tail, dtype=torch.bool),
                tactile_region_names=FINGERTIP_NAMES, tactile_sensor_names=["sharpa_real"] * 10,
                tactile_layout_present=torch.ones(10, dtype=torch.bool),
                tactile_spatial_valid=torch.ones(10, 1, 32, 32, dtype=torch.bool),
                tactile_query_mask=torch.ones(8, 10, dtype=torch.bool), tactile_codec=INTENSITY_CODEC,
                tactile_history_times_s=torch.arange(-3, 1).float() / 30,
                future_tactile_times_s=torch.arange(1, 9).float() / 10,
                tactile_history_source_times_s=torch.arange(-3, 1).double() / 30,
                future_tactile_source_times_s=torch.full((8,), float("nan")) if tail else torch.arange(1, 9).double() / 10)


def test_real_prepare_and_backward_preserve_tail_and_region_masks():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = tiny_model(torch.device("cpu"), torch.float32)
    model.preprocess = MethodType(bounded_encoders, model)
    model.set_training_runtime(use_gradient_checkpointing=True)
    ordinary, tail = sample(), sample(tail=True)
    # Source order need not be canonical slot order; named identities drive IDs.
    tail["tactile_region_names"] = FINGERTIP_NAMES[::-1]
    result = prepared_backward(model, [ordinary, tail])
    assert result["tactile_gradient_l1"] > 0
    assert result["prepared_shapes"]["tactile_maps"] == [2, 4, 10, 1, 32, 32]
    result = prepared_backward(model, [tail], tail=True)
    assert result["losses"]["loss_tactile"] == result["losses"]["loss_video"] == 0
    assert result["losses"]["loss_action"] > 0


def test_sample_receipt_rejects_noncausal_or_unused_slots():
    data = sample()
    receipt = sample_receipt(data, index=0, position="first")
    assert receipt["action_valid"] == 32 * 62
    data["tactile_history_source_times_s"][-1] = 1 / 30
    with pytest.raises(ValueError, match="Noncausal"):
        sample_receipt(data, index=0, position="first")
    data["tactile_history_source_times_s"][-1] = 0
    data["action_mask"][0, 79] = True
    with pytest.raises(ValueError, match="unused action slots"):
        sample_receipt(data, index=0, position="first")
