"""training.touchwam_p1: the host-sync-free paths compute the same values as the reference paths."""
import pytest
import torch

from openwam.train.utils.modality_optimizer import (
    DeferredModalitySync, _check_layout_states, _local_layout_state, validate_global_modality_layout,
)
from tests.test_touchwam_heterogeneous import inputs, model  # noqa: F401  (fixture)


def tactile_samples():
    return [{"training_modality": "video_tactile", "tactile_region_names": ["one"]},
            {"training_modality": "video_tactile", "tactile_region_names": ["one", "two"]}]


def run(model, data, seed, **fastpath):
    model.configure_training_fastpath(**fastpath)
    torch.manual_seed(seed)
    return model.compute_loss(**data)


@pytest.mark.parametrize("kind", ["deform", "rgb", "pressure"])
def test_fastpath_losses_and_encoder_activity_match_reference(model, kind):
    data = inputs(kind)
    data["tactile_encoder_ids"][1, 1] = -1  # padded region: never activates an encoder
    data["tactile_layout_present"][1, 1] = False
    data["tactile_region_ids"][1, 1] = -1
    data["tactile_valid"][1, :, 1] = False
    data["future_tactile_valid"][1, :, 1] = False
    data["tactile_observation_spatial_valid"][1, 1] = False
    ref = run(model, {k: v.clone() if torch.is_tensor(v) else v for k, v in data.items()}, 7,
              validate_inputs=True, encoder_activity_mask=False)
    fast = run(model, data, 7, validate_inputs=False, encoder_activity_mask=True)
    for key in ("loss", "loss_video", "loss_action", "loss_tactile"):
        torch.testing.assert_close(fast[key], ref[key], rtol=0, atol=0)
    mask = fast["observation_encoder_active_mask"]
    assert mask.dtype == torch.int64 and mask.shape == (3,)
    assert mask.nonzero().flatten().tolist() == ref["observation_encoder_active_ids"].tolist()


def test_encoder_activity_mask_is_empty_after_dropout_or_nonfinite_history(model):
    model.configure_training_fastpath(encoder_activity_mask=True)
    model.tactile_spec["history_dropout"] = 1.0
    assert run(model, inputs(), 1)["observation_encoder_active_mask"].sum() == 0
    model.tactile_spec["history_dropout"] = 0.0
    data = inputs()
    data["tactile_observation_maps"].fill_(float("nan"))
    assert run(model, data, 1)["observation_encoder_active_mask"].sum() == 0


def test_skipped_validation_still_rejects_layout_shape_errors(model):
    data = inputs()
    data["tactile_layout_present"] = data["tactile_layout_present"][:, :1]
    with pytest.raises(ValueError, match="Invalid region layout"):
        run(model, data, 1, validate_inputs=False)


def test_layout_state_split_matches_blocking_check():
    samples = tactile_samples()
    assert _check_layout_states([_local_layout_state(samples)]) == validate_global_modality_layout(samples, "cpu")
    bad = [_local_layout_state(samples), [0, 1, *_local_layout_state(samples)[2:]]]
    with pytest.raises(ValueError, match="one global modality"):
        _check_layout_states(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host buffer and CUDA event")
def test_deferred_sync_single_process_resolves_layout_codec_and_encoders():
    samples = tactile_samples()
    sync = DeferredModalitySync(torch.device("cuda"), 4, blocking_steps=0)
    sync.start_layout(samples, step=5)
    sync.start_encoders(torch.tensor([0, 1, 0, 1], device="cuda"))
    modality, _, ids = sync.resolve()
    assert modality == validate_global_modality_layout(samples, "cpu") and ids == [1, 3]
    with pytest.raises(RuntimeError, match="before start_layout"):
        sync.resolve()
