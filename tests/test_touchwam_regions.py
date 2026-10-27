"""Regional tactile semantics through real ActionDiT, bridges, loss and sampling."""

import copy

import numpy as np
import pytest
import torch

from openwam.model import build_architecture
from openwam.model.architectures.touchwam.regional_modules import DeformPatchCodec, regional_flow_loss
from openwam.tactile.regions import CODEC, REGION_NAMES, DeformHistory, collate_regions
from tests.test_touchwam import inputs, loss_inputs, open_gates, tiny_config
from tests.test_touchwam import pair as legacy_pair


@pytest.fixture
def models(monkeypatch, tmp_path):
    base, legacy, alpha = legacy_pair.__wrapped__(monkeypatch, tmp_path)
    regional = build_architecture("touchwam_regions", tiny_config())
    regional.set_dtype_device(torch.float32, torch.device("cpu"))
    regional.init_training_schedulers()
    regional.load_openwam_checkpoint(str(alpha))
    return base, legacy, regional


def regional_inputs(n=12, batch=2):
    args = inputs(batch)
    args.update(
        tactile_maps=torch.rand(batch, 2, n, 1, 8, 8),
        tactile_valid=torch.ones(batch, 2, n, dtype=torch.bool),
        tactile_region_ids=torch.arange(n)[None].expand(batch, -1).clone(),
        tactile_layout_present=torch.ones(batch, n, dtype=torch.bool),
        noisy_tactile=torch.randn(batch, 2 * n * 4, 16),
        tactile_future_valid=torch.ones(batch, 2 * n * 4, dtype=torch.bool),
    )
    return args


def regional_loss_inputs(n=12):
    result = loss_inputs()
    region = regional_inputs(n)
    for key in ("tactile_maps", "tactile_valid", "tactile_region_ids", "tactile_layout_present"):
        result[key] = region[key]
    result["future_tactile_maps"] = region["tactile_maps"].clone()
    result["future_tactile_valid"] = region["tactile_valid"].clone()
    return result


def test_deform_roundtrip_roi_nonfinite_and_real_zero():
    codec = DeformPatchCodec(8, 4)
    maps = torch.zeros(1, 2, 3, 1, 8, 8)
    maps[0, 1, 0, 0, 3, 5] = 0.6
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    spatial = torch.ones(1, 3, 1, 8, 8, dtype=torch.bool)
    clean, observed = codec.encode(maps, valid, spatial)
    torch.testing.assert_close(codec.decode(clean, 3), maps)
    assert observed.all() and (clean[:, :4] == -1).all()
    maps[0, 0, 1, 0, 0, 0] = float("nan")
    maps[0, 0, 2, 0, 0, 0] = float("nan")
    spatial[0, 2, 0, 0, 0] = False
    clean, observed = codec.encode(maps, valid, spatial)
    assert not observed[0, 0, 1] and observed[0, 0, 2]
    assert torch.isfinite(clean).all()
    maps[0, 1, 0, 0, 0, 0] = 255
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        codec.encode(maps, valid, spatial)


@pytest.mark.parametrize("n", [10, 11, 12])
def test_variable_layout_alpha_equivalence_and_backward(models, n):
    base, _, model = models
    args = regional_inputs(n)
    baseline = {k: v for k, v in args.items() if not k.startswith(("tactile", "noisy_tactile"))}
    out = model(**args)
    for a, b in zip(out[:2], base(**baseline)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert out[2].shape == (2, 8 * n, 16)
    open_gates(model)
    loss = model.compute_loss(**regional_loss_inputs(n), use_gradient_checkpointing=True)
    loss["loss"].backward()
    assert all(torch.isfinite(v) for v in loss.values())
    assert model.touch_v2.input_projection[0].weight.grad.abs().sum() > 0
    assert model.touch_v2.history_encoder.layers[0].linear1.weight.grad.abs().sum() > 0


def test_padding_and_disabled_palm_cannot_influence_va_or_fingers(models):
    _, _, model = models
    model.eval()
    open_gates(model)
    args = regional_inputs()
    args["tactile_layout_present"][:, 10:] = False
    before = model(**args)
    args["tactile_maps"][:, :, 10:] = float("nan")
    noise = args["noisy_tactile"].reshape(2, 2, 12, 4, 16)
    noise[:, :, 10:] = float("nan")
    after = model(**args)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not after[2].reshape(2, 2, 12, 4, 16)[:, :, 10:].any()
    args["tactile_layout_present"][:] = True
    for a, b in zip(before, model(**args, tactile_palm_mode="off")):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_region_permutation_preserves_identity_and_prediction(models):
    _, _, model = models
    model.eval()
    open_gates(model)
    args = regional_inputs()
    before = model(**args)
    order = torch.tensor([11, 3, 8, 0, 10, 1, 5, 9, 7, 2, 6, 4])
    after_args = dict(args)
    for key in ("tactile_maps", "tactile_valid"):
        after_args[key] = args[key][:, :, order]
    for key in ("tactile_region_ids", "tactile_layout_present"):
        after_args[key] = args[key][:, order]
    after_args["noisy_tactile"] = args["noisy_tactile"].reshape(2, 2, 12, 4, 16)[:, :, order].flatten(1, 3)
    after = model(**after_args)
    for a, b in zip(before[:2], after[:2]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        before[2].reshape(2, 2, 12, 4, 16)[:, :, order], after[2].reshape(2, 2, 12, 4, 16), rtol=1e-5, atol=1e-6
    )


def test_complete_future_labels_required_but_absent_palms_are_not_labels(models):
    _, _, model = models
    args = regional_loss_inputs()
    args["future_tactile_valid"][:, :, 10:] = False
    with pytest.raises(ValueError, match="complete real labels"):
        model.compute_loss(**args)
    args["tactile_layout_present"][:, 10:] = False
    args["future_tactile_maps"][:, :, 10:] = float("nan")
    assert torch.isfinite(model.compute_loss(**args)["loss"])


def test_palm_dropout_removes_future_and_history_while_history_dropout_keeps_query(models):
    _, _, model = models
    args = regional_loss_inputs()
    model.tactile_spec["palm_dropout"] = 1.0
    captured = []
    hook = model.register_forward_pre_hook(lambda m, a, kw: captured.append(kw), with_kwargs=True)
    model.compute_loss(**args)
    condition = captured[-1]["touch_condition"]
    assert not condition[3][:, 10:].any()
    assert not captured[-1]["tactile_future_valid"].reshape(2, 2, 12, 4)[:, :, 10:].any()
    model.tactile_spec["palm_dropout"] = 0
    model.tactile_spec["history_dropout"] = 1.0
    model.compute_loss(**args)
    assert not captured[-1]["touch_condition"][1].any()
    assert captured[-1]["tactile_future_valid"].all()
    hook.remove()


def test_all_absent_bypasses_with_open_gates_and_has_zero_tactile_loss(models):
    base, _, model = models
    open_gates(model)
    args = regional_inputs()
    args["tactile_layout_present"].zero_()
    out = model(**args)
    baseline = {k: v for k, v in args.items() if not k.startswith(("tactile", "noisy_tactile"))}
    for a, b in zip(base(**baseline), out[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not out[2].any()
    loss_args = regional_loss_inputs()
    loss_args["tactile_layout_present"].zero_()
    losses = model.compute_loss(**loss_args)
    assert losses["loss_tactile"] == 0
    losses["loss"].backward()


def test_region_normalized_loss_and_invalid_nan():
    codec = DeformPatchCodec(8, 4)
    clean = torch.zeros(2, 12, 16)
    target = torch.zeros_like(clean)
    pred = torch.ones_like(clean, requires_grad=True)
    valid = torch.ones_like(clean, dtype=torch.bool)
    valid[1] = False
    pred.data[1] = float("nan")
    assert regional_flow_loss(codec, pred, target, clean, valid, 3) == 1
    loss = regional_flow_loss(codec, pred, target, clean, valid, 3)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("merge", [False, True])
def test_generation_independent_query_named_outputs_and_cfg(models, merge):
    _, _, model = models
    args = regional_inputs(batch=1)
    kwargs = {
        k: v
        for k, v in args.items()
        if k in ("tactile_maps", "tactile_valid", "tactile_region_ids", "tactile_layout_present", "proprio")
    }
    requested = torch.ones(1, 2, 12, dtype=torch.bool)
    requested[:, :, 11] = False
    result = model.generate(
        schedule=[(1000, 1000), (0, 0)],
        prompt="pick",
        action_num_frames=5,
        decode_video=False,
        cfg_scale=2,
        cfg_merge=merge,
        tactile_query_mask=requested,
        **kwargs,
    )
    assert result["future_tactile_maps"].shape == (2, 11, 1, 8, 8)
    assert result["tactile_region_names"] == list(REGION_NAMES[:11])
    assert result["tactile_representation"] == "deform_gray"
    assert result["future_tactile_valid"].all()
    kwargs["tactile_valid"][:, :, 10] = False
    with pytest.raises(ValueError, match="no valid history"):
        model.generate(
            schedule=[(1000, 1000), (0, 0)], prompt="pick", action_num_frames=5, decode_video=False, **kwargs
        )
    model.tactile_spec["allow_missing_history"] = True
    result = model.generate(
        schedule=[(1000, 1000), (0, 0)], prompt="pick", action_num_frames=5, decode_video=False, **kwargs
    )
    assert result["future_tactile_valid"][:, 10].all() and not result["tactile_history_available"][10]


def test_strict_metadata_roundtrip_and_explicit_legacy_migration(models, tmp_path):
    _, legacy, model = models
    path = tmp_path / "regional.safetensors"
    model.save_checkpoint(str(path))
    other = copy.deepcopy(model)
    other.load_checkpoint(str(path))
    args = regional_inputs()
    for a, b in zip(model(**args), other(**args)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    old = tmp_path / "v2.safetensors"
    legacy.save_checkpoint(str(old))
    with pytest.raises(RuntimeError, match="metadata"):
        model.load_checkpoint(str(old))
    report = model.load_legacy_touch_checkpoint(str(old))
    assert report["optimizer_resumed"] is False
    assert "touch_v2.input_projection.0.weight" in report["initialized"]
    assert "action_backbone.action_encoder.weight" in report["reused"]
    other.tactile_spec["region_names"].reverse()
    with pytest.raises(RuntimeError, match="metadata"):
        other.load_checkpoint(str(path))


def test_causal_history_black_valid_and_layout_survives_packet_loss():
    history = DeformHistory(["left.palm", "right.index.tip"], history=2, size=8)
    black = np.zeros((16, 16), np.uint8)
    history.append(0, {"left.palm": black})
    result = history.append(1 / 30, {"right.index.tip": black})
    assert result["tactile_layout_present"].all()
    assert result["tactile_valid"].tolist() == [[True, False], [False, True]]
    assert not result["tactile_maps"].any()
    result = history.append(1, {})
    assert not result["tactile_valid"].any()


def test_collate_preserves_variable_layouts_and_rejects_pressure():
    samples = []
    for n in [10, 12]:
        args = regional_loss_inputs(n)
        sample = {
            k: v[0]
            for k, v in args.items()
            if k
            in (
                "tactile_maps",
                "tactile_valid",
                "future_tactile_maps",
                "future_tactile_valid",
                "tactile_layout_present",
            )
        }
        sample.update(tactile_region_names=list(REGION_NAMES[:n]), tactile_codec=CODEC)
        samples.append(sample)
    result = collate_regions(samples, map_size=8, history=2, future_frames=2)
    assert result["tactile_region_ids"][0, 10:].tolist() == [-1, -1]
    assert not result["tactile_query_mask"][0, :, 10:].any()
    assert result["tactile_query_mask"][1].all()
    samples[0]["tactile_codec"] = "pressure"
    with pytest.raises(ValueError, match="tactile_codec"):
        collate_regions(samples, map_size=8, history=2, future_frames=2)
