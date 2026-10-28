"""Four real modality paths; missing sensors are neither black images nor actions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from openwam.model import build_architecture
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.touchwam.regional_modules import IntensityPatchCodec, regional_flow_loss
from openwam.tactile.regions import CODEC, INTENSITY_CODEC, REGION_NAMES, collate_regions
from tests.test_touchwam import VideoBackbone, open_gates, tiny_config
from tests.test_touchwam_regions import models as regional_models, regional_inputs, regional_loss_inputs


@pytest.fixture
def models(monkeypatch, tmp_path):
    return regional_models.__wrapped__(monkeypatch, tmp_path)


def modality_inputs(modality, n=2):
    args = regional_loss_inputs(n)
    if modality in ("video_only", "video_action"):
        args = {k: v for k, v in args.items() if not k.startswith(("tactile_", "future_tactile_"))}
    if modality in ("video_only", "video_tactile"):
        args.pop("actions")
        args.pop("proprio")
        args["video_only_mask"] = torch.ones(2, dtype=torch.bool)
    return args


def test_empty_and_mixed_collation_preserve_real_absence():
    empty = collate_regions([{}, {}], map_size=8, history=2, future_frames=2, codec=INTENSITY_CODEC)
    assert empty["tactile_maps"].shape == (2, 2, 0, 1, 8, 8)
    assert empty["tactile_region_ids"].shape == (2, 0)
    sample = dict(
        tactile_region_names=[REGION_NAMES[0]], tactile_codec=INTENSITY_CODEC,
        tactile_sensor_names=["sharpa_real"], tactile_layout_present=torch.ones(1, dtype=torch.bool),
        tactile_maps=torch.zeros(2, 1, 1, 8, 8), tactile_valid=torch.ones(2, 1, dtype=torch.bool),
    )
    mixed = collate_regions([{}, sample], map_size=8, history=2, future_frames=2, codec=INTENSITY_CODEC)
    assert mixed["tactile_region_ids"].tolist() == [[-1], [0]]
    assert not mixed["tactile_valid"][0].any() and mixed["tactile_valid"][1].all()
    assert mixed["tactile_query_mask"][1].all() and not mixed["future_tactile_valid"].any()
    del sample["tactile_sensor_names"]
    with pytest.raises(ValueError, match="identity"):
        collate_regions([sample], map_size=8, history=2, future_frames=2, codec=INTENSITY_CODEC)


@pytest.mark.parametrize("modality", ["video_only", "video_action", "video_tactile", "video_action_tactile"])
def test_four_paths_backward_and_absent_branches_not_called(models, modality):
    _, _, model = models
    model.training_mode = "joint_finetune"
    model._apply_training_policy()
    open_gates(model)
    actions = modality in ("video_action", "video_action_tactile")
    tactile = modality in ("video_tactile", "video_action_tactile")
    with patch.object(model.action_backbone, "prepare_state", wraps=model.action_backbone.prepare_state) as action_call:
        with patch.object(model.touch_v2.history_encoder, "forward", wraps=model.touch_v2.history_encoder.forward) as touch_call:
            losses = model.compute_loss(**modality_inputs(modality), use_gradient_checkpointing=True)
            losses["loss"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert action_call.call_count == int(actions)
    assert touch_call.call_count == int(tactile)
    assert (losses["loss_action"] > 0) == actions
    assert (losses["loss_tactile"] > 0) == tactile
    assert model.video_backbone.dit.weight.grad.abs().sum() > 0
    assert (model.action_backbone.action_encoder.weight.grad is not None) == actions
    assert (model.touch_v2.input_projection[0].weight.grad is not None) == tactile


@pytest.mark.parametrize("with_action", [False, True])
def test_empty_layout_forward_is_exact_openwam(models, with_action):
    base, _, model = models
    open_gates(model)
    args = regional_inputs()
    args = {k: v for k, v in args.items() if not k.startswith(("tactile", "noisy_tactile"))}
    if not with_action:
        args.update(noisy_actions=None, action_timestep=None, proprio=None, _strict_video_only=True)
    expected = base(**args)
    got = model(**args)
    torch.testing.assert_close(expected[0], got[0], rtol=0, atol=0)
    if with_action:
        torch.testing.assert_close(expected[1], got[1], rtol=0, atol=0)
    else:
        assert got[1] is None
    assert got[2].shape == (2, 0, 16)


def test_partial_targets_do_not_change_query_or_leak_invalid_values(models):
    _, _, model = models
    open_gates(model)
    args = modality_inputs("video_tactile")
    args["future_tactile_valid"][:, 1, 1] = False
    args["future_tactile_maps"][:, 1, 1] = float("nan")
    captured = []
    hook = model.register_forward_pre_hook(lambda m, a, kw: captured.append(kw), with_kwargs=True)
    torch.manual_seed(99)
    first = model.compute_loss(**args)
    args["future_tactile_maps"][:, 1, 1] = 10000
    torch.manual_seed(99)
    second = model.compute_loss(**args)
    hook.remove()
    torch.testing.assert_close(first["loss"], second["loss"], rtol=0, atol=0)
    assert captured[0]["tactile_future_valid"].all()  # Request did not become label validity.
    assert "future_tactile_maps" not in captured[0] and "future_tactile_valid" not in captured[0]
    for lhs, rhs in zip(captured[0]["touch_condition"], captured[1]["touch_condition"]):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    assert first["tactile_labeled_frames"] == 6


def test_unlabeled_action_dimensions_cannot_influence_any_prediction(models):
    _, _, model = models
    open_gates(model)
    args = modality_inputs("video_action_tactile")
    args["action_is_pad"] = torch.zeros_like(args["actions"], dtype=torch.bool)
    args["action_is_pad"][:, :, -2:] = True
    args["actions"][:, :, -2:] = float("nan")
    torch.manual_seed(11)
    first = model.compute_loss(**args)
    args["actions"][:, :, -2:] = 10000
    torch.manual_seed(11)
    second = model.compute_loss(**args)
    for key in ("loss", "loss_video", "loss_action", "loss_tactile"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_new_codec_roundtrip_and_source_embedding(models):
    _, _, model = models
    model.tactile_spec["codec"] = INTENSITY_CODEC
    assert model.tactile_input_contract["representation"] == "contact_intensity"
    codec = IntensityPatchCodec(8, 4)
    maps = torch.rand(1, 2, 1, 1, 8, 8)
    packed, valid = codec.encode(maps, torch.ones(1, 2, 1, dtype=torch.bool), torch.ones(1, 1, 1, 8, 8, dtype=torch.bool))
    torch.testing.assert_close(codec.decode(packed, 1), maps)
    args = regional_inputs(n=2)
    model.eval()
    open_gates(model)
    one = model(**args)
    two = model(**args, tactile_sensor_ids=torch.ones(2, 2, dtype=torch.long))
    assert not torch.allclose(one[2], two[2])


@pytest.mark.parametrize("modality", ["video_only", "video_action", "video_tactile", "video_action_tactile"])
def test_real_prepare_inputs_handles_all_modalities_and_seconds(models, monkeypatch, modality):
    _, _, model = models
    sample = {"video": [torch.zeros(3, 8, 8)], "prompt": "pick", "training_modality": modality}
    has_action = modality in ("video_action", "video_action_tactile")
    has_tactile = modality in ("video_tactile", "video_action_tactile")
    # Exact base preprocessing/collation, replacing expensive image encoding only.
    monkeypatch.setattr(model, "preprocess", lambda **kw: dict(
        input_latents=torch.randn(1, 4, 3, 2, 2), context=torch.randn(1, 4, 32),
        first_frame_latents=torch.zeros(1, 4, 1, 2, 2), context_mask=torch.ones(1, 4, dtype=torch.bool),
    ))
    if has_action:
        sample.update(action=torch.rand(4, 7), proprio=torch.rand(7), action_mask=torch.ones(4, 7, dtype=torch.bool))
    else:
        sample.update(action=torch.full((4, 7), float("nan")), proprio=torch.full((7,), float("nan")),
                      action_mask=torch.zeros(4, 7, dtype=torch.bool), proprio_mask=torch.zeros(1, 7, dtype=torch.bool))
    if has_tactile:
        args = regional_loss_inputs(2)
        sample.update({k: v[0] for k, v in args.items() if k.startswith(("tactile_", "future_tactile_")) and k != "tactile_region_ids"})
        sample.update(tactile_region_names=list(REGION_NAMES[:2]), tactile_codec=CODEC,
                      future_tactile_times_s=[1 / 30, 2 / 30], tactile_history_times_s=[-1 / 30, 0])
    prepared = model.prepare_inputs([sample])
    assert (prepared["actions"] is not None) == has_action
    assert ("proprio" in prepared) == has_action
    assert prepared["tactile_maps"].shape[2] == (2 if has_tactile else 0)
    assert torch.isfinite(model.compute_loss(**prepared)["loss"])
    if has_tactile:
        sample["future_tactile_times_s"] = [0.1, 0.2]
        with pytest.raises(ValueError, match="times differ"):
            model.prepare_inputs([sample])


class _LossModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model.compute_loss(**inputs)["loss"]


def _distributed_worker(rank, init_file):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        # Uneven label availability must normalize like a single combined batch.
        codec = IntensityPatchCodec(8, 4)
        prediction = torch.ones(1, 4, 16, requires_grad=True)
        clean = torch.zeros_like(prediction)
        valid = torch.full_like(prediction, rank == 1, dtype=torch.bool)
        loss = regional_flow_loss(codec, prediction, clean, clean, valid, 1)
        loss.backward()
        dist.all_reduce(prediction.grad)
        torch.testing.assert_close(prediction.grad / 2, torch.full_like(prediction, 2 / 64))
        torch.manual_seed(17)
        with patch.object(BaseWAMArchitecture, "_init_video_backbone", lambda self, cfg: setattr(
            self, "video_backbone", VideoBackbone(dim=32, num_layers=3, num_heads=4)
        )):
            cfg = tiny_config()
            cfg["training_mode"] = "joint_finetune"
            model = build_architecture("touchwam_regions", cfg)
        model.set_dtype_device(torch.float32, torch.device("cpu"))
        model.init_training_schedulers()
        ddp = nn.parallel.DistributedDataParallel(_LossModule(model), find_unused_parameters=True)
        cases = [("video_only", "none"), ("video_action", "none"), ("video_tactile", "labels"),
                 ("video_action_tactile", "queries"), ("video_action_tactile", "empty"), ("video_only", "none")]
        for modality, missing in cases:
            ddp.zero_grad(set_to_none=True)
            inputs = modality_inputs(modality)
            if rank == 0 and "future_tactile_valid" in inputs:
                inputs["future_tactile_valid"].zero_()
                if missing == "queries":
                    inputs["tactile_valid"].zero_()
                    inputs["tactile_query_mask"] = torch.zeros_like(inputs["future_tactile_valid"])
                elif missing == "empty":
                    inputs = {k: v for k, v in inputs.items() if not k.startswith(("tactile_", "future_tactile_"))}
            loss = ddp(inputs)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ddp.parameters())
    finally:
        dist.destroy_process_group()


def test_cpu_ddp_modality_switch_and_rank_without_labels(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path / "gloo_init"),), nprocs=2, join=True)


@pytest.mark.parametrize("merge", [False, True])
def test_generation_without_tactile_returns_empty_regional_output(models, merge):
    _, _, model = models
    args = regional_inputs(batch=1)
    result = model.generate(
        schedule=[(1000, 1000), (0, 0)], prompt="pick", action_num_frames=5,
        decode_video=False, proprio=args["proprio"], cfg_scale=2, cfg_merge=merge,
    )
    assert result["future_tactile_maps"].shape == (2, 0, 1, 8, 8)
    assert result["tactile_region_names"] == []
    assert result["actions"].shape == (4, 7)


def test_video_loss_is_not_diluted_by_unlabeled_samples(models):
    _, _, model = models
    args = modality_inputs("video_only")
    args["video_is_pad"] = torch.tensor([[False, False], [True, True]])
    prediction = torch.ones_like(args["input_latents"])
    target = torch.zeros_like(prediction)
    indices = torch.zeros(2, dtype=torch.long)
    expected = model.video_scheduler.linear_timesteps_weights[0]
    assert model._compute_video_loss(prediction, target, indices, args, model.device) == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("modality", ["video_only", "video_action", "video_tactile", "video_action_tactile"])
def test_real_wan_four_modalities_with_checkpointed_backward(monkeypatch, dtype, modality):
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan21

    def init_video(self, cfg):
        dit = WanModel(
            dim=96, in_dim=4, out_dim=4, ffn_dim=192, text_dim=32, freq_dim=32,
            eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
            has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        )
        self.video_backbone = Wan21(SimpleNamespace(dit=dit, scheduler=ActionScheduler()))

    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", init_video)
    cfg = tiny_config()
    cfg.update(bridge_layers=[0, 1], attn_head_dim=24, video_dim=96, training_mode="joint_finetune")
    cfg["tactile"].update(bridge_layers=[0, 1], codec=INTENSITY_CODEC)
    model = build_architecture("touchwam_regions", cfg)
    model.set_dtype_device(dtype, torch.device("cpu"))
    model.init_training_schedulers()
    open_gates(model)
    args = modality_inputs(modality)
    args["input_latents"] = torch.randn(2, 4, 3, 4, 4)
    args["first_frame_latents"] = args["input_latents"][:, :, :1].clone()
    args["fuse_vae_embedding_in_latents"] = True
    args = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in args.items()}
    loss = model.compute_loss(**args, use_gradient_checkpointing=True)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert (model.touch_v2.input_projection[0].weight.grad is not None) == ("tactile" in modality)
    assert (model.action_backbone.action_encoder.weight.grad is not None) == ("action" in modality)
    if modality == "video_tactile":
        forward_args = regional_inputs(2)
        forward_args.update(noisy_actions=None, action_timestep=None, proprio=None, _strict_video_only=True,
                            latents=args["input_latents"], first_frame_latents=args["first_frame_latents"],
                            timestep=torch.full((2,), 500.0), fuse_vae_embedding_in_latents=True)
        forward_args = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                        for k, v in forward_args.items()}
        model.eval()
        before = model(**forward_args)[0]
        forward_args["noisy_tactile"] = forward_args["noisy_tactile"] + 100
        after = model(**forward_args)[0]
        torch.testing.assert_close(before[:, :, :1], after[:, :, :1], rtol=0, atol=0)
        assert not torch.equal(before[:, :, 1:], after[:, :, 1:])
