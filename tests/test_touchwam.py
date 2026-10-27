"""V2 contracts: real ActionDiT, deterministic tiny video backbone, CPU only."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from safetensors.torch import load_file, save_file

from openwam.model import build_architecture, resolve_architecture_config
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.touchwam.modules import TactilePatchCodec, tactile_flow_loss
from tests.test_openwam_trainer import _MockVideoBackbone


class _TrainableVideoBackbone(_MockVideoBackbone):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name in ("dit", "vae", "text_encoder"):
            module = self._modules.pop(f"_{name}")
            self.add_module(name, module)


class VideoBackbone(_TrainableVideoBackbone):
    def prepare(self, **inputs):
        state = super().prepare(**inputs)
        # Deterministic, differentiable observation path, with no real weights.
        x = inputs["latents"].permute(0, 2, 3, 4, 1).flatten(1, 3)
        state.hidden_states = x.repeat(1, 1, self.dim // x.shape[-1])
        state.context = inputs["context"]
        return state

    def finalize(self, state):
        x = self.dit(state.hidden_states)[..., :4]
        return x.transpose(1, 2).reshape(x.shape[0], 4, state.grid_frames, state.grid_height, state.grid_width)

    def preprocess_input_for_inference(self, **kwargs):
        generator = torch.Generator().manual_seed(kwargs["seed"])
        return dict(
            latents=torch.randn(1, 4, 3, 2, 2, generator=generator),
            context=torch.ones(1, 4, self.dim),
            context_mask=torch.ones(1, 4, dtype=torch.bool),
            uncond_context=torch.zeros(1, 4, self.dim),
            first_frame_latents=torch.zeros(1, 4, 1, 2, 2),
        )


def tiny_config():
    return dict(
        action_dim=7,
        state_dim=7,
        use_proprioception=True,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        attn_head_dim=8,
        text_dim=32,
        video_dim=32,
        bridge_layers=[0, 1, 2],
        attention_mask_mode="mutual",
        mot_checkpoint_mixed_attn=False,
        training_mode="new_only",
        tactile=dict(
            width=16,
            heads=4,
            layers=2,
            bridge_layers=[0, 2],
            history_layers=1,
            map_size=8,
            patch_size=4,
            history=2,
            future_frames=2,
            future_stride=1,
        ),
    )


@pytest.fixture
def pair(monkeypatch, tmp_path):
    monkeypatch.setattr(
        BaseWAMArchitecture,
        "_init_video_backbone",
        lambda self, cfg: setattr(self, "video_backbone", VideoBackbone(dim=32, num_layers=3, num_heads=4)),
    )
    cfg = tiny_config()
    base, touch = [build_architecture(name, cfg) for name in ("dual_system_self_attn", "touchwam_v2")]
    for model in (base, touch):
        model.set_dtype_device(torch.float32, torch.device("cpu"))
        model.init_training_schedulers()
    path = tmp_path / "alpha.safetensors"
    base.save_checkpoint(str(path))
    touch.load_openwam_checkpoint(str(path))
    return base, touch, path


def inputs(batch=2):
    g = torch.Generator().manual_seed(12)
    maps = torch.rand(batch, 2, 10, 2, 8, 8, generator=g)
    maps[:, :, :, 0] *= 0.015
    maps[:, :, :, 1] = maps[:, :, :, 1] > 0.8
    return dict(
        noisy_actions=torch.randn(batch, 4, 7, generator=g),
        action_timestep=torch.full((batch,), 500.0),
        proprio=torch.randn(batch, 7, generator=g),
        latents=torch.randn(batch, 4, 3, 2, 2, generator=g),
        context=torch.randn(batch, 4, 32, generator=g),
        context_mask=torch.ones(batch, 4, dtype=torch.bool),
        tactile_maps=maps,
        tactile_valid=torch.ones(batch, 2, 10, dtype=torch.bool),
        noisy_tactile=torch.randn(batch, 80, 32, generator=g),
        tactile_timestep=torch.full((batch,), 700.0),
        tactile_future_valid=torch.ones(batch, 80, dtype=torch.bool),
    )


def open_gates(model):
    with torch.no_grad():
        for block in model.touch_v2.blocks:
            block.video_gate.fill_(0.2)
            block.action_gate.fill_(0.2)


def test_codec_preserves_sparse_contact_and_missing_semantics():
    codec = TactilePatchCodec(8, 4)
    maps = torch.zeros(1, 2, 10, 2, 8, 8)
    maps[0, 0, 0, :, 3, 5] = torch.tensor([0.006, 1])
    valid = torch.ones(1, 2, 10, dtype=torch.bool)
    patches, mask = codec.encode(maps, valid)
    torch.testing.assert_close(codec.decode(patches), maps)
    assert mask.all()  # empty contact is a valid observation
    maps[0, 1, 0, 0, 0, 0] = float("nan")
    patches, mask = codec.encode(maps, valid)
    assert not mask[0, 40:44].any() and torch.isfinite(patches).all()


def test_zero_gate_alpha_equivalence_and_exact_weight_reuse(pair):
    base, touch, _ = pair
    args = inputs()
    baseline = {key: value for key, value in args.items() if not key.startswith(("tactile", "noisy_tactile"))}
    expected, actual = base(**baseline), touch(**args)
    for a, b in zip(expected, actual[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert actual[2].shape == args["noisy_tactile"].shape
    extra = set(touch.state_dict()) - set(base.state_dict())
    assert extra and all(key.startswith("touch_v2.") for key in extra)
    assert all(torch.equal(value, touch.state_dict()[key]) for key, value in base.state_dict().items())


def test_bidirectional_bridges_clean_prefix_and_history_causality(pair):
    _, touch, _ = pair
    touch.eval()
    open_gates(touch)
    args = inputs()
    before = touch(**args)
    changed = touch(**dict(args, noisy_tactile=args["noisy_tactile"] + 3))
    torch.testing.assert_close(before[0][:, :, :1], changed[0][:, :, :1], rtol=0, atol=0)
    assert not torch.allclose(before[0][:, :, 1:], changed[0][:, :, 1:])
    assert not torch.allclose(before[1], changed[1])
    changed_action = touch(**dict(args, noisy_actions=args["noisy_actions"] + 3))
    assert not torch.allclose(before[2], changed_action[2])
    history, _ = touch._history(args["tactile_maps"], args["tactile_valid"], 2)
    maps = args["tactile_maps"].clone()
    maps[:, 1] = 0
    changed_history, _ = touch._history(maps, args["tactile_valid"], 2)
    torch.testing.assert_close(history[:, :40], changed_history[:, :40], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Clean future"):
        touch(**args, future_tactile_maps=maps)


def test_absent_touch_bypasses_even_with_open_gates(pair):
    base, touch, _ = pair
    open_gates(touch)
    args = inputs()
    args["tactile_valid"].zero_()
    args["tactile_future_valid"].zero_()
    baseline = {k: v for k, v in args.items() if not k.startswith(("tactile", "noisy_tactile"))}
    actual = touch(**args)
    for a, b in zip(base(**baseline), actual[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not actual[2].any()


def loss_inputs():
    args = inputs()
    args["actions"] = args.pop("noisy_actions")
    args["input_latents"] = args.pop("latents")
    args["first_frame_latents"] = args["input_latents"][:, :, :1].clone()
    args["future_tactile_maps"] = args["tactile_maps"].flip(1).clone()
    args["future_tactile_valid"] = args["tactile_valid"].clone()
    for key in ("action_timestep", "noisy_tactile", "tactile_timestep", "tactile_future_valid"):
        args.pop(key)
    return args


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prepare_inputs_validates_offsets_on_batch_device(pair, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _, touch, _ = pair
    monkeypatch.setattr(BaseWAMArchitecture, "prepare_inputs", lambda self, batch: {})
    sample = {
        key: value[0].to(device)
        for key, value in loss_inputs().items()
        if key in ("tactile_maps", "tactile_valid", "future_tactile_maps", "future_tactile_valid")
    }
    sample["future_tactile_offsets"] = torch.tensor([1, 2], device=device)
    prepared = touch.prepare_inputs([sample])
    assert prepared["future_tactile_maps"].device.type == device
    sample["future_tactile_offsets"] = torch.tensor([2, 4], device=device)
    with pytest.raises(ValueError, match="offsets differ"):
        touch.prepare_inputs([sample])
    del sample["future_tactile_offsets"]
    with pytest.raises(ValueError, match="offsets differ"):
        touch.prepare_inputs([sample])


@pytest.mark.parametrize("mode", ["new_only", "action_finetune", "joint_finetune"])
def test_loss_backward_and_training_policies(pair, mode):
    _, touch, _ = pair
    touch.training_mode = mode
    touch.freeze_modules(list(touch.preprocess_modules))
    open_gates(touch)
    touch.train()
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    params = build_trainable_parameters(SimpleNamespace(architecture=touch))
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected_params = {id(p) for p in touch.parameters() if p.requires_grad}
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == expected_params
    probes = {
        "video": touch.video_backbone.dit.weight,
        "action": touch.action_backbone.action_encoder.weight,
        "proprio": touch.proprio_encoder.weight,
        "touch": touch.touch_v2.input_projection[0].weight,
        "vae": next(touch.video_backbone.vae.parameters()),
        "text": next(touch.video_backbone.text_encoder.parameters()),
    }
    before = {name: param.detach().clone() for name, param in probes.items()}
    losses = touch.compute_loss(**loss_inputs(), use_gradient_checkpointing=True)
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["loss"].backward()
    assert touch.touch_v2.input_projection[0].weight.grad.abs().sum() > 0
    assert touch.touch_v2.blocks[0].read_va.k.weight.grad.abs().sum() > 0
    assert touch.touch_v2.history_encoder.layers[0].linear1.weight.grad.abs().sum() > 0
    for name, parameter in touch.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
        if name.startswith("action_backbone."):
            assert parameter.requires_grad == (mode != "new_only")
    optimizer.step()
    expected = {"touch"}
    if mode != "new_only":
        expected |= {"action", "proprio"}
    if mode == "joint_finetune":
        expected.add("video")
    for name, param in probes.items():
        assert (not torch.equal(before[name], param)) == (name in expected), name
        if name in expected:
            assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_checkpointed_forward_and_gradients_match(pair):
    _, touch, _ = pair
    open_gates(touch)
    touch.train()
    results, grads = [], []
    for enabled in (False, True):
        touch.zero_grad(set_to_none=True)
        result = touch(**inputs(), use_gradient_checkpointing=enabled)
        sum(value.square().mean() for value in result).backward()
        results.append(tuple(value.detach() for value in result))
        grads.append({name: p.grad.clone() for name, p in touch.named_parameters() if p.grad is not None})
    for a, b in zip(results[0], results[1]):
        torch.testing.assert_close(a, b)
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-5, atol=1e-6)


def test_masked_loss_and_zero_gate_touch_learning(pair):
    _, touch, _ = pair
    losses = touch.compute_loss(**loss_inputs())
    losses["loss"].backward()
    # Future touch learns on step one even though its writes to Alpha are zero.
    assert touch.touch_v2.blocks[0].read_va.k.weight.grad.abs().sum() > 0
    codec = touch.touch_v2.codec
    args = inputs()
    clean, valid = codec.encode(args["tactile_maps"], args["tactile_valid"])
    pred, target = torch.zeros_like(clean), torch.zeros_like(clean)
    valid[:, :4] = False
    pred[:, :4] = 100
    assert tactile_flow_loss(codec, pred, target, clean, valid)[0] == 0
    valid.zero_()
    assert tactile_flow_loss(codec, pred, target, clean, valid)[0] == 0


def test_clean_future_cannot_leak_when_fully_noised(pair):
    _, touch, _ = pair
    open_gates(touch)
    touch.eval()
    touch.tactile_scheduler.sigmas.fill_(1)
    outputs = []
    hook = touch.register_forward_hook(
        lambda module, args, result: outputs.append(tuple(x.detach().clone() for x in result))
    )
    args = loss_inputs()
    torch.manual_seed(91)
    touch.compute_loss(**args)
    args["future_tactile_maps"] = torch.zeros_like(args["future_tactile_maps"])
    torch.manual_seed(91)
    touch.compute_loss(**args)
    hook.remove()
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_strict_checkpoint_roundtrip_and_migration_validation(pair, tmp_path):
    _, touch, alpha = pair
    open_gates(touch)
    path = tmp_path / "v2.safetensors"
    touch.save_checkpoint(str(path))
    other = copy.deepcopy(touch)
    other.load_checkpoint(str(path))
    for a, b in zip(touch(**inputs()), other(**inputs())):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="metadata"):
        touch.load_checkpoint(str(alpha))
    state = load_file(str(alpha))
    state.pop("action_backbone.action_encoder.weight")
    save_file(state, str(tmp_path / "bad.safetensors"))
    with pytest.raises(RuntimeError, match="key mismatch"):
        touch.load_openwam_checkpoint(str(tmp_path / "bad.safetensors"))
    other.tactile_spec["future_stride"] = 2
    with pytest.raises(RuntimeError, match="metadata"):
        other.load_checkpoint(str(path))


@pytest.mark.parametrize("cfg_merge", [False, True])
def test_joint_generation_encode_once_cfg_and_no_future_inputs(pair, cfg_merge):
    _, touch, _ = pair
    open_gates(touch)
    args = inputs(1)
    calls = []
    hook = touch.touch_v2.history_encoder.register_forward_hook(lambda *a: calls.append(1))
    result = touch.generate(
        schedule=[(1000, 1000), (500, 1000), (0, 500), (0, 0)],
        prompt="pick",
        tactile_maps=args["tactile_maps"],
        tactile_valid=args["tactile_valid"],
        proprio=args["proprio"],
        action_num_frames=5,
        decode_video=False,
        cfg_scale=2.0,
        cfg_merge=cfg_merge,
    )
    hook.remove()
    assert len(calls) == 1
    assert result["actions"].shape == (4, 7)
    assert result["future_tactile_maps"].shape == (2, 10, 2, 8, 8)
    assert result["future_tactile_valid"].all()
    assert result["future_tactile_offsets"].tolist() == [1, 2]
    assert torch.isfinite(torch.from_numpy(result["actions"])).all()
    result = touch.generate(
        schedule=[(1000, 1000), (0, 0)], prompt="pick", proprio=args["proprio"], action_num_frames=5, decode_video=False
    )
    assert not result["future_tactile_valid"].any() and not result["future_tactile_maps"].any()


def test_cfg_merged_equals_sequential_and_masks_retained(pair):
    from openwam.model.architectures.touchwam.inference import _predict

    _, touch, _ = pair
    touch.eval()
    open_gates(touch)
    args = inputs()
    shared = {k: args[k] for k in ("latents", "context", "context_mask", "proprio", "tactile_future_valid")}
    shared["uncond_context"] = torch.zeros_like(shared["context"])
    shared["touch_condition"] = touch._history(args["tactile_maps"], args["tactile_valid"], 2)
    results = [
        _predict(
            touch,
            args["noisy_actions"],
            args["action_timestep"],
            args["noisy_tactile"],
            shared,
            args["action_timestep"],
            2.0,
            merge,
        )
        for merge in (False, True)
    ]
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_hydra_and_extra_loss_logging():
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.training_utils import reduce_step_metrics

    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train_touchwam")
        stage2 = compose(
            config_name="train_touchwam",
            overrides=[
                "model.architecture.training_mode=action_finetune",
                "training.initialize_touch_from_openwam=false",
            ],
        )
    assert resolve_architecture_config(cfg.model).registry_name == "touchwam_v2"
    assert cfg.dataloader.tactile_future_frames == 8 and cfg.dataloader.tactile_future_stride == 2
    assert stage2.model.architecture.training_mode == "action_finetune"
    assert not stage2.training.initialize_touch_from_openwam
    assert cfg.model.architecture.training_mode == "joint_finetune"

    trainer = object.__new__(OpenWAMTrainer)
    trainer.lambda_video = trainer.lambda_action = 1
    trainer.architecture = SimpleNamespace(
        prepare_inputs=lambda batch: {"actions": 1},
        compute_loss=lambda **kw: {
            "loss": torch.tensor(4.0),
            "loss_tactile": torch.tensor(2.0),
            "loss_tactile_contact": torch.tensor(1.2),
        },
    )
    losses = trainer.compute_loss([{}])
    metrics = reduce_step_metrics(None, losses, torch.tensor(0.0))
    assert metrics["loss_tactile"] == 2.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("variant", ["touchwam_v2", "touchwam_regions"])
def test_real_wan_ti2v_and_action_forward_backward(monkeypatch, dtype, variant):
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan21

    def init_video(self, cfg):
        # 24 head dimensions split into three even RoPE axis widths.
        dit = WanModel(
            dim=96,
            in_dim=4,
            out_dim=4,
            ffn_dim=192,
            text_dim=32,
            freq_dim=32,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=4,
            num_layers=2,
            has_image_input=False,
            seperated_timestep=True,
            fuse_vae_embedding_in_latents=True,
        )
        self.video_backbone = Wan21(SimpleNamespace(dit=dit, scheduler=ActionScheduler()))

    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", init_video)
    cfg = dict(
        action_dim=7,
        state_dim=7,
        use_proprioception=True,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        attn_head_dim=24,
        text_dim=32,
        bridge_layers=[0, 1],
        attention_mask_mode="mutual",
        mot_checkpoint_mixed_attn=False,
        tactile=dict(
            width=16,
            heads=4,
            layers=2,
            history_layers=1,
            map_size=8,
            patch_size=4,
            history=2,
            future_frames=2,
            future_stride=1,
        ),
    )
    model = build_architecture(variant, cfg)
    model.set_dtype_device(dtype, torch.device("cpu"))
    model.init_training_schedulers()
    if variant == "touchwam_regions":
        from tests.test_touchwam_regions import regional_loss_inputs

        args = regional_loss_inputs()
    else:
        args = loss_inputs()
    args["input_latents"] = torch.randn(2, 4, 3, 4, 4)
    args["first_frame_latents"] = args["input_latents"][:, :, :1].clone()
    args["fuse_vae_embedding_in_latents"] = True
    args = {
        key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in args.items()
    }
    loss = model.compute_loss(**args, use_gradient_checkpointing=True)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.touch_v2.input_projection[0].weight.grad.abs().sum() > 0
    assert model.touch_v2.blocks[0].action_gate.grad.abs() > 0
