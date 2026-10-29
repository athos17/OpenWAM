"""S9 must preserve TouchWAM's four graphs, codecs, gradients and clean prefix."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.model import build_architecture, resolve_architecture_config
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.compile_options import eager_compiled_regions
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan21
from openwam.tactile.regions import INTENSITY_CODEC
from openwam.train.openwam_trainer import OpenWAMTrainer
from tests.test_touchwam import open_gates, tiny_config
from tests.test_touchwam_multimodal import modality_inputs
from tests.test_touchwam_regions import regional_inputs
from tests.test_touchwam_rgb import rgb_inputs


def small_model(monkeypatch, *, device="cpu", dtype=torch.float32):
    def init_video(self, cfg):
        dit = WanModel(dim=96, in_dim=4, out_dim=4, ffn_dim=192, text_dim=32, freq_dim=32,
                       eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
                       has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True)
        self.video_backbone = Wan21(SimpleNamespace(dit=dit, scheduler=ActionScheduler()))
    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", init_video)
    cfg = tiny_config()
    cfg.update(bridge_layers=[0, 1], attn_head_dim=24, video_dim=96, training_mode="joint_finetune")
    cfg["tactile"].update(bridge_layers=[0, 1], codec=INTENSITY_CODEC,
                          rgb_enabled=True, rgb_map_size=16, rgb_patch_size=8)
    torch.manual_seed(42)
    model = build_architecture("touchwam_regions", cfg)
    model.set_dtype_device(dtype, torch.device(device))
    model.init_training_schedulers()
    open_gates(model)
    return model


def loss_args(modality, *, rgb=False, device="cpu", dtype=torch.float32):
    torch.manual_seed(17)
    args = rgb_inputs(modality) if rgb else modality_inputs(modality)
    args["input_latents"] = torch.randn(2, 4, 3, 4, 4)
    args["first_frame_latents"] = args["input_latents"][:, :, :1].clone()
    args["fuse_vae_embedding_in_latents"] = True
    return {k: (v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                if isinstance(v, torch.Tensor) else v) for k, v in args.items()}


def enable_s9(model):
    model.video_backbone.set_frame_time_modulation(True)
    model.video_backbone.set_block_fusion(True)
    model.mot_driver.mot_split_attention = True
    model.set_training_runtime(use_gradient_checkpointing=True, gradient_checkpointing_layers=[0])


def forward_backward(model, args):
    torch.manual_seed(66)
    model.zero_grad(set_to_none=True)
    losses = model.compute_loss(**args, use_gradient_checkpointing=True)
    losses["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return losses["loss"].detach().clone(), {n: p.grad.detach().clone() for n, p in model.named_parameters()
                                          if p.grad is not None}


CASES = [(m, False) for m in ("video_only", "video_action", "video_tactile", "video_action_tactile")]
CASES += [(m, True) for m in ("video_tactile", "video_action_tactile")]


@pytest.mark.parametrize("modality,rgb", CASES)
def test_s9_four_graphs_and_codecs_preserve_loss_and_gradients(monkeypatch, modality, rgb):
    model = small_model(monkeypatch)
    args = loss_args(modality, rgb=rgb)
    baseline, before = forward_backward(model, args)
    enable_s9(model)
    with patch.object(model.mot_driver, "_step_checkpointed", wraps=model.mot_driver._step_checkpointed) as gc:
        with eager_compiled_regions():  # exact fusion formulas here; actual CUDA compile tested below
            accelerated, after = forward_backward(model, args)
    torch.testing.assert_close(accelerated, baseline, rtol=2e-5, atol=2e-6)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=3e-4, atol=2e-5, msg=name)
    if "action" in modality:
        assert [call.args[0] for call in gc.call_args_list] == [0]
    if modality == "video_only":
        # Pure V already uses native unmasked attention. Importing the VA mask
        # here would change the baseline attention graph, not just its kernel.
        assert not model.mot_driver._split_plans
    else:
        assert model.mot_driver._split_plans and all(model.mot_driver._split_plans.values())
    assert any("action_backbone" in n for n in after) == ("action" in modality)
    assert any("touch_v2." in n for n in after) == ("tactile" in modality)


@pytest.mark.parametrize("clean_frames", [1, 2])
def test_split_vt_keeps_all_clean_prefix_frames_isolated(monkeypatch, clean_frames):
    model = small_model(monkeypatch).eval()
    enable_s9(model)
    args = regional_inputs(2)
    args.update(noisy_actions=None, action_timestep=None, proprio=None, _strict_video_only=True,
                latents=torch.randn(2, 4, 3, 4, 4), timestep=torch.full((2,), 500.0),
                fuse_vae_embedding_in_latents=True, num_clean_prefix_frames=clean_frames)
    args["first_frame_latents"] = args["latents"][:, :, :clean_frames].clone()
    with torch.no_grad(), eager_compiled_regions():
        before = model(**args)[0]
        after = model(**dict(args, noisy_tactile=args["noisy_tactile"] + 100))[0]
    torch.testing.assert_close(before[:, :, :clean_frames], after[:, :, :clean_frames], rtol=0, atol=0)
    assert not torch.equal(before[:, :, clean_frames:], after[:, :, clean_frames:])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for actual compiled S9 kernels")
@pytest.mark.parametrize("modality", ["video_tactile", "video_action_tactile"])
def test_cuda_s9_compiled_backward_and_optimizer_update(monkeypatch, modality):
    from openwam.model.video_backbone.wan import fused_block
    import torch._dynamo.utils as dynamo_utils

    model = small_model(monkeypatch, device="cuda", dtype=torch.bfloat16)
    args = loss_args(modality, rgb=True, device="cuda", dtype=torch.bfloat16)
    baseline, before = forward_backward(model, args)
    torch._dynamo.reset()
    fused_block._compiled.clear()
    enable_s9(model)
    graphs_before = dynamo_utils.counters["stats"]["unique_graphs"]
    accelerated, after = forward_backward(model, args)
    assert dynamo_utils.counters["stats"]["unique_graphs"] > graphs_before
    assert before.keys() == after.keys()
    # BF16 fused reductions/GEMM shape changes round differently. Check overall
    # gradient direction as well as loss, and perform a real optimizer update.
    torch.testing.assert_close(accelerated, baseline, rtol=0.03, atol=0.01)
    a = torch.cat([before[n].float().flatten() for n in before])
    b = torch.cat([after[n].float().flatten() for n in before])
    assert torch.nn.functional.cosine_similarity(a, b, dim=0) > 0.995
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    weight = model.touch_v2.rgb_input_projection[0].weight if hasattr(model.touch_v2, "rgb_input_projection") else next(
        p for n, p in model.named_parameters() if "touch_v2." in n and p.grad is not None and p.grad.abs().sum() > 0)
    saved = weight.detach().clone()
    optimizer.step()
    assert not torch.equal(weight, saved)
    loss, _ = forward_backward(model, args)
    assert torch.isfinite(loss)


def test_s9_recipe_preserves_data_full_initialization_and_schedule():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        base = compose(config_name="train_touchwam_six_source_20261029")
        fast = compose(config_name="train_touchwam_six_source_20261029_s9")
    assert OmegaConf.to_container(base.dataloader, resolve=True) == OmegaConf.to_container(fast.dataloader, resolve=True)
    for key in ("finetune_ckpt_path", "initialize_touch_from_openwam", "action_backbone_init",
                "batch_size", "num_epochs", "max_steps", "learning_rate", "gradient_accumulation_steps",
                "save_full_states_for_resume", "keep_last_k_ckpts"):
        assert base.training[key] == fast.training[key]
    assert fast.model.architecture.tactile == base.model.architecture.tactile
    resolved = resolve_architecture_config(fast.model)
    assert resolved.registry_name == "touchwam_regions" and resolved.params["mot_split_attention"]
    assert fast.training.overlap_param_allgather and fast.training.dit_block_fusion


def test_buffered_logs_keep_one_schema_across_modalities(tmp_path):
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({"training": {"loss_log_csv": True, "log_flush_every": 3}})
    trainer.architecture = SimpleNamespace(model_version="touchwam_regions_v1")
    trainer.accelerator = None
    for step, extra in enumerate(({}, {"loss_tactile_intensity": 0.2}, {"loss_tactile_rgb": 0.3}, {}), 1):
        metrics = dict(loss_total=1., loss_video=0.5, loss_action=0.5, grad_norm=0.1, **extra)
        trainer.log_step(metrics=metrics, global_step=step, opt_step=step, epoch=0, lr=1e-4,
                         steps_per_sec=1., batch_size=1, pbar=None, wandb_run=None, debug=False,
                         output_path=str(tmp_path))
    trainer._flush_logs()
    rows = list(csv.DictReader((tmp_path / "debug_loss_history.csv").open()))
    assert len(rows) == 4 and rows[1]["loss_tactile_intensity"] == "0.2" and rows[2]["loss_tactile_rgb"] == "0.3"
    records = [json.loads(line) for line in (tmp_path / "training_metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2, 3, 4]


@pytest.mark.parametrize("overlap", [False, True])
def test_parameter_wait_precedes_trainable_tactile_encoding(monkeypatch, overlap):
    from openwam.train.utils import deferred_allgather

    events = []
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer._overlap_frozen_inputs = overlap
    trainer.lambda_video = trainer.lambda_action = 1.0
    trainer.accelerator = None
    def prepare(batch):
        events.append("frozen_inputs")
        return {}
    def loss(**kw):
        events.append("trainable_touch_video_action")
        return {"loss": torch.tensor(1.)}
    trainer.architecture = SimpleNamespace(prepare_inputs=prepare, compute_loss=loss)
    monkeypatch.setattr(deferred_allgather, "wait", lambda: events.append("wait"))
    trainer.compute_loss([{"training_modality": "video_only"}])
    assert events[-2:] == ["wait", "trainable_touch_video_action"]
    assert events[0] == ("frozen_inputs" if overlap else "wait")


def test_training_failure_restores_deepspeed_patch_and_flushes_logs(monkeypatch):
    from openwam.train.utils import deferred_allgather

    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer._owns_deferred_allgather = True
    calls = []
    trainer._flush_logs = lambda: calls.append("flush")
    def fail(**kw):
        raise RuntimeError("intentional")
    trainer._train = fail
    monkeypatch.setattr(deferred_allgather, "uninstall", lambda: calls.append("uninstall"))
    with pytest.raises(RuntimeError, match="intentional"):
        trainer.train()
    assert calls == ["flush", "uninstall"] and not trainer._owns_deferred_allgather
