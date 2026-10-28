"""Generation preserves the four training graphs without invented robot state."""

from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tests.test_touchwam import open_gates
from tests.test_touchwam_regions import models as regional_models, regional_inputs


@pytest.fixture
def model(monkeypatch, tmp_path):
    return regional_models.__wrapped__(monkeypatch, tmp_path)[2]


def history():
    args = regional_inputs(n=2, batch=1)
    return {k: v for k, v in args.items() if k in (
        "tactile_maps", "tactile_valid", "tactile_region_ids", "tactile_layout_present")}


@pytest.mark.parametrize("mode", ["video_only", "video_tactile"])
@pytest.mark.parametrize("scale,merge", [(1, False), (2, False), (2, True)])
def test_no_action_generation_never_enters_robot_branches(model, mode, scale, merge):
    kwargs = history() if mode == "video_tactile" else {}
    with patch.object(model.action_backbone, "prepare_state", side_effect=AssertionError("ActionDiT entered")):
        with patch.object(model, "_append_proprio_context_token", side_effect=AssertionError("proprio entered")):
            result = model.generate(
                [(1000, 1000), (500, 1000), (500, 500), (0, 0)], "pick",
                generation_modality=mode, decode_video=False, cfg_scale=scale, cfg_merge=merge, **kwargs,
            )
    assert result["actions"] is None
    assert result["generation_modality"] == mode
    assert result["future_tactile_maps"].shape == (2, 2 if kwargs else 0, 1, 8, 8)
    assert np.isfinite(result["future_tactile_maps"]).all()


def test_vt_cfg_merge_matches_sequential_and_tracks_tactile_time(model):
    open_gates(model)
    model.tactile_scheduler.num_train_timesteps = 2000
    kwargs = history()
    captured = []
    hook = model.register_forward_pre_hook(lambda m, args, kw: captured.append((args, kw)), with_kwargs=True)
    results = [model.generate(
        [(1000, 2000), (500, 2000), (500, 800), (0, 0)], "pick", generation_modality="video_tactile",
        decode_video=False, cfg_scale=2, cfg_merge=merge, **kwargs,
    ) for merge in (False, True)]
    hook.remove()
    np.testing.assert_allclose(results[0]["future_tactile_maps"], results[1]["future_tactile_maps"], rtol=2e-5, atol=2e-6)
    assert [kw["tactile_timestep"][0].item() for _, kw in captured[-3:]] == [2000, 2000, 800]
    assert all(args == (None, None) for args, _ in captured)
    # The first update changes V only: the touch noise passed to the next step is unchanged.
    torch.testing.assert_close(captured[-3][1]["noisy_tactile"], captured[-2][1]["noisy_tactile"], rtol=0, atol=0)


@pytest.mark.parametrize("with_touch", [False, True])
def test_explicit_action_modes_match_existing_api_bitwise(model, with_touch):
    kwargs = history() if with_touch else {}
    kwargs.update(proprio=torch.zeros(1, 7), action_num_frames=5, decode_video=False)
    old = model.generate([(1000, 1000), (500, 500), (0, 0)], "pick", **kwargs)
    new = model.generate([(1000, 1000), (500, 500), (0, 0)], "pick",
                         generation_modality="video_action_tactile" if with_touch else "video_action", **kwargs)
    np.testing.assert_array_equal(old["actions"], new["actions"])
    np.testing.assert_array_equal(old["future_tactile_maps"], new["future_tactile_maps"])


def test_conditional_only_wan_preprocessor_gets_empty_text_cfg_with_its_own_mask(model, monkeypatch):
    # Native Wan preprocess ignores cfg_scale and supplies only context/seq_lens.
    preprocess = model.video_backbone.preprocess_input_for_inference

    def conditional_only(**kwargs):
        shared = preprocess(**kwargs)
        shared.pop("uncond_context")
        shared.pop("context_mask")
        shared["seq_lens"] = torch.tensor([3])
        return shared

    monkeypatch.setattr(model.video_backbone, "preprocess_input_for_inference", conditional_only)
    model.video_backbone._tokenizer = lambda *args, **kwargs: (
        torch.zeros(1, 4, dtype=torch.long), torch.tensor([[1, 0, 0, 0]]))
    monkeypatch.setattr(model.video_backbone.text_encoder, "forward", lambda ids, mask: torch.zeros(1, 4, 32))
    captured = []
    hook = model.register_forward_pre_hook(lambda m, args, kw: captured.append(kw), with_kwargs=True)
    open_gates(model)
    kwargs, cache = history(), {}
    outputs = [model.generate(
        [(1000, 1000), (0, 0)], "pick", generation_modality="video_tactile", decode_video=False,
        cfg_scale=2, cfg_merge=merge, prompt_embed_cache=cache, **kwargs,
    ) for merge in (False, True)]
    hook.remove()
    assert "" in cache
    assert captured[-1]["context_mask"].tolist() == [[True, False, False, False], [True, True, True, False]]
    np.testing.assert_allclose(outputs[0]["future_tactile_maps"], outputs[1]["future_tactile_maps"], rtol=2e-5, atol=2e-6)


def test_real_wan_preprocessing_and_tiny_dit_support_vt_cfg(monkeypatch):
    from openwam.model import build_architecture
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import Wan21
    from tests.test_touchwam import tiny_config

    # Only the expensive learned T5/VAE encoders are replaced. Execute actual
    # Wan preprocessing, empty-text encoding helper, conditioning, noise,
    # real Wan DiT and all tactile bridges through both CFG strategies.
    class TextEncoder(torch.nn.Module):
        def forward(self, ids, mask):
            return ids.float()[..., None].expand(-1, -1, 32).clone()

    class VAE(torch.nn.Module):
        def batch_encode(self, pixels, device):
            return torch.zeros(len(pixels), 4, (pixels.shape[2] - 1) // 4 + 1,
                               pixels.shape[3] // 4, pixels.shape[4] // 4, device=device)

    prompts = []

    def tokenizer(texts, **kwargs):
        prompts.extend(texts)
        active = bool(texts[0])
        return torch.full((1, 4), int(active), dtype=torch.long), torch.tensor([[1, int(active), int(active), 0]])

    def init_video(self, cfg):
        dit = WanModel(dim=96, in_dim=4, out_dim=4, ffn_dim=192, text_dim=32, freq_dim=32,
                       eps=1e-6, patch_size=(1, 2, 2), num_heads=4, num_layers=2,
                       has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True)
        holder = SimpleNamespace(dit=dit, scheduler=ActionScheduler(), text_encoder=TextEncoder(), vae=VAE(),
                                 tokenizer=tokenizer, height_division_factor=16, width_division_factor=16,
                                 time_division_factor=4, time_division_remainder=1,
                                 latent_spec=SimpleNamespace(z_dim=4, spatial_compression=4,
                                                             temporal_compression=4, causal_temporal=True))
        self.video_backbone = Wan21(holder)

    monkeypatch.setattr(BaseWAMArchitecture, "_init_video_backbone", init_video)
    cfg = tiny_config()
    cfg.update(bridge_layers=[0, 1], attn_head_dim=24, video_dim=96)
    cfg["tactile"]["bridge_layers"] = [0, 1]
    model = build_architecture("touchwam_regions", cfg)
    model.set_dtype_device(torch.float32, torch.device("cpu"))
    open_gates(model)
    kwargs = history()
    with patch.object(model.action_backbone, "prepare_state", side_effect=AssertionError("ActionDiT entered")):
        results = [model.generate(
            [(1000, 1000), (0, 0)], "pick", generation_modality="video_tactile",
            first_frame_image=[Image.new("RGB", (16, 16))], height=16, width=16, num_frames=9,
            decode_video=False, cfg_scale=2, cfg_merge=merge, **kwargs,
        ) for merge in (False, True)]
    assert prompts == ["pick", "", "pick", ""]
    assert results[0]["actions"] is None and np.isfinite(results[0]["future_tactile_maps"]).all()
    np.testing.assert_allclose(results[0]["future_tactile_maps"], results[1]["future_tactile_maps"], rtol=2e-5, atol=2e-6)


def test_explicit_generation_contract_rejects_inconsistent_modes(model):
    common = dict(schedule=[(1000, 1000), (0, 0)], prompt="pick", decode_video=False)
    with pytest.raises(ValueError, match="cannot consume proprio"):
        model.generate(**common, generation_modality="video_tactile", proprio=torch.zeros(7), **history())
    with pytest.raises(ValueError, match="nonempty explicit"):
        model.generate(**common, generation_modality="video_tactile")
    with pytest.raises(ValueError, match="cannot consume a tactile"):
        model.generate(**common, generation_modality="video_only", **history())
    with pytest.raises(ValueError, match="Unknown generation"):
        model.generate(**common, generation_modality="tactile_only")


def test_vat_deployment_selects_real_action_normalizer_but_ego_vt_cannot(tmp_path):
    from omegaconf import OmegaConf
    from openwam.dataloader.utils.normalization_package import save_normalization_package
    from openwam.deploy.model_loader import _build_normalizer
    from tests.dataloader.test_mixed_training_contract import package_sources

    sources = package_sources(tmp_path)
    sources["sharpa"]["training_modality"] = "video_action_tactile"
    sources["ego"]["training_modality"] = "video_tactile"
    output = tmp_path / "checkpoint"
    save_normalization_package(output, sources)
    cfg = OmegaConf.create({"dataloader": {"type": "mixture"}})
    normalizer = _build_normalizer(cfg, str(output), normalization_source="sharpa")
    assert np.isfinite(normalizer.normalize(np.zeros((1, 20), np.float32))).all()
    with pytest.raises(ValueError, match="no action supervision"):
        _build_normalizer(cfg, str(output), normalization_source="ego")
