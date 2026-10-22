"""``finetune_load_prefixes`` / ``finetune_video_dit_path``: warm-start the video
backbone only (from the OpenWAM ckpt or an original Wan DiT) and keep the action
branch at its fresh init. A toy module stands in for the 25 GB architecture."""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from openwam.train.utils.ckpt_model_loader import load_partial_finetune_weights


class _Video(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(4, 4)
        self.vae = nn.Linear(4, 2)


class _Arch(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_backbone = _Video()
        self.action_backbone = nn.Linear(3, 3)
        self.proprio_encoder = nn.Linear(2, 4)


def _ckpt(tmp_path, value=1.0):
    sd = {k: torch.full_like(v, value, dtype=torch.bfloat16) for k, v in _Arch().state_dict().items()}
    path = tmp_path / "checkpoint_step_1.safetensors"
    save_file(sd, str(path))
    return str(path)


def test_video_only_keeps_fresh_action(tmp_path):
    torch.manual_seed(0)
    arch = _Arch()
    fresh = {k: v.clone() for k, v in arch.state_dict().items()}
    load_partial_finetune_weights(arch, _ckpt(tmp_path), load_prefixes=["video_backbone."])
    sd = arch.state_dict()
    assert all((sd[k] == 1).all() for k in sd if k.startswith("video_backbone."))
    assert all(torch.equal(sd[k], fresh[k]) for k in sd if not k.startswith("video_backbone."))


def test_raw_dit_override_cast_to_ckpt_dtype(tmp_path):
    arch = _Arch()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    save_file({"weight": torch.full((4, 4), 2.0)}, str(raw_dir / "m-00001-of-00002.safetensors"))
    save_file({"bias": torch.full((4,), 2.0)}, str(raw_dir / "m-00002-of-00002.safetensors"))
    load_partial_finetune_weights(
        arch, _ckpt(tmp_path), load_prefixes=["video_backbone."], video_dit_path=str(raw_dir)
    )
    sd = arch.state_dict()
    assert (sd["video_backbone.dit.weight"] == 2).all() and (sd["video_backbone.vae.weight"] == 1).all()


def test_raw_dit_key_mismatch_raises(tmp_path):
    raw = tmp_path / "raw.safetensors"
    save_file({"weight": torch.zeros(4, 4)}, str(raw))
    with pytest.raises(ValueError, match="keys differ"):
        load_partial_finetune_weights(_Arch(), _ckpt(tmp_path), load_prefixes=["video_backbone."], video_dit_path=str(raw))


def test_incomplete_prefix_raises(tmp_path):
    path = tmp_path / "partial.safetensors"
    save_file({"video_backbone.dit.weight": torch.zeros(4, 4)}, str(path))
    with pytest.raises(RuntimeError, match="not_loaded"):
        load_partial_finetune_weights(_Arch(), str(path), load_prefixes=["video_backbone."])
