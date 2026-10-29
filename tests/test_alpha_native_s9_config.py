"""The 8-node alpha_native recipe and its s9 speed-test variant compose as intended."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
S9_OPTIONS = {
    "deepspeed_zero", "text_padding", "text_dedup", "video_uint8_in_workers", "log_flush_every",
    "vae_cudnn_benchmark", "vae_channels_last", "frame_time_modulation", "gradient_checkpointing_layers",
    "dit_block_fusion", "vae_fused_norm", "overlap_param_allgather", "loss_log_csv", "step_timing", "output_path",
}


def _compose(name):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return OmegaConf.to_container(compose(config_name=name), resolve=False)


def test_alpha_native_recipe_matches_the_formal_run():
    cfg = _compose("train_alpha_native_64gpu_aug")
    t, data = cfg["training"], cfg["dataloader"]
    assert (t["batch_size"], t["gradient_accumulation_steps"], t["num_epochs"], t["max_steps"]) == (48, 1, 1, None)
    assert t["stop_after_steps"] is None
    assert t["action_backbone_init"] == "pretrained" and t["dataset_num_workers"] == 8
    assert data["type"] == "mixture" and data["weight_strategy"] == "proportional"
    assert data["datasets"]["sharpa"]["type"] == "sharpadex_native_tcp"
    for source in data["datasets"].values():
        assert source["color_jitter"] == {"enabled": True, "brightness": 0.2, "contrast": 0.2,
                                          "saturation": 0.2, "hue": 0.0}


def test_s9_variant_changes_only_acceleration_options():
    base, s9 = _compose("train_alpha_native_64gpu_aug"), _compose("train_alpha_native_64gpu_aug_s9")
    assert s9["dataloader"] == base["dataloader"]
    changed = {k for k in set(base["training"]) | set(s9["training"])
               if base["training"].get(k) != s9["training"].get(k)}
    assert changed <= S9_OPTIONS, changed - S9_OPTIONS
    assert s9["training"]["dit_block_fusion"] and s9["training"]["vae_fused_norm"]
    assert s9["training"]["overlap_param_allgather"]
    assert s9["model"]["architecture"]["mot_split_attention"] is True
    base["model"]["architecture"].pop("mot_split_attention", None)
    s9["model"]["architecture"].pop("mot_split_attention")
    assert s9["model"] == base["model"]


def test_native_sharpadex_reader_is_registered():
    from openwam.dataloader.registry import DATASET_REGISTRY, list_registered_datasets

    list_registered_datasets()
    assert DATASET_REGISTRY["sharpadex_native_tcp"].__name__ == "SharpaDexNativeTCPDataset"
