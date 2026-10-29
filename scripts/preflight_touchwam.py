#!/usr/bin/env python3
"""Validate the exact source selection and complete OpenWAM initialization.

No training starts here. A metadata-only result is never a training acceptance.
The launch wrapper invokes the full check with the very same Hydra overrides.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.dataloader.publication import source_publication_contract
from openwam.dataloader.registry import build_dataset
from openwam.tactile.regions import INTENSITY_CODEC, RGB_CODEC


MODALITIES = {
    "ego": {"video_only", "video_tactile"},
    "robot": {"video_action", "video_action_tactile"},
}


def validate_source_contract(name, source):
    """A robot source must never silently become an ego V/VT source."""
    kind = source.get("source_kind")
    expected = source.get("expected_modalities")
    if kind not in MODALITIES:
        raise ValueError("Each source requires source_kind: ego or robot")
    if isinstance(expected, str) or not expected:
        raise ValueError("Each source requires an explicit expected_modalities list")
    expected = list(expected)
    if len(set(expected)) != len(expected) or set(expected) - MODALITIES[kind]:
        raise ValueError(f"{kind} sources allow only {sorted(MODALITIES[kind])}")
    family = str(source.get("source_family", "")).lower()
    path_parts = Path(str(source.get("dataset_dir", ""))).parts
    ftp = family == "ftp1" or name.lower().startswith(("ftp_", "ftp1_")) or "ftp1" in path_parts
    if ftp and kind != "robot":
        raise ValueError("FTP-1 is a robot source and requires measured action supervision")
    if kind == "robot" and source.get("type") in ("lerobot_video_intensity", "egodex_reuse_video"):
        raise ValueError("Robot sources require an action reader; V/VT readers are forbidden")
    representation = source.get("tactile_representation", "intensity")
    if representation not in ("intensity", "rgb", "touchscale_force"):
        raise ValueError("Unknown tactile_representation")
    return {"source_kind": kind, "expected_modalities": expected,
            "expected_tactile_codec": RGB_CODEC if representation == "rgb" else INTENSITY_CODEC}


def validate_training_schedule(training):
    """Match the trainer's checkpoint and optimizer-boundary semantics."""
    for key in ("gradient_accumulation_steps", "batch_size"):
        value = training.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    keep = training.get("keep_last_k_ckpts", 1)
    if type(keep) is not int or (keep != -1 and keep < 1):
        raise ValueError("keep_last_k_ckpts must be -1 (all) or a positive integer")
    for key in ("max_steps", "num_epochs"):
        value = training.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be null or a positive integer")
    if training.get("max_steps") is None and training.get("num_epochs") is None:
        raise ValueError("Training needs num_epochs or max_steps")
    save = training.get("save_steps")
    if save is not None and (type(save) is not int or save < 0):
        raise ValueError("save_steps must be null or a nonnegative integer")
    if type(training.get("save_full_states_for_resume", False)) is not bool:
        raise ValueError("save_full_states_for_resume must be boolean")
    accum = training.get("gradient_accumulation_steps", 1)
    deferred = [key for key in ("save_steps", "max_steps")
                if training.get(key) and training[key] % accum]
    return {"gradient_accumulation_steps": accum, "keep_last_k_ckpts": keep,
            "checkpoint_and_stop_boundary": "optimizer_update_or_epoch_end",
            "deferred_boundary_settings": deferred}


def validate_tactile_sample(sample, spec):
    """Exercise the real collator with the live model contract, without loading weights."""
    from openwam.tactile.regions import collate_regions, observation_batch_shape

    if spec is None:
        raise ValueError("Tactile preflight requires the resolved model tactile specification")
    codec = sample.get("tactile_codec")
    if codec == RGB_CODEC:
        if spec.get("rgb_enabled", False) is not True:
            raise ValueError("RGB tactile data requires the model's rgb_enabled adapter")
        size, channels, meaning = spec.get("rgb_map_size", 128), 3, "optical_rgb"
    elif codec == spec["codec"] == INTENSITY_CODEC:
        size, channels, meaning = spec["map_size"], 1, "normalized_contact_intensity"
    else:
        raise ValueError("Sample tactile codec is not supported by the live model")
    if type(size) is not int or size < 1:
        raise ValueError("Selected tactile map size must be a positive integer")
    for key in ("map_size", "history", "future_frames", "future_stride", "fps"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f"Model tactile {key} must be a positive integer")
    sensors = list(spec["sensor_names"])
    if not sensors or len(set(sensors)) != len(sensors) or any(not isinstance(s, str) or not s for s in sensors):
        raise ValueError("Model sensor_names must be nonempty and unique")
    expected = {
        "tactile_history_times_s": torch.arange(1 - spec["history"], 1, dtype=torch.float64) / spec["fps"],
        "future_tactile_times_s": torch.arange(1, spec["future_frames"] + 1, dtype=torch.float64)
            * spec["future_stride"] / spec["fps"],
    }
    for key, grid in expected.items():
        if key not in sample:
            raise ValueError(f"Tactile samples require explicit {key}")
        times = torch.as_tensor(sample[key], dtype=torch.float64).cpu()
        if times.shape != grid.shape or not torch.allclose(times, grid, atol=1e-6, rtol=0):
            raise ValueError(f"Dataset/model {key} differ; resample to the configured model grid")
    observation = spec.get("observation_encoder") or {}
    observation_kwargs = {}
    if observation.get("kind") == "ftp1_heterogeneous_v1":
        observation_kwargs = dict(
            observation_map_size=observation_batch_shape([sample], observation),
            encoder_names=[item["name"] for item in observation["sensors"]],
        )
    batch = collate_regions(
        [sample], map_size=size, history=spec["history"], future_frames=spec["future_frames"],
        region_names=list(spec["region_names"]), sensor_names=sensors, codec=codec, **observation_kwargs,
    )
    present = batch["tactile_layout_present"]
    spatial = batch["tactile_spatial_valid"]
    query = batch["tactile_query_mask"]
    measured_region = present & spatial.flatten(2).any(-1)
    if (query & ~measured_region[:, None]).any():
        raise ValueError("Tactile queries require a present region with a nonempty spatial ROI")
    counts = {}
    for key, validity in (("tactile_maps", "tactile_valid"), ("future_tactile_maps", "future_tactile_valid")):
        valid = batch[validity]
        if (valid & ~measured_region[:, None]).any():
            raise ValueError(f"{validity} labels a region absent from the layout or spatial ROI")
        pixels = valid[..., None, None, None] & spatial[:, None]
        observed = batch[key][pixels.expand_as(batch[key])]
        if not torch.isfinite(observed).all() or (observed < 0).any() or (observed > 1).any():
            raise ValueError(f"Observed tactile {meaning} values outside [0,1]")
        counts[validity] = int(valid.sum())
    if observation_kwargs:
        obs = batch["tactile_observation_maps"]
        obs_roi = batch["tactile_observation_spatial_valid"]
        measured = batch["tactile_valid"]
        if (measured & ~(present & obs_roi.flatten(2).any(-1))[:, None]).any():
            raise ValueError("Observed history requires a present region with nonempty observation ROI")
        pixels = measured[..., None, None, None] & obs_roi[:, None]
        values = obs[pixels.expand_as(obs)]
        if not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Observed heterogeneous tactile values must be finite and in [0,1]")
        source_mapping = observation.get("source_to_encoder", {})
        encoder_specs = list(observation["sensors"])
        for source_id, encoder_id in zip(batch["tactile_sensor_ids"][0], batch["tactile_encoder_ids"][0]):
            source = sensors[int(source_id)]
            encoder = encoder_specs[int(encoder_id)]
            allowed = source_mapping.get(source, ())
            allowed = [allowed] if isinstance(allowed, str) else list(allowed)
            if encoder["name"] not in allowed:
                raise ValueError(f"Observation source {source} is not mapped to encoder {encoder['name']}")
            representation = encoder.get("input_representation", "pressure" if encoder["kind"] == "matrix" else None)
            if (codec == RGB_CODEC) != (representation == "rgb"):
                raise ValueError("Observation encoder input representation differs from the tactile codec")
    for key, validity, grid_key in (
        ("tactile_history_source_times_s", "tactile_valid", "tactile_history_times_s"),
        ("future_tactile_source_times_s", "future_tactile_valid", "future_tactile_times_s"),
    ):
        if key not in sample:
            continue
        times = torch.as_tensor(sample[key], dtype=torch.float64).cpu()
        grid = expected[grid_key].cpu()
        if times.ndim == 2:
            # Independent native glove clocks: do not collapse them to one time.
            measured = torch.as_tensor(sample[validity]).cpu()
            if times.shape != measured.shape or times.shape[0] != len(grid):
                raise ValueError(f"Invalid measured {key}")
            grid = grid[:, None].expand_as(times)
        else:
            measured = batch[validity][0].any(-1).cpu()
        if times.shape != grid.shape or not torch.isfinite(times[measured]).all():
            raise ValueError(f"Invalid measured {key}")
        if grid_key == "tactile_history_times_s":
            if (times[measured] > grid[measured] + 1e-6).any():
                raise ValueError("Tactile history must not borrow future measurements")
        elif (times[measured] <= 0).any():
            raise ValueError("Future tactile supervision requires strictly future measurements")
    # Availability of a future label never determines the prediction query.
    effective = batch["future_tactile_valid"] & query
    return {"regions": list(sample["tactile_region_names"]),
            "tactile_codec": codec, "tactile_representation": meaning,
            "channels": channels, "map_size": size,
            "history_observations": counts["tactile_valid"],
            "future_labels": counts["future_tactile_valid"],
            "supervised_future_regions": int(effective.sum()),
            "history_shape": list(batch["tactile_maps"].shape),
            "future_shape": list(batch["future_tactile_maps"].shape),
            "collator_validated": True,
            **({"observation_shape": list(batch["tactile_observation_maps"].shape),
                "observation_encoder_names": list(observation_kwargs["encoder_names"])}
               if observation_kwargs else {})}, bool(effective.any())


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def validate_sample(sample, mode, *, tactile_spec=None, source_kind=None, expected_modalities=None,
                    expected_tactile_codec=None):
    if mode not in MODALITIES["ego"] | MODALITIES["robot"]:
        raise ValueError(f"Unknown training modality: {mode}")
    if source_kind is not None and mode not in MODALITIES[source_kind]:
        raise ValueError(f"{source_kind} source unexpectedly emits {mode}")
    if expected_modalities is not None and mode not in expected_modalities:
        raise ValueError(f"Source unexpectedly emits {mode}; expected {list(expected_modalities)}")
    if sample.get("training_modality", "video_action") != mode:
        raise ValueError("Sample and dataset supervision modes differ")
    video = sample["video"]
    if not video or any(im.size != video[0].size for im in video):
        raise ValueError("Inconsistent scene video")
    valid = torch.as_tensor(sample["video_mask"])
    if valid.dtype != torch.bool or valid.shape != (len(video),):
        raise ValueError("video_mask must be bool [video_frames]")
    if not valid[0]:
        raise ValueError("Missing observed image prefix")
    supervised = bool(valid[1:].any())
    if not str(sample.get("prompt", "")).strip():
        raise ValueError("Empty language")
    action = mode in ("video_action", "video_action_tactile")
    if not action and any(k in sample for k in ("action", "proprio")):
        raise ValueError("A V/VT sample must not fabricate action or proprio")
    if not action and any(torch.as_tensor(sample[k]).any() for k in ("action_mask", "proprio_mask") if k in sample):
        raise ValueError("A V/VT sample cannot contain valid action or proprio labels")
    if action:
        for key in ("action", "proprio"):
            if key not in sample or key + "_mask" not in sample:
                raise ValueError(f"Robot samples require measured {key} and {key}_mask")
            value, mask = torch.as_tensor(sample[key]), torch.as_tensor(sample[key + "_mask"])
            if (value.ndim != 2 or value.shape[-1] != 80 or not value.is_floating_point()
                    or mask.dtype != torch.bool or mask.shape != value.shape
                    or not torch.isfinite(value[mask]).all()):
                raise ValueError(f"Invalid unified {key} labels")
            if key == "proprio" and value.shape[0] != 1:
                raise ValueError("Robot proprio must contain exactly one observed state")
            if not mask.any():
                raise ValueError(f"Robot sample has no future action supervision or observed {key}; zero placeholders are forbidden")
        supervised |= bool(torch.as_tensor(sample["action_mask"]).any())
    summary = {"modality": mode, "video_frames": len(video), "image_size": list(video[0].size),
               "video_observed_frames": int(valid.sum()), "prompt": sample["prompt"].strip()[:120]}
    tactile = mode in ("video_tactile", "video_action_tactile")
    if tactile:
        if not sample.get("tactile_region_names"):
            raise ValueError("Tactile modalities require an explicit nonempty regional layout")
        if expected_tactile_codec is not None and sample.get("tactile_codec") != expected_tactile_codec:
            raise ValueError("Reader tactile codec differs from the selected source representation")
        touch, touch_supervised = validate_tactile_sample(sample, tactile_spec)
        summary.update(touch)
        supervised |= touch_supervised
    elif sample.get("tactile_region_names") or any(k in sample for k in ("tactile_maps", "future_tactile_maps")):
        raise ValueError("V/VA publication unexpectedly emits tactile labels")
    if not supervised:
        raise ValueError("Sample has no future video/action/tactile supervision")
    return summary


def run(config, overrides, output, *, metadata_only=False, alpha_audit=None):
    started = time.time()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name=config, overrides=overrides)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    report = {"schema": "touchwam_pretrain_readiness_v1", "status": "checking", "config_sha256": digest,
              "config": resolved, "sources": {}, "errors": [], "training_launch_allowed": False,
              "validation_scope": "source_sample_collation_and_initialization_contracts",
              "full_training_loop_verified": False}
    try:
        t = cfg.training
        report["training_schedule"] = validate_training_schedule(t)
        if int(t.get("zero_stage", 2)) != 2:
            raise ValueError("This regional pretraining recipe is validated with ZeRO stage 2")
        if str(t.get("offload_optimizer_device", "none")) != "none":
            raise ValueError("Regional modality-aware AdamW requires offload_optimizer_device=none")
        if t.action_backbone_init != "pretrained":
            raise ValueError("This pretraining recipe must preserve complete OpenWAM action weights")
        if cfg.model.architecture.variant != "touchwam_regions":
            raise ValueError("Expected the regional TouchWAM architecture")
        if cfg.model.architecture.tactile.codec != "tactile_intensity_v1":
            raise ValueError("Expected the normalized intensity representation")
        if str(cfg.model.architecture.training_mode) == "new_only":
            raise ValueError("Mixed V/VA/VT/VAT pretraining requires joint_finetune; use a tactile-only recipe for new_only")
        if cfg.dataloader.type != "mixture":
            raise ValueError("Expected an explicit source mixture")
        for name, source in cfg.dataloader.datasets.items():
            if not source.get("enabled", True):
                continue
            if name.lower() in ("ftp_sharpa", "ftp1_sharpa") or "/ftp1/sharpa" in str(source.dataset_dir).lower():
                raise ValueError("FTP-1 Sharpa is excluded from this publication")
            # Always inspect the complete published population before applying
            # user-selected tasks or hours. An incomplete source cannot shrink.
            entry = {"samples": []}
            report["sources"][name] = entry
            dataset, leaves = None, None
            try:
                contract = validate_source_contract(name, source)
                entry.update(contract)
                if (contract["expected_tactile_codec"] == RGB_CODEC
                        and cfg.model.architecture.tactile.get("rgb_enabled", False) is not True):
                    raise ValueError("RGB source selected but the model RGB adapter is disabled")
                entry["publication"] = source_publication_contract(source)
                if metadata_only:
                    continue
                source = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
                source.seed = int(cfg.project.seed if cfg.project.seed is not None else 42)
                dataset = build_dataset(source, split="train")
                mode = getattr(dataset, "training_modality", "video_action")
                leaves = getattr(dataset, "buckets", [dataset])
                if not len(dataset) or not leaves:
                    raise ValueError("Selected source has no training windows")
                # Inspect every bucket's declared modality, even when only a
                # subset of buckets is decoded. A hidden V/VT robot bucket
                # must not enter the mixture through publication sampling.
                for leaf in leaves:
                    leaf_mode = getattr(leaf, "training_modality", mode)
                    if leaf_mode not in contract["expected_modalities"]:
                        raise ValueError(f"Bucket emits {leaf_mode}; expected {contract['expected_modalities']}")
                    if len(leaf) == 0:
                        raise ValueError("Selected source contains an empty bucket")
                # Every source plus first/middle/last selected task is decoded;
                # conversion publishers independently validate all files.
                for i in sorted({0, len(leaves) // 2, len(leaves) - 1}):
                    leaf = leaves[i]
                    for index in sorted({0, len(leaf) // 2, len(leaf) - 1}):
                        # Inspect the requested index, not a reader retry of a
                        # different, healthy window.
                        read = getattr(leaf, "_getitem_impl", leaf.__getitem__)
                        entry["samples"].append(validate_sample(
                            read(index), getattr(leaf, "training_modality", mode),
                            tactile_spec=cfg.model.architecture.tactile, **contract,
                        ))
                entry.update(training_modality=mode, windows=len(dataset), buckets=len(leaves),
                             data_contract=getattr(dataset, "training_data_contract", None))
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append(f"{name}: {entry['error']}")
            finally:
                del dataset, leaves
                gc.collect()
        if not report["sources"]:
            raise ValueError("No enabled data sources")
        if report["errors"]:
            raise ValueError("One or more selected data publications are not ready; see source errors in the report")
        if not metadata_only and not t.get("resume_ckpt_path"):
            if not t.get("initialize_touch_from_openwam") or not t.get("finetune_ckpt_path"):
                raise ValueError("Full OpenWAM initialization must be explicitly enabled")
            from scripts.validation.touchwam_alpha_init import run as audit_alpha
            checkpoint = Path(str(t.finetune_ckpt_path))
            header = audit_alpha(checkpoint, Path(output).with_suffix(".alpha_header.json"),
                                 config=config, header_only=True, live_cfg=cfg)
            report["alpha_header"] = header
            if not alpha_audit:
                raise ValueError("Provide --alpha-audit from the full per-tensor OpenWAM verification")
            proof = json.loads(Path(alpha_audit).read_text())
            for key in ("checkpoint_stat", "checkpoint_header_sha256", "resolved_model_config_sha256"):
                if proof.get(key) is None or proof[key] != header.get(key):
                    raise ValueError(f"Full OpenWAM audit is stale or incompatible: {key}")
            if proof.get("status") != "passed" or not proof.get("all_pretrained_bits_preserved"):
                raise ValueError("Complete pretrained tensor preservation has not passed")
            report["alpha_full_audit"] = {"path": str(alpha_audit), "sha256": hashlib.sha256(Path(alpha_audit).read_bytes()).hexdigest(),
                                         "checkpoint_sha256": proof["checkpoint_sha256"]}
        if t.get("resume_ckpt_path") and any(t.get(k, False) for k in (
                "initialize_touch_from_openwam", "initialize_touch_from_legacy", "initialize_touch_registry")):
            raise ValueError("Resume must disable all fresh initialization flags")
        report["status"] = "metadata_validated" if metadata_only else "passed"
        report["training_launch_allowed"] = not metadata_only
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.time() - started
        atomic_json(output, report)
    print(json.dumps({"status": report["status"], "sources": len(report["sources"]), "report": str(output)}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train_touchwam_pretrain")
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha-audit")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    run(args.config, args.overrides, args.output, metadata_only=args.metadata_only, alpha_audit=args.alpha_audit)
