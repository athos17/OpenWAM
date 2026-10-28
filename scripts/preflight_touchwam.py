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


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def validate_sample(sample, mode):
    from openwam.tactile.registry import REGION_NAMES
    if sample.get("training_modality", "video_action") != mode:
        raise ValueError("Sample and dataset supervision modes differ")
    video = sample["video"]
    if not video or any(im.size != video[0].size for im in video):
        raise ValueError("Inconsistent scene video")
    valid = torch.as_tensor(sample["video_mask"], dtype=torch.bool)
    if not valid[0]:
        raise ValueError("Missing observed image prefix")
    supervised = bool(valid[1:].any())
    if not str(sample.get("prompt", "")).strip():
        raise ValueError("Empty language")
    action = mode in ("video_action", "video_action_tactile")
    if not action and any(k in sample for k in ("action", "proprio")):
        raise ValueError("A V/VT sample must not fabricate action or proprio")
    if action:
        for key in ("action", "proprio"):
            value, mask = torch.as_tensor(sample[key]), torch.as_tensor(sample[key + "_mask"]).bool()
            if value.shape[-1] != 80 or not torch.isfinite(value[mask.expand_as(value)]).all():
                raise ValueError(f"Invalid unified {key} labels")
        supervised |= bool(torch.as_tensor(sample["action_mask"]).any())
    summary = {"modality": mode, "video_frames": len(video), "image_size": list(video[0].size),
               "video_observed_frames": int(valid.sum()), "prompt": sample["prompt"].strip()[:120]}
    tactile = mode in ("video_tactile", "video_action_tactile")
    if tactile:
        names = sample["tactile_region_names"]
        if len(names) != len(set(names)) or set(names) - set(REGION_NAMES):
            raise ValueError("Invalid or duplicated canonical tactile regions")
        if sample.get("tactile_codec") != "tactile_intensity_v1":
            raise ValueError("Expected the intensity-v1 data contract")
        for key, validity in (("tactile_maps", "tactile_valid"), ("future_tactile_maps", "future_tactile_valid")):
            values = torch.as_tensor(sample[key])
            mask = torch.as_tensor(sample[validity]).bool()[..., None, None, None].expand_as(values)
            if values.shape[-3:] != (1, 32, 32) or values.shape[1] != len(names):
                raise ValueError("Expected [time, regions, 1, 32, 32] intensity maps")
            observed = values[mask]
            if not torch.isfinite(observed).all() or (observed < 0).any() or (observed > 1).any():
                raise ValueError("Observed tactile intensity outside [0,1]")
        summary.update(regions=names, history_observations=int(sample["tactile_valid"].sum()),
                       future_labels=int(sample["future_tactile_valid"].sum()))
        supervised |= bool(torch.as_tensor(sample["future_tactile_valid"]).any())
    elif "tactile_maps" in sample:
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
              "config": resolved, "sources": {}, "errors": [], "training_launch_allowed": False}
    try:
        t = cfg.training
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
            try:
                entry["publication"] = source_publication_contract(source)
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append(f"{name}: {entry['error']}")
                continue
            if metadata_only:
                continue
            source = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
            source.seed = int(cfg.project.seed or 42)
            dataset = build_dataset(source, split="train")
            mode = getattr(dataset, "training_modality", "video_action")
            leaves = getattr(dataset, "buckets", [dataset])
            # Every source plus first/middle/last selected task is decoded;
            # conversion publishers independently validate all files.
            for i in sorted({0, len(leaves) // 2, len(leaves) - 1}):
                leaf = leaves[i]
                for index in sorted({0, len(leaf) // 2, len(leaf) - 1}):
                    # Acceptance must inspect the requested index, not a
                    # reader's runtime retry of a different, healthy window.
                    read = getattr(leaf, "_getitem_impl", leaf.__getitem__)
                    entry["samples"].append(validate_sample(read(index), getattr(leaf, "training_modality", mode)))
            entry.update(training_modality=mode, windows=len(dataset), buckets=len(leaves),
                         data_contract=getattr(dataset, "training_data_contract", None))
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
