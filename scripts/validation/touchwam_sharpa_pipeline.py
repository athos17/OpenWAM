#!/usr/bin/env python3
"""Bounded real Sharpa reader → production preparation → tiny real MoT backward.

This does not load Alpha, run production VAE/T5, or perform optimizer updates.
The image/text encoder substitutes are deterministic functions of real inputs.
Publication bypass is explicit, restricted to one cataloged completed bucket,
and never changes the production training configuration or publication gates.
"""

import argparse
import gc
import hashlib
import json
import random
import resource
import subprocess
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from openwam.dataloader.registry import build_dataset
from openwam.model.architectures.utils.common import downsample_video_mask_to_latent
from openwam.tactile.registry import FINGERTIP_NAMES, REGION_NAMES
from scripts.validation.touchwam_multisource_gpu import tiny_model


ACTION_SLOTS = [*range(0, 9), *range(10, 32), *range(34, 43), *range(44, 66)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(value):
    value = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def bounded_encoders(self, *, frames, text, vace_videos, ref_images):
    """Explicit VAE/T5 substitutes; preparation/collation stay unmodified."""
    latents, contexts = [], []
    for images, prompt in zip(frames, text):
        require(len(images) == 9, "Expected nine production decoded scene frames")
        require(isinstance(prompt, str) and prompt.strip(), "Language must be nonempty")
        rgb = torch.stack([torch.from_numpy(np.array(im.convert("RGB").resize((4, 4)), copy=True))
                           .permute(2, 0, 1).float() / 255 for im in images])
        temporal = torch.stack([rgb[0], rgb[1:5].mean(0), rgb[5:9].mean(0)])
        channels = torch.cat([temporal, temporal.mean(1, keepdim=True)], dim=1)
        latents.append(channels.permute(1, 0, 2, 3) * 2 - 1)
        # This tests the real language plumbing, not semantic language encoding.
        code = hashlib.shake_256(prompt.encode("utf8")).digest(128)
        contexts.append(torch.tensor(list(code)).reshape(4, 32).float() / 127.5 - 1)
    encoded = torch.stack(latents).to(device=self.device, dtype=self.dtype)
    return dict(input_latents=encoded, first_frame_latents=encoded[:, :, :1].clone(),
                context=torch.stack(contexts).to(device=self.device, dtype=self.dtype),
                context_mask=torch.ones(len(frames), 4, dtype=torch.bool, device=self.device),
                fuse_vae_embedding_in_latents=True)


def sample_receipt(sample, *, index, position, width=320, height=384):
    require(sample["training_modality"] == "video_action_tactile", "Sharpa must provide VAT")
    names = list(sample["tactile_region_names"])
    require(len(names) == 10 and set(names) == set(FINGERTIP_NAMES), "Ten-finger identity mismatch")
    require(list(sample["tactile_sensor_names"]) == ["sharpa_real"] * 10, "Sharpa sensor identity mismatch")
    require(len(sample["video"]) == 9, "Wrong scene video length")
    require(all(image.size == (width, height) for image in sample["video"]), "Wrong scene frame size")
    require(isinstance(sample["prompt"], str) and sample["prompt"].strip(), "Missing language")
    action, mask = torch.as_tensor(sample["action"]), torch.as_tensor(sample["action_mask"])
    require(action.shape == mask.shape == (32, 80), "Action is not the 32 × 80 production layout")
    unused = sorted(set(range(80)) - set(ACTION_SLOTS))
    require(not mask[:, unused].any(), "One of the 18 unused action slots is supervised")
    require(mask.any() and torch.isfinite(action[mask]).all(), "Missing/nonfinite valid native actions")
    require(not torch.as_tensor(sample["proprio_mask"])[..., unused].any(), "Unused state slots are valid")
    for key, shape in (("tactile_maps", (4, 10, 1, 32, 32)),
                       ("future_tactile_maps", (8, 10, 1, 32, 32))):
        values = torch.as_tensor(sample[key])
        require(values.shape == shape, f"Wrong {key} shape")
        require(torch.isfinite(values).all() and (values >= 0).all() and (values <= 1).all(),
                f"Invalid normalized {key}")
    history = torch.as_tensor(sample["tactile_history_source_times_s"])
    future = torch.as_tensor(sample["future_tactile_source_times_s"])
    require((history[torch.isfinite(history)] <= 1e-8).all(), "Noncausal history")
    require(torch.isfinite(history[-1]) and abs(float(history[-1])) <= 1e-8,
            "Current tactile frame was lost by timestamp rounding")
    require(torch.as_tensor(sample["tactile_valid"])[-1].all(), "Current ten-finger tactile is absent")
    require((future[torch.isfinite(future)] > 0).all(), "Future labels are not future observations")
    require(torch.allclose(torch.as_tensor(sample["tactile_history_times_s"]).double(),
                           torch.arange(-3, 1, dtype=torch.float64) / 30, atol=1e-6, rtol=0),
            "History physical clock mismatch")
    require(torch.allclose(torch.as_tensor(sample["future_tactile_times_s"]).double(),
                           torch.arange(1, 9, dtype=torch.float64) / 10, atol=1e-6, rtol=0),
            "Future physical clock mismatch")
    return dict(index=index, position=position, episode_id=sample.get("episode_id"),
                episode_start_frame=sample.get("episode_start_frame"), tactile_region_names=names,
                prompt=sample["prompt"], prompt_sha256=hashlib.sha256(sample["prompt"].encode()).hexdigest(),
                scene_frame_sha256=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in sample["video"]],
                scene_frame_size=[width, height], video_mask=torch.as_tensor(sample["video_mask"]).tolist(),
                action_sha256=tensor_digest(action), action_mask_sha256=tensor_digest(mask),
                action_valid=int(mask.sum()), history_valid=int(torch.as_tensor(sample["tactile_valid"]).sum()),
                future_valid=int(torch.as_tensor(sample["future_tactile_valid"]).sum()),
                current_tactile_source_offset_s=float(history[-1]),
                history_source_offsets_s=[float(x) if torch.isfinite(x) else None for x in history],
                future_source_offsets_s=[float(x) if torch.isfinite(x) else None for x in future],
                tactile_maps_sha256=tensor_digest(sample["tactile_maps"]),
                future_tactile_maps_sha256=tensor_digest(sample["future_tactile_maps"]))


def prepared_backward(model, samples, *, tail=False):
    """Exercise the actual preparation/collation methods with auditable masks."""
    model.zero_grad(set_to_none=True)
    prepared = model.prepare_inputs(samples)
    expected_actions = torch.stack([torch.as_tensor(s["action"]) for s in samples]).to(model.device, model.dtype)
    require(torch.equal(prepared["actions"], expected_actions), "Preparation changed native action values")
    expected_proprio = torch.stack([torch.as_tensor(s["proprio"]).reshape(80) for s in samples]).to(model.device, model.dtype)
    require(torch.equal(prepared["proprio"], expected_proprio), "Preparation changed native state values")
    require(torch.equal(prepared["action_is_pad"].cpu(),
                        torch.stack([~torch.as_tensor(s["action_mask"]) for s in samples])),
            "Preparation changed action masks")
    require(torch.equal(prepared["proprio_mask"].cpu(),
                        torch.stack([torch.as_tensor(s["proprio_mask"]) for s in samples])),
            "Preparation changed state masks")
    expected_pad = torch.stack([downsample_video_mask_to_latent(~torch.as_tensor(s["video_mask"]))
                                for s in samples])
    require(torch.equal(prepared["video_is_pad"].cpu(), expected_pad), "Wrong latent video padding")
    ids = torch.tensor([[REGION_NAMES.index(n) for n in sample["tactile_region_names"]] for sample in samples])
    require(torch.equal(prepared["tactile_region_ids"].cpu(), ids), "Regional collation changed identities")
    require((prepared["tactile_sensor_ids"] == 0).all(), "Sharpa sensor ID is not zero")
    for key in ("tactile_maps", "tactile_valid", "future_tactile_maps", "future_tactile_valid",
                "tactile_layout_present", "tactile_query_mask", "tactile_spatial_valid"):
        require(torch.equal(prepared[key].cpu(), torch.stack([torch.as_tensor(s[key]) for s in samples])),
                f"Production regional collation changed {key}")
    if tail:
        require(not prepared["future_tactile_valid"].any(), "Train episode tail has future labels")
        require(prepared["tactile_query_mask"].all(), "Missing future labels removed valid output queries")
        require(prepared["video_is_pad"].all(), "Train tail incorrectly supervises padded video")
        require((~prepared["action_is_pad"]).any(), "Train tail lost its remaining action labels")
    else:
        require(prepared["future_tactile_valid"].any(), "Batch has no tactile labels for gradient validation")
    losses = model.compute_loss(**prepared)
    require(all(torch.isfinite(value).all() for value in losses.values()), "Nonfinite loss/metrics")
    losses["loss"].backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Nonfinite gradient")
    tactile_grad = sum(float(p.grad.detach().float().abs().sum()) for p in model.touch_v2.parameters()
                       if p.grad is not None)
    if tail:
        require(float(losses["loss_tactile"]) == 0, "Unlabeled tail contributed tactile loss")
        require(float(losses["loss_video"]) == 0, "Padded tail contributed video loss")
    else:
        require(tactile_grad > 0 and np.isfinite(tactile_grad), "No finite nonzero tactile gradients")
    return dict(batch_size=len(samples), tail=tail, finite_backward=True,
                tactile_gradient_l1=tactile_grad,
                losses={key: float(value.detach()) for key, value in losses.items()},
                prepared_shapes={key: list(value.shape) for key, value in prepared.items() if torch.is_tensor(value)},
                video_is_pad=prepared["video_is_pad"].cpu().tolist(),
                regional_ids=prepared["tactile_region_ids"].cpu().tolist(),
                action_masks_preserved=True, proprio_masks_preserved=True, tactile_masks_preserved=True)


def run(args):
    started = time.time()
    torch.set_num_threads(1)
    # CPU BF16 DNNL backward is unsupported on some development hosts.
    if args.device == "cpu":
        torch.backends.mkldnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    repo = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="train_touchwam_pretrain")
    source = OmegaConf.to_container(cfg.dataloader.datasets.sharpadex, resolve=True)
    original_source = dict(source)
    cache = Path(args.tactile_root or source["tactile_root"])
    catalog_path = cache / "tactile_catalog.json"
    catalog_raw = catalog_path.read_bytes()
    catalog = json.loads(catalog_raw)
    entries = catalog["bucket_manifest_by_name"]
    name = args.bucket or sorted(entries)[0]
    entry = entries[name]
    bucket = Path(entry["source_bucket"])
    manifest = cache / entry["manifest"]
    require(not entry.get("decode_errors"), "Selected bucket has decode errors")
    require(digest(bucket / "meta/info.json") == entry["source_info_sha256"], "Catalog/source identity mismatch")
    require(digest(manifest) == entry["manifest_sha256"], "Catalog/sidecar identity mismatch")
    require(args.allow_single_bucket_publication_bypass,
            "Use --allow-single-bucket-publication-bypass for this bounded completed-bucket check")
    # Source gate is intentionally not weakened in the production YAML.
    source.update(dataset_dir=str(bucket), tactile_root=str(cache), require_publication=False)
    info = json.loads((bucket / "meta/info.json").read_text())
    splits = list(info["splits"])
    require(set(splits) <= {"train", "val"}, f"Unexpected actual split names: {splits}")
    require("train" in splits, "A train split is needed for episode-tail acceptance")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        require(free >= args.minimum_free_mib * 1024 ** 2, "Insufficient free GPU memory; other workloads preserved")
        torch.cuda.set_per_process_memory_fraction(args.max_allocated_mib * 1024 ** 2 / total, device)
        torch.cuda.reset_peak_memory_stats(device)
    model = tiny_model(device, dtype)
    model.preprocess = MethodType(bounded_encoders, model)
    model.set_training_runtime(use_gradient_checkpointing=True)
    reports = []
    for split in splits:
        ds = build_dataset(OmegaConf.create(source), split=split)
        require(ds.touch.digest == entry["manifest_sha256"], "Reader resolved a different tactile manifest")
        require(len(ds) >= 3, f"Actual {split} split is too small for first/middle/tail")
        indices = [("first", 0), ("middle", len(ds) // 2), ("tail", len(ds) - 1)]
        samples, receipts = [], []
        for position, index in indices:
            # Deliberately bypass _safe_get: never substitute a failed sample.
            sample = ds._getitem_impl(index)
            receipts.append(sample_receipt(sample, index=index, position=position,
                                           width=source["width"], height=source["height"]))
            samples.append(sample)
        accepted = prepared_backward(model, samples)
        tail_result = prepared_backward(model, samples[-1:], tail=True) if split == "train" else None
        reports.append(dict(split=split, split_definition=info["splits"][split], reader_length=len(ds),
                            selected_episode_count=len(ds._eps_df), samples=receipts,
                            batch_forward_backward=accepted, tail_forward_backward=tail_result))
        print(json.dumps({"split": split, "status": "passed", "samples": len(samples),
                          "tactile_gradient_l1": accepted["tactile_gradient_l1"]}), flush=True)
        del ds, samples, sample
        gc.collect()
    report = dict(schema="touchwam_sharpa_real_pipeline_acceptance_v1", status="passed",
                  validation_only=True, formal_training_started=False, optimizer_steps=0,
                  full_alpha_loaded=False, production_vae_and_text_encoder_used=False,
                  scope="One cataloged completed bucket; real scene/text/tactile/actions; production preparation and tiny real Wan/ActionDiT/regional backward",
                  substitutions={"VAE": "RGB + luminance, 4×4 image pooling, causal 1+4+4 temporal grouping",
                                 "T5": "SHAKE256 of real UTF-8 language, 4×32 context",
                                 "model": "Tiny random real architecture, two layers; not full Alpha acceptance"},
                  device=str(device), dtype=str(dtype), seed=args.seed, torch_version=torch.__version__,
                  model_parameters=sum(p.numel() for p in model.parameters()),
                  tiny_tactile_spec=model.tactile_spec,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                  script_sha256=digest(__file__), source_config=source, original_source_config=original_source,
                  publication_bypass={"explicit": True, "scope": "only this completed bucket",
                                      "catalog_status": catalog["status"], "training_gate_changed": False},
                  catalog_path=str(catalog_path), catalog_snapshot_sha256=hashlib.sha256(catalog_raw).hexdigest(),
                  catalog_counts={k: catalog.get(k) for k in ("buckets_expected", "buckets_completed", "frames", "completed_frames")},
                  bucket=name, bucket_catalog_entry=entry, actual_splits=splits,
                  reader_retries=0, reports=reports, elapsed_s=time.time() - started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else None)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tactile-root")
    parser.add_argument("--bucket")
    parser.add_argument("--allow-single-bucket-publication-bypass", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-free-mib", type=int, default=1024)
    parser.add_argument("--max-allocated-mib", type=int, default=512)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
