"""Held-out tactile map/image metrics; never scores flow velocity as observations.

Use ``evaluate_sample(model, sample, ...)`` with an already normalized reader
sample, or the CLI's small JSON manifest + NPZ sample interchange (see docs).
Only the first scene image and tactile history enter generation, never targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from openwam.tactile.regions import CODEC, INTENSITY_CODEC, RGB_CODEC


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def maps(value):
    value = array(value)
    if value.dtype == np.uint8:
        return value.astype(np.float32) / 255
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError("Tactile maps/images must be uint8 or float [0,1]")
    return value.astype(np.float32)


def metric(prediction, target, valid):
    """Float64 sums support exact pixel-weighted aggregation across samples."""
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Prediction, target and pixel mask shapes must match")
    if not np.isfinite(prediction[valid]).all() or not np.isfinite(target[valid]).all():
        raise ValueError("Nonfinite generated/target values on evaluated pixels")
    difference = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    count = int(difference.size)
    absolute, squared = float(np.abs(difference).sum()), float(np.square(difference).sum())
    return dict(pixels=count, absolute_error_sum=absolute, squared_error_sum=squared,
                mae=absolute / count if count else None, rmse=(squared / count) ** .5 if count else None)


def last_observed(history, valid):
    """Last actually observed map per region; missing history is not a zero map."""
    if valid.dtype != np.bool_ or valid.shape != history.shape[:2]:
        raise ValueError("History validity must be bool [H,N]")
    held = np.zeros_like(history[0])
    available = valid.any(0)
    for region in np.flatnonzero(available):
        held[region] = history[np.flatnonzero(valid[:, region])[-1], region]
    return held, available


def score_maps(prediction, target, valid, history, history_valid, region_names):
    """Predicted maps and zero/hold-last baselines on explicit pixel support."""
    held, available = last_observed(history, history_valid)
    held = np.broadcast_to(held, target.shape)
    zero = np.zeros_like(target)
    common = valid & available[None, :, None, None, None]
    predictions = {"model": prediction, "zero": zero, "hold_last": held}

    def scores(mask):
        return {key: metric(value, target, mask & common if key == "hold_last" else mask)
                for key, value in predictions.items()}

    return dict(
        all_target_support=scores(valid), common_history_support=scores(common),
        by_region={name: scores(valid & (np.arange(len(region_names)) == i)[None, :, None, None, None])
                   for i, name in enumerate(region_names)},
        by_horizon=[scores(valid & (np.arange(len(target)) == i)[:, None, None, None, None])
                    for i in range(len(target))],
    ), held, common


def save_region_panels(directory, names, target, prediction, held, valid, common, times):
    """One PNG per region, retaining RGB channels; pink marks missing pixels."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    size, left, top = 96, 125, 32
    rows = ("target", "generated", "black image" if target.shape[2] == 3 else "zero", "hold last", "absolute error")
    paths = []
    for region, name in enumerate(names):
        canvas = Image.new("RGB", (left + size * len(target), top + size * len(rows)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), name, fill="black")
        values = (target, prediction, np.zeros_like(target), held, np.abs(prediction - target))
        for t, seconds in enumerate(times):
            draw.text((left + t * size + 4, 16), f"+{seconds:.3f}s", fill="black")
        for row, (label, value) in enumerate(zip(rows, values)):
            draw.text((4, top + row * size + 8), label, fill="black")
            for t in range(len(target)):
                mask = (common if label == "hold last" else valid)[t, region].all(axis=0)
                tile_values = (np.nan_to_num(value[t, region]).clip(0, 1) * 255).round().astype(np.uint8)
                rgb = np.moveaxis(tile_values, 0, -1)
                if rgb.shape[-1] == 1:
                    rgb = np.repeat(rgb, 3, axis=-1)
                rgb[~mask] = [245, 190, 220]
                tile = Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)
                canvas.paste(tile, (left + t * size, top + row * size))
        path = directory / f"region_{region:02d}.png"
        canvas.save(path)
        paths.append({"region": name, "path": path.name})
    return paths


@torch.no_grad()
def evaluate_sample(model, sample, *, schedule, output_dir=None, seed=42, cfg_scale=1.0, cfg_merge=False):
    """Evaluate one held-out reader sample, with normalized proprio for VAT.

    Caller must select a held-out split. This evaluates VT (ego tactile) and
    VAT (robot tactile) without assuming every source has robot actions.
    """
    modality = sample.get("training_modality")
    if modality not in ("video_tactile", "video_action_tactile"):
        raise ValueError("Tactile evaluation requires VT or VAT")
    codec = sample.get("tactile_codec")
    if codec not in (CODEC, INTENSITY_CODEC, RGB_CODEC):
        raise ValueError("Evaluation requires a supported tactile codec")
    if modality == "video_tactile" and any(sample.get(k) is not None for k in ("action", "proprio")):
        raise ValueError("Ego VT evaluation cannot carry robot action/proprio")
    spec = model.tactile_spec
    if codec == RGB_CODEC:
        if getattr(model, "model_version", None) != "touchwam_regions_v1" or not spec.get("rgb_enabled", False):
            raise ValueError("RGB evaluation requires a regional checkpoint with the RGB adapter enabled")
        channels, size, representation = 3, spec.get("rgb_map_size", 128), "optical_rgb"
    else:
        channels, size, representation = 1, spec["map_size"], "normalized_contact_intensity"
    generation_codec = codec
    if codec in (CODEC, INTENSITY_CODEC) and spec.get("codec") in (CODEC, INTENSITY_CODEC):
        # Older scalar checkpoints use the same [-1,1] patch transform under
        # the legacy deform name. This alias does not apply to optical RGB.
        generation_codec = spec["codec"]
    names = list(sample["tactile_region_names"])
    if not names or len(set(names)) != len(names) or set(names) - set(spec["region_names"]):
        raise ValueError("Expected unique registered region names")
    history, target = maps(sample["tactile_maps"]), maps(sample["future_tactile_maps"])
    expected = (len(names), channels, size, size)
    roi_shape = (len(names), 1, size, size)
    if history.shape != (spec["history"], *expected) or target.shape != (spec["future_frames"], *expected):
        raise ValueError("Evaluation map shape differs from checkpoint")
    valid, history_valid = array(sample["future_tactile_valid"]), array(sample["tactile_valid"])
    spatial = array(sample.get("tactile_spatial_valid", np.ones(roi_shape, dtype=bool)))
    layout = array(sample["tactile_layout_present"])
    if (valid.dtype != np.bool_ or valid.shape != target.shape[:2] or history_valid.dtype != np.bool_
            or history_valid.shape != history.shape[:2] or spatial.dtype != np.bool_ or spatial.shape != roi_shape
            or layout.dtype != np.bool_ or layout.shape != (len(names),)):
        raise ValueError("Invalid evaluation validity/ROI/layout masks")
    times = np.arange(1, spec["future_frames"] + 1) * spec["future_stride"] / spec["fps"]
    history_times = np.arange(1 - spec["history"], 1) / spec["fps"]
    for key, expected_times in (("future_tactile_times_s", times), ("tactile_history_times_s", history_times)):
        value = array(sample[key])
        if value.shape != expected_times.shape or not np.allclose(value, expected_times, atol=1e-6, rtol=0):
            raise ValueError(f"{key} differs from checkpoint")
    observed = np.broadcast_to(history_valid[..., None, None, None] & spatial[None]
                               & layout[None, :, None, None, None], history.shape)
    labeled = np.broadcast_to(valid[..., None, None, None] & spatial[None]
                              & layout[None, :, None, None, None], target.shape)
    for value, mask in ((history, observed), (target, labeled)):
        if not np.isfinite(value[mask]).all() or ((value[mask] < 0) | (value[mask] > 1)).any():
            raise ValueError(f"Observed {representation} values must be finite in [0,1]")
    sensors = sample.get("tactile_sensor_names")
    if sensors is not None:
        if len(sensors) != len(names) or set(sensors) - set(spec["sensor_names"]):
            raise ValueError("Unregistered sensor/source names")
        sensors = np.array([spec["sensor_names"].index(name) for name in sensors], dtype=np.int64)
    else:
        sensors = sample.get("tactile_sensor_ids")
        if sensors is None:
            raise ValueError("Explicit tactile sensor/source identity is required")
    first_frame = sample.get("first_frame_image")
    if first_frame is None and sample.get("video") is not None:
        first_frame = sample["video"][0]
    if first_frame is None:
        raise ValueError("Held-out generation requires the observed first scene image")
    if isinstance(first_frame, (list, tuple)):
        if len(first_frame) != 1:
            raise ValueError("Expected a single observed first-frame image")
        first_frame = first_frame[0]
    if not isinstance(first_frame, Image.Image):
        first_frame = Image.fromarray(array(first_frame))
    kwargs = dict(
        generation_modality=modality, first_frame_image=first_frame,
        height=first_frame.height, width=first_frame.width, num_frames=sample.get("num_frames", 9),
        num_inference_steps=max(1, len(schedule) - 1),
        decode_video=False, seed=seed, cfg_scale=cfg_scale, cfg_merge=cfg_merge,
        tactile_maps=history, tactile_valid=history_valid,
        tactile_region_ids=np.array([spec["region_names"].index(name) for name in names], dtype=np.int64),
        tactile_layout_present=layout, tactile_spatial_valid=spatial, tactile_sensor_ids=sensors,
    )
    if getattr(model, "model_version", None) == "touchwam_regions_v1":
        kwargs["tactile_codec"] = generation_codec
    if "tactile_query_mask" in sample:
        kwargs["tactile_query_mask"] = sample["tactile_query_mask"]
    if modality == "video_action_tactile":
        if sample.get("proprio") is None or sample.get("action") is None:
            raise ValueError("VAT held-out samples require real action and normalized proprio")
        kwargs["proprio"] = sample["proprio"]
        kwargs["action_num_frames"] = len(sample["action"]) + 1
        if "action_mask" in sample:
            mask = array(sample["action_mask"])
            if mask.dtype != np.bool_ or mask.shape not in (array(sample["action"]).shape,
                                                           (len(sample["action"]),)) or not mask.any():
                raise ValueError("VAT evaluation needs a valid action supervision mask")
            if mask.ndim == 2:
                kwargs["active_action_mask"] = mask.any(0)
    was_training = getattr(model, "training", False)
    try:
        result = model.generate(schedule, sample["prompt"], **kwargs)
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
    if codec == RGB_CODEC and result.get("tactile_codec") != codec:
        raise ValueError("Generation returned the wrong tactile codec")
    prediction = np.zeros_like(target)
    generated_valid = np.zeros_like(valid)
    if len(set(result["tactile_region_names"])) != len(result["tactile_region_names"]):
        raise ValueError("Generation returned duplicate regions")
    for output_index, name in enumerate(result["tactile_region_names"]):
        index = names.index(name)
        prediction[:, index] = result["future_tactile_maps"][:, output_index]
        generated_valid[:, index] = result["future_tactile_valid"][:, output_index]
    requested = array(sample.get("tactile_query_mask", np.broadcast_to(layout, valid.shape)))
    if requested.dtype != np.bool_ or requested.shape != valid.shape:
        raise ValueError("Query mask must be bool [F,N]")
    requested = requested & layout[None] & spatial.reshape(len(names), -1).any(-1)[None]
    if not np.array_equal(generated_valid, requested):
        raise ValueError("Generated validity differs from request; do not hide missing predictions in metrics")
    pixels = labeled & generated_valid[..., None, None, None]
    if not pixels.any():
        raise ValueError("No measured future pixels overlap the generation request")
    scores, held, common = score_maps(prediction, target, pixels, history, history_valid & layout[None], names)
    report = dict(schema="touchwam_tactile_eval_v1", tactile_codec=codec, generation_codec=generation_codec,
                  tactile_representation=representation,
                  units="RGB channel values [0,1]" if codec == RGB_CODEC else "normalized intensity [0,1]",
                  channels=channels, metric_elements="channel_values" if channels == 3 else "pixels",
                  baselines={"zero": "black RGB image" if channels == 3 else "zero intensity", "hold_last": "last observed image/map"},
                  generation_modality=modality, seed=int(seed), cfg_scale=float(cfg_scale), cfg_merge=cfg_merge,
                  region_names=names, future_times_s=times.tolist(), metrics=scores)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", prediction=prediction, target=target, pixel_valid=pixels,
                            hold_last=held, common_history_valid=common, region_names=np.asarray(names),
                            future_times_s=times, tactile_codec=np.asarray(codec),
                            tactile_representation=np.asarray(representation))
        report["panels"] = save_region_panels(output, names, target, prediction, held, pixels, common, times)
        (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def aggregate(reports):
    """Aggregate exact pixel counts/sums, not an unweighted mean of MAEs."""
    if len({report.get("tactile_codec", INTENSITY_CODEC) for report in reports}) > 1:
        raise ValueError("Aggregate RGB and intensity metrics separately by tactile codec")
    result = {}
    for support in ("all_target_support", "common_history_support"):
        result[support] = {}
        for method in ("model", "zero", "hold_last"):
            rows = [report["metrics"][support][method] for report in reports]
            count = sum(row["pixels"] for row in rows)
            absolute = sum(row["absolute_error_sum"] for row in rows)
            squared = sum(row["squared_error_sum"] for row in rows)
            result[support][method] = dict(pixels=count, absolute_error_sum=absolute, squared_error_sum=squared,
                                          mae=absolute / count if count else None,
                                          rmse=(squared / count) ** .5 if count else None)
    return result


def load_sample(root, entry):
    """NPZ never uses pickle; JSON carries prompt/provenance/explicit modality."""
    with np.load(root / entry["arrays"], allow_pickle=False) as values:
        sample = {key: values[key] for key in values.files}
    sample.update(prompt=entry["prompt"], training_modality=entry["training_modality"],
                  tactile_codec=entry["tactile_codec"], num_frames=entry.get("num_frames", 9))
    for key in ("tactile_region_names", "tactile_sensor_names"):
        if key in sample:
            sample[key] = sample[key].tolist()
    with Image.open(root / entry["first_frame_image"]) as image:
        sample["first_frame_image"] = image.convert("RGB")
    return sample


def aggregate_by_source(reports):
    """A source may be evaluated through two adapters; keep their units separate."""
    result = {}
    for source in sorted({row["sample"]["source_id"] for row in reports}):
        rows = [row for row in reports if row["sample"]["source_id"] == source]
        codecs = sorted({row["tactile_codec"] for row in rows})
        result[source] = aggregate(rows) if len(codecs) == 1 else {
            "metrics": None,
            "by_representation": {codec: aggregate([row for row in rows if row["tactile_codec"] == codec])
                                  for codec in codecs}}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="TouchWAM safetensors file beside its config.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-merge", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
        raise ValueError("Choose an explicit checkpoint .safetensors file for reproducible evaluation")
    manifest_path = Path(args.manifest)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema") != "touchwam_tactile_eval_samples_v1"
            or manifest.get("split") not in ("validation", "test") or not manifest.get("samples")):
        raise ValueError("Expected a nonempty held-out validation/test sample manifest")
    if args.steps < 1:
        raise ValueError("Denoising steps must be positive")
    # Strict self-contained loader loads the trained model without choosing an
    # action normalizer. Samples already contain normalized training proprio;
    # ego VT needs no robot normalizer or fabricated action source at all.
    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir
    from openwam.deploy.denoise_schedule import schedule_sync

    _, model, _ = build_architecture_from_ckpt_dir(str(checkpoint), weights_required=True)
    if getattr(model, "model_version", None) != "touchwam_regions_v1":
        raise ValueError("Evaluation requires a trained regional TouchWAM checkpoint")
    model.set_dtype_device(torch.bfloat16 if args.dtype == "bf16" else torch.float32, torch.device(args.device))
    model.move_frozen_to_device(torch.device(args.device))
    model.eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, entry in enumerate(manifest["samples"]):
        if not all(key in entry for key in ("source_id", "episode_index", "reference_time_s")):
            raise ValueError("Each held-out sample requires source, episode and reference-time provenance")
        sample = load_sample(manifest_path.parent, entry)
        auxiliary = model.tactile_scheduler if sample["training_modality"] == "video_tactile" else model.action_scheduler
        schedule = schedule_sync(model.video_scheduler, auxiliary, num_steps=args.steps)
        report = evaluate_sample(model, sample, schedule=schedule, output_dir=output / f"sample_{index:05d}",
                                 seed=args.seed + index, cfg_scale=args.cfg_scale, cfg_merge=args.cfg_merge)
        report["sample"] = entry
        reports.append(report)
    summary = dict(schema="touchwam_tactile_eval_v1", split=manifest["split"],
                   manifest_sha256=hashlib.sha256(raw).hexdigest(), checkpoint=str(checkpoint),
                   checkpoint_size=checkpoint.stat().st_size, checkpoint_mtime_ns=checkpoint.stat().st_mtime_ns,
                   steps=args.steps, seed=args.seed,
                   metrics=aggregate(reports) if len({r["tactile_codec"] for r in reports}) == 1 else None,
                   by_representation={codec: aggregate([row for row in reports if row["tactile_codec"] == codec])
                                      for codec in sorted({row["tactile_codec"] for row in reports})},
                   by_source=aggregate_by_source(reports),
                   samples=reports)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["metrics"] or summary["by_representation"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
