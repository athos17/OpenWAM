#!/usr/bin/env python3
"""Check the frozen-encoder training fast paths on real batches: speed and numerics.

Runs on one GPU with the training flags (``seed_everything`` -> cuDNN deterministic,
no benchmark), on real micro-batches drawn from each source of a training config:

* text  : online ``encode_text`` vs ``TextEmbedCache.lookup`` (``--cache``), bitwise.
* video : legacy per-frame preprocess vs batched preprocess (bitwise), then the VAE
          encode under deterministic flags (baseline) vs cuDNN autotune, autotune +
          channels_last_3d, and sub-batching. Diffs are reported against the
          baseline and, for scale, the baseline's own bf16-vs-fp32 error.

Example:
  python scripts/profiling/check_train_encoders.py --config-name train_action_random_32gpu \\
      --cache /mnt/pfs/v89atp/workspace/cache/text_embed/alpha154000_three_source.safetensors \\
      --out work/encoder_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "precompute"))

from build_text_embed_cache import load_text_encoder  # noqa: E402

VAE_PREFIX = "video_backbone.vae."


def load_vae(ckpt: str, device, dtype=torch.bfloat16):
    from safetensors import safe_open

    from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38

    vae = WanVideoVAE38()
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        sd = {k[len(VAE_PREFIX) :]: f.get_tensor(k) for k in f.keys() if k.startswith(VAE_PREFIX)}
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert not [m for m in missing if not m.endswith(("mean", "std", "scale"))], missing[:5]
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


def sync_time(fn, iters: int, warmup: int = 1):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return out, ts[len(ts) // 2]


def diff(a, b) -> dict:
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {
        "bitwise": bool(torch.equal(a, b)),
        "equal_frac": (d == 0).float().mean().item(),
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_l2": ((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
        "cos": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    }


def real_batches(cfg, batch: int) -> dict:
    """One micro-batch per mixture source, samples evenly spaced through the source;
    robot sources (video_action) are also merged into one mixed batch as in training."""
    from openwam.dataloader.registry import build_dataset

    ds = build_dataset(cfg.dataloader)
    out, robot = {}, []
    for name, sub, modality in zip(ds._names, ds._datasets, ds.source_modalities):
        idx = [int(i * (len(sub) - 1) / max(1, batch - 1)) for i in range(batch)]
        out[name] = [sub[i] for i in idx]
        if modality == "video_action":
            robot.append(out[name])
    if len(robot) > 1:
        out["robot_mixed"] = [s for group in zip(*robot) for s in group][:batch]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=str(REPO / "configs"))
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--cache", default=None, help="text embed cache to check against online encoding")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir

    from openwam.model.video_backbone.wan import encode as wan_encode
    from openwam.model.video_backbone.wan.preprocess import preprocess_video, preprocess_video_batch
    from openwam.model.video_backbone.wan_backbone import _cudnn_autotune
    from openwam.train.utils.seeding import seed_everything

    with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base=None):
        cfg = compose(config_name=args.config_name)
    ckpt = args.ckpt or str(cfg.training.finetune_ckpt_path)
    tok_dir = args.tokenizer or str(Path(ckpt).parent / "tokenizer/google/umt5-xxl")
    seed_everything(42)  # training flags: cudnn.deterministic=True, benchmark=False
    device, dtype = torch.device("cuda"), torch.bfloat16

    t0 = time.time()
    batches = real_batches(cfg, args.batch)
    print(f"[check] batches {[(k, len(v)) for k, v in batches.items()]} ({time.time() - t0:.0f}s)", flush=True)
    tokenizer, text_encoder = load_text_encoder(ckpt, tok_dir, device)
    vae = load_vae(ckpt, device)
    report = {"config": args.config_name, "ckpt": ckpt, "batch": args.batch, "gpu": torch.cuda.get_device_name(0),
              "text": {}, "video": {}}

    # ------------------------------------------------------------------ text
    cache = None
    if args.cache:
        from openwam.model.video_backbone.wan.text_embed_cache import TextEmbedCache

        cache = TextEmbedCache.load(args.cache, device=device)
    for name, samples in batches.items():
        prompts = [s["prompt"] for s in samples]
        (ctx, lens), t_online = sync_time(
            lambda: wan_encode.encode_text(prompts, tokenizer=tokenizer, text_encoder=text_encoder, device=device),
            args.iters,
        )
        r = {"online_s": t_online, "unique_prompts": len(set(prompts))}
        if cache is not None:
            r["missing"] = len(cache.missing(prompts))
            if not r["missing"]:
                (cctx, clens), t_cache = sync_time(lambda: cache.lookup(prompts), args.iters)
                r.update(cache_s=t_cache, seq_lens_equal=bool(torch.equal(clens.cpu(), lens.cpu())),
                         context=diff(cctx, ctx))
        report["text"][name] = r
        print(f"[text] {name}: {r}", flush=True)
    del text_encoder
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ video
    for name, samples in batches.items():
        frames = [s["video"] for s in samples]
        legacy, t_pre_old = sync_time(
            lambda: torch.cat([preprocess_video(c, dtype=dtype, device=device) for c in frames], dim=0), args.iters
        )
        fast, t_pre_new = sync_time(lambda: preprocess_video_batch(frames, dtype=dtype, device=device), args.iters)
        r = {"shape": list(legacy.shape), "preprocess_legacy_s": t_pre_old, "preprocess_batched_s": t_pre_new,
             "preprocess_bitwise": bool(torch.equal(legacy, fast))}
        x = legacy

        base, r["encode_deterministic_s"] = sync_time(lambda: vae.batch_encode(x, device), args.iters)
        r["deterministic_rerun"] = diff(vae.batch_encode(x, device), base)

        vae32 = vae.float()
        ref32 = vae32.batch_encode(x.float(), device)
        vae.to(dtype=dtype)
        r["bf16_vs_fp32_reference"] = diff(base, ref32)

        def autotune():
            with _cudnn_autotune(True):
                return vae.batch_encode(x, device)

        lat, r["encode_autotune_s"] = sync_time(autotune, args.iters, warmup=2)
        r["autotune_vs_deterministic"] = diff(lat, base)
        r["autotune_vs_fp32"] = diff(lat, ref32)

        vae.model.to(memory_format=torch.channels_last_3d)
        try:
            xcl = x.contiguous(memory_format=torch.channels_last_3d)
            with _cudnn_autotune(True):
                lat_cl, r["encode_autotune_channels_last_s"] = sync_time(lambda: vae.batch_encode(xcl, device),
                                                                         args.iters, warmup=2)
            r["autotune_channels_last_vs_deterministic"] = diff(lat_cl.contiguous(), base)
        except Exception as e:  # noqa: BLE001 - report, keep going
            r["autotune_channels_last_error"] = f"{type(e).__name__}: {e}"
        finally:
            vae.model.to(memory_format=torch.contiguous_format)

        half = x.shape[0] // 2
        sub, r["encode_deterministic_subbatch2_s"] = sync_time(
            lambda: torch.cat([vae.batch_encode(x[:half], device), vae.batch_encode(x[half:], device)]), args.iters
        )
        r["subbatch2_vs_deterministic"] = diff(sub, base)
        r["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["video"][name] = r
        print(f"[video] {name}: " + json.dumps(r, indent=1), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"[check] wrote {args.out}")


if __name__ == "__main__":
    main()
