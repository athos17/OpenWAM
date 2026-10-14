#!/usr/bin/env python3
"""Build / verify the offline UMT5-context and Wan-VAE-latent caches.

The encoders are run through the exact training call path (same Hydra config,
same checkpoint loader, same freeze, ``seed_everything`` cudnn settings, B=48,
``wan.encode.encode_text`` / ``preprocess_video`` + ``encode_video``), and the
bf16 outputs are stored as-is, so a cached row equals the online value whenever
the online batch grouping matches.

Sub-commands (Hydra overrides = the training run's overrides, after ``--``)::

    encode   --out DIR --shard I --num-shards N [--kinds text latent]  -- <overrides>
    finalize --out DIR --num-shards N [--kinds ...]                    -- <overrides>
    verify   --out DIR [--kinds ...]
    check    --out DIR [--groups 2] [--kinds ...]                      -- <overrides>

``encode`` runs one GPU shard (one process per GPU, any node); ``finalize``
assembles ``meta.json`` once all shard files are present; ``verify`` re-hashes
every file on the current host and stamps ``VERIFIED-<host>.json``; ``check``
recomputes online and compares against the verified cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BATCH = 48  # encoder call batch (--batch): use the training per-GPU batch for bit-exact online parity


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compose_cfg(overrides: list):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=list(overrides))
    # Caches are built from the online path, never from another cache.
    OmegaConf.update(cfg, "dataloader.latent_cache_dir", None, force_add=True)
    OmegaConf.update(cfg, "dataloader.text_cache_dir", None, force_add=True)
    seed = OmegaConf.select(cfg, "project.seed", default=None)
    if seed is not None:
        OmegaConf.update(cfg.dataloader, "seed", int(seed), force_add=True)
    return cfg


def build_dataset(cfg):
    from openwam.dataloader.registry import build_dataset as _build

    dataset = _build(cfg.dataloader, split="train")
    if type(dataset).__name__ != "MultiTaskRoboTwinDataset" and not hasattr(dataset, "encode_cache_window_table"):
        raise ValueError(f"{type(dataset).__name__} does not implement the encode-cache protocol")
    return dataset


def check_window(dataset, windows, row: int, sample: dict) -> None:
    """The sample produced for global index ``row`` must be exactly that window."""
    if "window_index" in sample:  # protocol readers report the index they served
        ok = int(sample["window_index"]) == row
    else:  # MultiTaskRoboTwinDataset: (sub-dataset, episode, start)
        d, e, s = windows[row]
        sub = dataset._sub_datasets[d]
        ok = (sample["episode_path"], sample["start_frame"]) == (sub._episode_files[e], int(s))
    if not ok:
        raise ValueError(f"window order mismatch at row {row}")


def build_model(cfg, device):
    """Mirror OpenWAMTrainer.__init__: ckpt-dir build -> dtype/device -> freeze."""
    import torch

    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir

    torch.cuda.set_device(device)
    _, arch, _ = build_architecture_from_ckpt_dir(
        cfg.training.finetune_ckpt_path, weights_required=True, override_cfg=cfg
    )
    arch.set_dtype_device(arch.dtype, device)
    arch.freeze_modules(list(cfg.model.get("freeze", [])))
    vb = arch.video_backbone
    if vb.video_encoder is not None:
        raise ValueError("latent cache supports the native Wan VAE only")
    return arch, vb


# ----------------------------------------------------------------------------
# Online encoders — the same calls as WanBackbone.preprocess_input_for_train
# ----------------------------------------------------------------------------


def online_text(vb, prompts):
    from openwam.model.video_backbone.wan import encode as wan_encode

    return wan_encode.encode_text(prompts, tokenizer=vb._tokenizer, text_encoder=vb.text_encoder, device=vb.device)


def online_latents(vb, clips):
    from openwam.model.video_backbone.wan import encode as wan_encode

    import torch

    videos = [wan_encode.preprocess_video(c, encoder=vb.video_encoder, dtype=vb.dtype, device=vb.device) for c in clips]
    stacked = torch.cat(videos, dim=0)
    latents = wan_encode.encode_video(stacked, vae=vb.vae, encoder=vb.video_encoder)
    return latents.to(dtype=vb.dtype, device=vb.device)


def padded(items: list) -> list:
    """Repeat the last item so every encoder call has the training batch shape."""
    return items + [items[-1]] * (BATCH - len(items))


def token_lengths(vb, prompts) -> list:
    out = []
    for i in range(0, len(prompts), 4096):
        _, mask = vb._tokenizer(
            prompts[i : i + 4096],
            return_mask=True,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        out.extend(mask.gt(0).sum(dim=1).tolist())
    return out


# ----------------------------------------------------------------------------
# encode
# ----------------------------------------------------------------------------


def encode_text_shard(args, cfg, dataset, vb, out: Path):
    import numpy as np
    import torch

    from openwam.dataloader.utils import encode_cache as ec

    prompts = ec.candidate_prompt_list(dataset)
    lens = token_lengths(vb, prompts)
    lmax = max(lens)
    if lmax >= ec.TEXT_MAX_LENGTH:
        raise ValueError("a prompt reaches the 512-token limit; trimming would not be lossless")
    signature = ec.text_signature(dataset)
    start, end = ec.shard_bounds(len(prompts), args.shard, args.num_shards)
    name = ec.shard_name(args.shard, args.num_shards)
    dim = None
    part = out / (name + ".partial")
    rows = None
    log(f"text shard {args.shard}: prompts [{start}, {end}) of {len(prompts)}, lmax={lmax}")
    t0 = time.time()
    with torch.no_grad():
        for b in range(start, end, BATCH):
            batch = prompts[b : min(b + BATCH, end)]
            context, seq_lens = online_text(vb, padded(batch))
            if context.dtype != torch.bfloat16 or context.shape[1] != ec.TEXT_MAX_LENGTH:
                raise ValueError(f"unexpected online context {context.dtype} {tuple(context.shape)}")
            if seq_lens[: len(batch)].tolist() != lens[b : b + len(batch)]:
                raise ValueError("tokenizer lengths differ between passes")
            if bool(context[:, lmax:].any()):
                raise ValueError("non-zero context past max seq_len; trimming would be lossy")
            if rows is None:
                dim = context.shape[-1]
                rows = np.memmap(part, dtype=np.int16, mode="w+", shape=(end - start, lmax, dim))
            rows[b - start : b - start + len(batch)] = (
                context[: len(batch), :lmax].contiguous().view(torch.int16).cpu().numpy()
            )
            if (b - start) // BATCH % 50 == 0:
                log(f"text shard {args.shard}: {b - start + len(batch)}/{end - start} ({time.time() - t0:.0f}s)")
    if rows is None:  # empty shard (more shards than prompts)
        part.write_bytes(b"")
        dim = vb.text_encoder.dim if hasattr(vb.text_encoder, "dim") else 4096
    else:
        rows.flush()
        del rows
    part.replace(out / name)
    np.save(out / f"seq_lens-{args.shard:02d}.npy", np.asarray(lens[start:end], dtype=np.int32))
    ec.write_shard_record(
        out,
        args.shard,
        args.num_shards,
        start,
        end,
        out / name,
        extra={
            "signature_sha256": ec.digest(signature),
            "encoder_fingerprint": ec.text_encoder_fingerprint(vb),
            "tokenizer_sha256": ec.tree_digest(Path(cfg.training.finetune_ckpt_path) / "tokenizer"),
            "row_shape": [lmax, dim],
            "num_rows": len(prompts),
            "encode_batch": BATCH,
            "encode_seconds": time.time() - t0,
        },
    )
    log(f"text shard {args.shard}: done in {time.time() - t0:.0f}s")


def encode_latent_shard(args, cfg, dataset, vb, out: Path):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from openwam.dataloader.utils import encode_cache as ec

    signature = ec.latent_signature(dataset)
    if signature["color_jitter"]:
        raise ValueError("color_jitter must be disabled for a latent cache")
    windows = ec.window_table(dataset)
    start, end = ec.shard_bounds(len(dataset), args.shard, args.num_shards)
    name = ec.shard_name(args.shard, args.num_shards)
    part = out / (name + ".partial")
    loader = DataLoader(
        Subset(dataset, range(start, end)),
        batch_size=BATCH,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=list,
        prefetch_factor=4,
        persistent_workers=False,
    )
    rows = None
    row_shape = None
    done = start
    log(f"latent shard {args.shard}: windows [{start}, {end}) of {len(dataset)}")
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            for k, sample in enumerate(batch):
                check_window(dataset, windows, done + k, sample)
            latents = online_latents(vb, padded([s["video"] for s in batch]))
            if latents.dtype != torch.bfloat16:
                raise ValueError(f"unexpected latent dtype {latents.dtype}")
            if rows is None:
                row_shape = list(latents.shape[1:])
                rows = np.memmap(part, dtype=np.int16, mode="w+", shape=(end - start, *row_shape))
            rows[done - start : done - start + len(batch)] = (
                latents[: len(batch)].contiguous().view(torch.int16).cpu().numpy()
            )
            done += len(batch)
            if (done - start) // BATCH % 50 == 0:
                rate = (done - start) / max(time.time() - t0, 1e-9)
                log(f"latent shard {args.shard}: {done - start}/{end - start} ({rate:.1f} windows/s)")
    if done != end:
        raise ValueError(f"latent shard {args.shard} wrote {done - start} rows, expected {end - start}")
    if rows is None:
        raise ValueError(f"latent shard {args.shard} is empty; use fewer shards")
    rows.flush()
    del rows
    part.replace(out / name)
    ec.write_shard_record(
        out,
        args.shard,
        args.num_shards,
        start,
        end,
        out / name,
        extra={
            "signature_sha256": ec.digest(signature),
            "encoder_fingerprint": ec.vae_fingerprint(vb),
            "tokenizer_sha256": None,
            "row_shape": row_shape,
            "num_rows": len(dataset),
            "encode_batch": BATCH,
            "encode_seconds": time.time() - t0,
        },
    )
    log(f"latent shard {args.shard}: done in {time.time() - t0:.0f}s")


def cmd_encode(args, overrides):
    import torch

    from openwam.train.utils.seeding import seed_everything

    cfg = compose_cfg(overrides)
    seed_everything(int(cfg.project.seed), rank=args.shard)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    dataset = build_dataset(cfg)
    _, vb = build_model(cfg, device)
    for kind in args.kinds:
        out = Path(args.out) / kind
        out.mkdir(parents=True, exist_ok=True)
        (encode_text_shard if kind == "text" else encode_latent_shard)(args, cfg, dataset, vb, out)


# ----------------------------------------------------------------------------
# finalize / verify
# ----------------------------------------------------------------------------


def cmd_finalize(args, overrides):
    import numpy as np

    from openwam.dataloader.utils import encode_cache as ec

    cfg = compose_cfg(overrides)
    dataset = build_dataset(cfg)
    for kind in args.kinds:
        out = Path(args.out) / kind
        records = ec.load_shard_records(out, args.num_shards)
        row_shape = records[0]["row_shape"]
        if kind == "text":
            prompts = ec.candidate_prompt_list(dataset)
            (out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False))
            lens = np.concatenate([np.load(out / f"seq_lens-{i:02d}.npy") for i in range(args.num_shards)])
            if len(lens) != len(prompts):
                raise ValueError("seq_lens rows differ from prompt count")
            np.save(out / "seq_lens.npy", lens)
            side = ["prompts.json", "seq_lens.npy"]
            signature, num_rows = ec.text_signature(dataset), len(prompts)
        else:
            np.save(out / "windows.npy", ec.window_table(dataset))
            side = ["windows.npy"]
            signature, num_rows = ec.latent_signature(dataset), len(dataset)
        meta = ec.finalize_meta(
            out,
            kind=kind,
            signature=signature,
            num_rows=num_rows,
            row_shape=row_shape,
            count=args.num_shards,
            side_files={name: ec.sha256_file(out / name) for name in side},
            extra={
                "built_by": "scripts/precompute/build_encode_cache.py",
                "batch": records[0].get("encode_batch", BATCH),
                "encode_seconds_per_shard": [r["encode_seconds"] for r in records],
                "hosts": sorted({r["host"] for r in records}),
            },
        )
        log(f"{kind}: meta.json rows={num_rows} shape={row_shape} manifest={meta['manifest_sha256'][:16]}")


def cmd_verify(args, _overrides):
    from openwam.dataloader.utils import encode_cache as ec

    for kind in args.kinds:
        stamp = ec.verify_cache(Path(args.out) / kind)
        log(f"{kind}: verified {stamp['files']} files on {stamp['host']} manifest={stamp['manifest_sha256'][:16]}")


# ----------------------------------------------------------------------------
# check: cache vs online recomputation
# ----------------------------------------------------------------------------


def compare(a, b) -> dict:
    import torch

    a, b = a.float().flatten(), b.float().flatten()
    diff = (a - b).abs()
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs": float(diff.max()),
        "rel_l2": float(diff.norm() / b.norm().clamp_min(1e-12)),
        "cos": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    }


def aligned_group(rng, shards) -> list:
    """Row indices of one encode-pass batch (B=48 grouping inside a non-empty shard)."""
    shard = rng.choice([r for r in shards if r["row_end"] > r["row_start"]])
    j = rng.randrange(0, max(1, (shard["row_end"] - shard["row_start"] + BATCH - 1) // BATCH))
    return list(range(shard["row_start"] + j * BATCH, min(shard["row_start"] + (j + 1) * BATCH, shard["row_end"])))


def random_group(rng, n: int) -> list:
    """A shuffled batch: distinct rows when possible, otherwise drawn with replacement."""
    return rng.sample(range(n), BATCH) if n >= BATCH else [rng.randrange(n) for _ in range(BATCH)]


def cmd_check(args, overrides):
    global BATCH
    import random

    import torch

    from openwam.dataloader.utils import encode_cache as ec
    from openwam.train.utils.seeding import seed_everything

    cfg = compose_cfg(overrides)
    seed_everything(int(cfg.project.seed), rank=0)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    dataset = build_dataset(cfg)
    _, vb = build_model(cfg, device)
    rng = random.Random(args.seed)
    report = {}
    with torch.no_grad():
        if "text" in args.kinds:
            cache = ec.TextCache(Path(args.out) / "text", dataset)
            BATCH = cache.meta.get("batch", BATCH)  # recompute with the cache's own call shape
            prompts = json.loads((Path(args.out) / "text" / "prompts.json").read_text())
            if ec.text_encoder_fingerprint(vb) != cache.meta["encoder_fingerprint"]:
                raise ValueError("text encoder weights differ from the cache")
            report["text"] = []
            for mode in ("aligned", "shuffled"):
                for _ in range(args.groups):
                    if mode == "aligned":  # same grouping as the encode pass
                        idx = aligned_group(rng, cache.meta["shards"])
                    else:  # with replacement when there are fewer rows than a batch
                        idx = random_group(rng, len(prompts))
                    batch = [prompts[i] for i in idx]
                    context, seq_lens = online_text(vb, padded(batch))
                    embeds = [cache.get(p) for p in batch]
                    cached, cached_lens = ec_load_text(embeds, device)
                    entry = compare(cached, context[: len(batch)])
                    entry.update(mode=mode, rows=[idx[0], idx[-1]], seq_lens_equal=cached_lens.tolist() == seq_lens[: len(batch)].tolist())
                    report["text"].append(entry)
                    log(f"text {mode}: {entry}")
        if "latent" in args.kinds:
            cache = ec.LatentCache(Path(args.out) / "latent", dataset)
            BATCH = cache.meta.get("batch", BATCH)
            if ec.vae_fingerprint(vb) != cache.meta["encoder_fingerprint"]:
                raise ValueError("VAE weights differ from the cache")
            report["latent"] = []
            for mode in ("aligned", "shuffled"):
                for _ in range(args.groups):
                    if mode == "aligned":
                        idx = aligned_group(rng, cache.meta["shards"])
                    else:
                        idx = random_group(rng, len(dataset))
                    samples = [dataset[i] for i in idx]
                    online = online_latents(vb, padded([s["video"] for s in samples]))[: len(idx)]
                    cached = torch.stack([cache.get(i) for i in idx]).to(device)
                    entry = compare(cached, online)
                    entry.update(mode=mode, rows=[idx[0], idx[-1]])
                    report["latent"].append(entry)
                    log(f"latent {mode}: {entry}")
    out = Path(args.out) / f"check-{os.uname().nodename}.json"
    out.write_text(json.dumps(report, indent=2))
    aligned = [e for k in report for e in report[k] if e["mode"] == "aligned"]
    if not all(e["bitwise_equal"] for e in aligned) or not all(e.get("seq_lens_equal", True) for e in aligned):
        raise SystemExit(f"FAILED: aligned groups are not bitwise equal; see {out}")
    log(f"OK: aligned groups bitwise equal; report {out}")


def ec_load_text(embeds, device):
    from openwam.model.video_backbone.wan import encode as wan_encode

    return wan_encode.load_cached_text([e for e, _ in embeds], [n for _, n in embeds], device=device)


def raise_nofile_limit():
    """DataLoader workers pass every tensor through a file descriptor; 1024 is too few."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def main():
    raise_nofile_limit()
    argv = sys.argv[1:]
    overrides = []
    if "--" in argv:
        cut = argv.index("--")
        argv, overrides = argv[:cut], argv[cut + 1 :]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["encode", "finalize", "verify", "check"])
    ap.add_argument("--out", required=True, help="cache root; holds text/ and latent/")
    ap.add_argument("--kinds", nargs="+", default=["text", "latent"], choices=["text", "latent"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--groups", type=int, default=2, help="check: batches per mode and kind")
    ap.add_argument("--seed", type=int, default=0, help="check: batch selection seed")
    ap.add_argument("--batch", type=int, default=48,
                    help="encoder call batch; the VAE is batch-size sensitive, so match the training per-GPU batch")
    args = ap.parse_args(argv)
    global BATCH
    BATCH = args.batch
    if not 0 <= args.shard < args.num_shards:
        ap.error("--shard must be in [0, --num-shards)")
    {"encode": cmd_encode, "finalize": cmd_finalize, "verify": cmd_verify, "check": cmd_check}[args.command](
        args, overrides
    )


if __name__ == "__main__":
    main()
