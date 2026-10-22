#!/usr/bin/env python3
"""Precompute the frozen UMT5 text embeddings for every prompt a training config can emit.

Prompts come from the training dataset itself (``dataset.prompt_vocabulary()``),
so the cache keys are exactly the strings ``prepare_inputs`` sees. Each prompt is
encoded through the online ``encode_text`` call with the training call shape
(``--batch-size`` prompts per call, padded to 512), which keeps cached rows bitwise
identical to online encoding. The trainer re-checks this at startup
(``training.text_embed_cache.verify_prompts``) with its own text encoder.

Example (bdy, 8 GPUs, ~1 min):
  python scripts/precompute/build_text_embed_cache.py \\
      --config-name train_action_random_32gpu \\
      --out /mnt/pfs/v89atp/workspace/cache/text_embed/alpha154000_three_source.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parents[2]
TEXT_PREFIX = "video_backbone.text_encoder."


def load_text_encoder(ckpt: str, tokenizer_dir: str, device, dtype=torch.bfloat16):
    """UMT5 + tokenizer exactly as the training backbone builds them (model yaml:
    ``WanTextEncoder()`` / ``HuggingfaceTokenizer(seq_len=512, clean='whitespace')``),
    weights strictly loaded from the ``video_backbone.text_encoder.*`` slice of ``ckpt``."""
    from safetensors import safe_open

    from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer, WanTextEncoder

    tokenizer = HuggingfaceTokenizer(name=tokenizer_dir, seq_len=512, clean="whitespace")
    with torch.device("meta"):
        encoder = WanTextEncoder()
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        sd = {k[len(TEXT_PREFIX) :]: f.get_tensor(k) for k in f.keys() if k.startswith(TEXT_PREFIX)}
    if not sd:
        raise KeyError(f"{ckpt}: no {TEXT_PREFIX}* tensors")
    encoder.load_state_dict(sd, strict=True, assign=True)
    encoder = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return tokenizer, encoder


def collect_prompts(config_dir: str, config_name: str, overrides: list) -> tuple[list, dict]:
    from hydra import compose, initialize_config_dir

    from openwam.dataloader.registry import build_dataset

    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    dataset = build_dataset(cfg.dataloader)
    prompts = sorted(dataset.prompt_vocabulary())
    if not prompts or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError("dataset returned an empty prompt vocabulary or a non-string / empty prompt")
    return prompts, cfg


def _worker(rank: int, args, prompts: list, batches: list, tmpdir: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    from openwam.model.video_backbone.wan.text_embed_cache import encode_prompt_rows

    tokenizer, encoder = load_text_encoder(args.ckpt, args.tokenizer, device)
    out = {}
    for b in batches[rank :: args.num_gpus]:
        chunk = prompts[b * args.batch_size : (b + 1) * args.batch_size]
        rows = encode_prompt_rows(chunk, tokenizer=tokenizer, text_encoder=encoder, device=device,
                                  batch_size=args.batch_size)
        out[b] = [r.cpu() for r in rows]
    torch.save(out, os.path.join(tmpdir, f"rank{rank}.pt"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=str(REPO / "configs"))
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--override", action="append", default=[], help="hydra override, repeatable")
    ap.add_argument("--ckpt", default=None, help="default: training.finetune_ckpt_path")
    ap.add_argument("--tokenizer", default=None, help="default: <ckpt dir>/tokenizer/google/umt5-xxl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16, help="encode call shape; keep = training micro-batch")
    ap.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    args = ap.parse_args()

    t0 = time.time()
    prompts, cfg = collect_prompts(args.config_dir, args.config_name, args.override)
    args.ckpt = args.ckpt or str(cfg.training.finetune_ckpt_path)
    args.tokenizer = args.tokenizer or str(Path(args.ckpt).parent / "tokenizer/google/umt5-xxl")
    print(f"[text-cache] {len(prompts)} unique prompts from {args.config_name} ({time.time() - t0:.0f}s)", flush=True)

    t1 = time.time()
    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    args.num_gpus = max(1, min(args.num_gpus, n_batches))
    with tempfile.TemporaryDirectory(dir=str(Path(args.out).parent)) as tmp:
        mp.spawn(_worker, args=(args, prompts, list(range(n_batches)), tmp), nprocs=args.num_gpus, join=True)
        by_batch = {}
        for r in range(args.num_gpus):
            by_batch.update(torch.load(os.path.join(tmp, f"rank{r}.pt")))
    rows = [row for b in range(n_batches) for row in by_batch[b]]
    assert len(rows) == len(prompts)
    print(f"[text-cache] encoded on {args.num_gpus} GPU(s) in {time.time() - t1:.0f}s", flush=True)

    from openwam.model.video_backbone.wan.text_embed_cache import TextEmbedCache

    st = os.stat(args.ckpt)
    meta = {
        "encode_batch_size": args.batch_size,
        "source": {
            "config_name": args.config_name,
            "overrides": args.override,
            "ckpt": args.ckpt,
            "ckpt_size": st.st_size,
            "ckpt_mtime": int(st.st_mtime),
            "tokenizer": args.tokenizer,
            "prompts_sha256": hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "host": socket.gethostname(),
            "created_unix": int(time.time()),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    TextEmbedCache.save(args.out, prompts, rows, max_length=512, meta=meta)
    lens = [int(r.shape[0]) for r in rows]
    print(
        f"[text-cache] wrote {args.out}: {len(prompts)} prompts, {sum(lens)} rows "
        f"(len mean {sum(lens) / len(lens):.1f}, max {max(lens)}), "
        f"{os.path.getsize(args.out) / 2**30:.2f} GiB, total {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
