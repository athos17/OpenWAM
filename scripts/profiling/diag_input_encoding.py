"""Diagnostics for why prepare_inputs encoding is slow inside real training.

1. noise:   batch-composition noise of the baseline (B=1 vs B=4, both 512-padded)
            vs. the dynamic-padding difference — tells whether dynamic padding is
            within noise the baseline already has.
2. launch:  CPU enqueue time vs. total time (launch-bound => sensitive to CPU contention).
3. memcap:  run under a tight per-process memory cap (simulating the DiT+optimizer
            state left by training) and count allocator retries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_input_encoding import (  # noqa: E402
    PROMPTS,
    diff,
    encode_text_dynamic,
    load_subset,
    make_frames,
    preprocess_fast,
)

from openwam.dataloader.transforms.multiview import format_prompt_for_inference  # noqa: E402
from openwam.model.video_backbone.wan import encode as wan_encode  # noqa: E402
from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer, WanTextEncoder  # noqa: E402
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38  # noqa: E402


def enqueue_vs_total(fn, iters=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    enq, tot = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        enq.append(t1 - t0)
        tot.append(t2 - t0)
    return float(np.median(enq)), float(np.median(tot))


def count_kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    evs = [e for e in prof.events() if e.device_type.name == "CUDA"]
    return len(evs), sum(e.device_time for e in evs) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--mem-cap-gib", type=float, default=None,
                    help="cap this process to N GiB total to emulate a nearly full training GPU")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev, dt = torch.device("cuda"), torch.bfloat16
    res = {}
    prompts = [format_prompt_for_inference(p) for p in PROMPTS]

    tok = HuggingfaceTokenizer(name=args.tokenizer, seq_len=512, clean="whitespace")
    with torch.device("meta"):
        te = WanTextEncoder()
    te = te.to_empty(device="cpu")
    load_subset(te, args.ckpt, "video_backbone.text_encoder.", dev, dt)
    vae = WanVideoVAE38()
    load_subset(vae, args.ckpt, "video_backbone.vae.", dev, dt)
    clips = make_frames(4, 9, 384, 320)
    x = preprocess_fast(clips, dtype=dt, device=dev)

    def text_base():
        return wan_encode.encode_text(prompts, tokenizer=tok, text_encoder=te, device=dev)

    def text_dyn():
        return encode_text_dynamic(prompts, tokenizer=tok, text_encoder=te, device=dev)

    def video():
        return wan_encode.encode_video(x, vae=vae, encoder=None)

    with torch.no_grad():
        if args.mem_cap_gib is None:
            # ---- 1. noise
            c4, _ = text_base()
            c1 = torch.cat([wan_encode.encode_text([p], tokenizer=tok, text_encoder=te, device=dev)[0] for p in prompts])
            cd, _ = text_dyn()
            res["noise"] = {
                "baseline_B1_vs_B4": diff(c1, c4),
                "dynamic_vs_baseline_B4": diff(cd, c4),
                "dynamic_vs_baseline_B1": diff(cd, c1),
            }
            print(json.dumps(res["noise"], indent=2))

            # ---- 2. launch-bound?
            r = {}
            for name, fn in [("text_base", text_base), ("text_dyn", text_dyn), ("vae", video)]:
                e, t = enqueue_vs_total(fn)
                n, gpu_s = count_kernels(fn)
                r[name] = {"cpu_enqueue_s": e, "total_s": t, "num_kernels": n, "gpu_kernel_sum_s": gpu_s}
            res["launch"] = r
            print(json.dumps(r, indent=2))

            # fp32 VAE for reference
            vae.model.float()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            xf = x.float()
            e, t = enqueue_vs_total(lambda: wan_encode.encode_video(xf, vae=vae, encoder=None))
            res["vae_fp32_s"] = t
            torch.backends.cudnn.allow_tf32 = True
            e, t = enqueue_vs_total(lambda: wan_encode.encode_video(xf, vae=vae, encoder=None))
            res["vae_fp32_tf32_s"] = t
            vae.model.to(dt)
            print("vae fp32:", res["vae_fp32_s"], "tf32:", res["vae_fp32_tf32_s"])
        else:
            # ---- 3. memory pressure: fill the GPU up to the cap with a "training state" blob
            free, total = torch.cuda.mem_get_info()
            used_now = torch.cuda.memory_reserved()
            cap = args.mem_cap_gib * 2**30
            # fragment: many medium blocks, free every other one
            blob, keep = [], []
            budget = cap - used_now - 1.5 * 2**30
            chunk = 96 * 2**20
            n = int(budget // chunk)
            for i in range(n):
                blob.append(torch.empty(chunk // 2, dtype=torch.bfloat16, device=dev))
            for i in range(0, n, 2):
                keep.append(blob[i])
            del blob
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))
            torch.cuda.reset_peak_memory_stats()
            r = {}
            for name, fn in [("text_base", text_base), ("text_dyn", text_dyn), ("vae", video)]:
                before = torch.cuda.memory_stats().get("num_alloc_retries", 0)
                try:
                    e, t = enqueue_vs_total(fn)
                    r[name] = {"total_s": t, "alloc_retries_per_call": (torch.cuda.memory_stats()
                               .get("num_alloc_retries", 0) - before) / 7}
                except torch.OutOfMemoryError as err:
                    r[name] = {"oom": str(err).splitlines()[0]}
            r["reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            res["memcap"] = {"cap_gib": args.mem_cap_gib, **r}
            print(json.dumps(res["memcap"], indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
