"""Benchmark the text / video encoding stages of ``prepare_inputs`` on a single GPU.

Only the frozen ``video_backbone.text_encoder`` (UMT5-XXL, ~11.4 GB bf16) and
``video_backbone.vae`` (Wan2.2 VAE38, ~1.4 GB) are loaded from the checkpoint, so
this fits on a 24 GB card. The DiT / action backbone are never materialized.

Usage:
    python scripts/profiling/bench_input_encoding.py \
        --ckpt models/OpenWAM-Alpha-Sim-RoboDojo/checkpoint_step_60000.safetensors \
        --tokenizer models/OpenWAM-Alpha-Sim-RoboDojo/tokenizer/google/umt5-xxl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from openwam.dataloader.transforms.multiview import format_prompt_for_inference
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer, WanTextEncoder
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38

PROMPTS = [
    "Grab the narrow-necked bottle and lift it upward from the table.",
    "Push the center of the white and black colored bell using the left arm.",
    "Pick up the red block and place it on top of the blue block.",
    "Use the right arm to put the bread into the basket on the left side of the table.",
]


def load_subset(module: torch.nn.Module, ckpt: str, prefix: str, device, dtype) -> None:
    sd = {}
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix) :]] = f.get_tensor(k)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith(("mean", "std", "scale"))]
    assert not unexpected, unexpected[:5]
    assert not missing, missing[:5]
    module.to(device=device, dtype=dtype).requires_grad_(False)
    for m in module.modules():
        m.training = False  # same as ArchitectureBase.freeze_modules


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


@contextlib.contextmanager
def cudnn_flags(benchmark: bool, deterministic: bool):
    old = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old


def diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    return {
        "max_abs": (a - b).abs().max().item(),
        "rel_l2": ((a - b).norm() / b.norm().clamp_min(1e-12)).item(),
        "cos": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    }


# --------------------------------------------------------------------------- text


def encode_text_dynamic(prompts, *, tokenizer, text_encoder, device, max_length=512):
    """Same output as ``wan_encode.encode_text`` but only runs T5 on the longest real
    sequence in the batch, then zero-pads back to ``max_length`` (T5 is a
    bidirectional encoder with key-masking + relative position bias, so padding
    tokens never influence the real tokens)."""
    ids, mask = tokenizer(
        prompts, return_mask=True, add_special_tokens=True, max_length=max_length, padding="max_length", truncation=True
    )
    seq_lens = mask.gt(0).sum(dim=1).long()
    L = int(seq_lens.max())
    ctx = text_encoder(ids[:, :L].to(device), mask[:, :L].to(device))
    for i, v in enumerate(seq_lens):
        ctx[i, v:] = 0
    ctx = F.pad(ctx, (0, 0, 0, max_length - L))
    return ctx, seq_lens.to(device)


class PromptEmbedCache:
    """prompt -> (unpadded embedding on GPU). Exact: frozen T5 runs in eval mode."""

    def __init__(self, tokenizer, text_encoder, device, max_length=512):
        self.tok, self.enc, self.device, self.max_length = tokenizer, text_encoder, device, max_length
        self.store: dict[str, torch.Tensor] = {}

    def __call__(self, prompts):
        miss = list(dict.fromkeys(p for p in prompts if p not in self.store))
        if miss:
            ctx, lens = encode_text_dynamic(miss, tokenizer=self.tok, text_encoder=self.enc, device=self.device)
            for p, c, n in zip(miss, ctx, lens.tolist()):
                self.store[p] = c[:n].clone()
        seq_lens = torch.tensor([self.store[p].shape[0] for p in prompts], device=self.device)
        ctx = torch.zeros(len(prompts), self.max_length, self.store[prompts[0]].shape[-1], device=self.device,
                          dtype=self.store[prompts[0]].dtype)
        for i, p in enumerate(prompts):
            ctx[i, : self.store[p].shape[0]] = self.store[p]
        return ctx, seq_lens


# --------------------------------------------------------------------------- video


def make_frames(B, T, H, W, seed=0):
    rng = np.random.default_rng(seed)
    clips = []
    for _ in range(B):
        base = rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8)
        clip = []
        for t in range(T):
            img = Image.fromarray(np.roll(base, t, axis=1)).resize((W, H), Image.BILINEAR)
            clip.append(img)
        clips.append(clip)
    return clips


def preprocess_fast(clips, *, dtype, device):
    """List[List[PIL]] -> (B,3,T,H,W) in [-1,1]: one uint8 stack, one pinned H2D copy,
    normalization on GPU. Same op order as ``wan.preprocess.preprocess_image`` (cast to
    ``dtype`` first, then ``* (2/255) + (-1)`` in ``dtype``) so the result is bit-identical."""
    arr = np.stack([np.stack([np.asarray(im) for im in clip]) for clip in clips])  # B T H W C uint8
    x = torch.from_numpy(arr).pin_memory().to(device, non_blocking=True)
    x = x.permute(0, 4, 1, 2, 3).to(dtype)
    return (x * (2.0 / 255) + (-1.0)).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=9)  # num_frames=33, video_stride=4
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev, dt = torch.device("cuda"), torch.bfloat16
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "batch": args.batch}
    prompts = [format_prompt_for_inference(p) for p in PROMPTS[: args.batch]]
    while len(prompts) < args.batch:
        prompts.append(prompts[len(prompts) % len(PROMPTS)])

    # ---------------------------------------------------------------- text
    if not args.skip_text:
        tok = HuggingfaceTokenizer(name=args.tokenizer, seq_len=512, clean="whitespace")
        with torch.device("meta"):
            te = WanTextEncoder()
        te = te.to_empty(device="cpu")
        load_subset(te, args.ckpt, "video_backbone.text_encoder.", dev, dt)
        print(f"[text] loaded, mem={torch.cuda.memory_allocated() / 2**30:.1f} GiB")

        with torch.no_grad():
            t_base, (c_base, l_base) = timeit(
                lambda: wan_encode.encode_text(prompts, tokenizer=tok, text_encoder=te, device=dev))
            t_dyn, (c_dyn, l_dyn) = timeit(
                lambda: encode_text_dynamic(prompts, tokenizer=tok, text_encoder=te, device=dev))
            cache = PromptEmbedCache(tok, te, dev)
            cache(prompts)  # populate
            t_cache, (c_cache, l_cache) = timeit(lambda: cache(prompts))
            t_tok, _ = timeit(lambda: tok(prompts, return_mask=True, add_special_tokens=True, max_length=512,
                                          padding="max_length", truncation=True))
        assert torch.equal(l_base.cpu(), l_dyn.cpu()) and torch.equal(l_base.cpu(), l_cache.cpu())
        res["text"] = {
            "seq_lens": l_base.tolist(),
            "baseline_pad512_s": t_base,
            "tokenizer_only_s": t_tok,
            "dynamic_pad_s": t_dyn,
            "cache_hit_s": t_cache,
            "diff_dynamic_vs_base": diff(c_dyn, c_base),
            "diff_cache_vs_base": diff(c_cache, c_base),
            "cache_bytes_per_prompt": int(np.mean([v.numel() * v.element_size() for v in cache.store.values()])),
        }
        print(json.dumps(res["text"], indent=2))
        del te, cache, c_base, c_dyn, c_cache
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- video
    if not args.skip_video:
        vae = WanVideoVAE38()
        load_subset(vae, args.ckpt, "video_backbone.vae.", dev, dt)
        print(f"[video] loaded, mem={torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        clips = make_frames(args.batch, args.frames, args.height, args.width)

        def native_pre():
            return torch.cat([wan_encode.preprocess_video(c, encoder=None, dtype=dt, device=dev) for c in clips])

        v = {}
        with torch.no_grad():
            t_pre_native, x_native = timeit(native_pre)
            t_pre_fast, x_fast = timeit(lambda: preprocess_fast(clips, dtype=dt, device=dev))
            v["preprocess_native_s"] = t_pre_native
            v["preprocess_fast_s"] = t_pre_fast
            v["preprocess_bit_identical"] = bool(torch.equal(x_native, x_fast))

            def enc(x):
                return wan_encode.encode_video(x, vae=vae, encoder=None)

            # training default when project.seed is set: seed_everything() -> deterministic, no benchmark
            with cudnn_flags(benchmark=False, deterministic=True):
                t, z_ref = timeit(lambda: enc(x_native))
            v["vae_det_nobench_s"] = t
            with cudnn_flags(benchmark=False, deterministic=False):
                t, z = timeit(lambda: enc(x_native))
            v["vae_nondet_nobench_s"] = t
            v["diff_nondet_nobench"] = diff(z, z_ref)
            with cudnn_flags(benchmark=True, deterministic=False):
                t, z = timeit(lambda: enc(x_native), warmup=5)
            v["vae_bench_s"] = t
            v["diff_bench"] = diff(z, z_ref)
            with cudnn_flags(benchmark=True, deterministic=True):
                t, z = timeit(lambda: enc(x_native), warmup=5)
            v["vae_bench_det_s"] = t
            v["diff_bench_det"] = diff(z, z_ref)

            # channels_last_3d memory format
            def set_conv_format(fmt):
                for m in vae.model.modules():
                    if isinstance(m, torch.nn.Conv3d):
                        m.weight.data = m.weight.data.contiguous(memory_format=fmt)

            set_conv_format(torch.channels_last_3d)
            with cudnn_flags(benchmark=True, deterministic=False):
                t, z = timeit(lambda: enc(x_native.contiguous(memory_format=torch.channels_last_3d)), warmup=5)
            v["vae_bench_channels_last_s"] = t
            v["diff_bench_channels_last"] = diff(z, z_ref)
            set_conv_format(torch.contiguous_format)

            # per-sample loop (how much batching helps)
            with cudnn_flags(benchmark=True, deterministic=False):
                t, _ = timeit(lambda: torch.cat([enc(x_native[i : i + 1]) for i in range(x_native.shape[0])]))
            v["vae_bench_per_sample_loop_s"] = t

            if args.compile:
                vae.model.encoder = torch.compile(vae.model.encoder, dynamic=False)
                with cudnn_flags(benchmark=True, deterministic=False):
                    t, z = timeit(lambda: enc(x_native), warmup=5)
                v["vae_bench_compile_s"] = t
                v["diff_bench_compile"] = diff(z, z_ref)

        v["latent_shape"] = list(z_ref.shape)
        v["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        res["video"] = v
        print(json.dumps(v, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
