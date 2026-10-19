"""fp32 reference for the UMT5 text embedding (blocks streamed to GPU one at a time,
TF32 off), to judge whether bf16 dynamic padding is more or less accurate than the
bf16 512-padded baseline used in training."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_input_encoding import PROMPTS, diff, encode_text_dynamic, load_subset  # noqa: E402

from openwam.dataloader.transforms.multiview import format_prompt_for_inference  # noqa: E402
from openwam.model.video_backbone.wan import encode as wan_encode  # noqa: E402
from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer, WanTextEncoder  # noqa: E402


@torch.no_grad()
def fp32_forward(te_cpu, ids, mask, dev):
    x = te_cpu.token_embedding.to(dev).float()(ids)
    te_cpu.token_embedding.to("cpu")
    for blk in te_cpu.blocks:
        blk.to(dev).float()
        x = blk(x, mask, pos_bias=None)
        blk.to("cpu", torch.bfloat16)
    x = te_cpu.norm.to(dev).float()(x)
    te_cpu.norm.to("cpu", torch.bfloat16)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    prompts = [format_prompt_for_inference(p) for p in PROMPTS]
    tok = HuggingfaceTokenizer(name=args.tokenizer, seq_len=512, clean="whitespace")

    with torch.device("meta"):
        te = WanTextEncoder()
    te = te.to_empty(device="cpu")
    load_subset(te, args.ckpt, "video_backbone.text_encoder.", "cpu", torch.bfloat16)

    ids, mask = tok(prompts, return_mask=True, add_special_tokens=True, max_length=512,
                    padding="max_length", truncation=True)
    seq_lens = mask.sum(1)
    L = int(seq_lens.max())
    ref = fp32_forward(te, ids[:, :L].to(dev), mask[:, :L].to(dev), dev)
    for i, v in enumerate(seq_lens.tolist()):
        ref[i, v:] = 0
    ref = torch.nn.functional.pad(ref, (0, 0, 0, 512 - L))
    ref512 = fp32_forward(te, ids.to(dev), mask.to(dev), dev)
    for i, v in enumerate(seq_lens.tolist()):
        ref512[i, v:] = 0

    te.to(dev)
    with torch.no_grad():
        base, _ = wan_encode.encode_text(prompts, tokenizer=tok, text_encoder=te, device=dev)
        dyn, _ = encode_text_dynamic(prompts, tokenizer=tok, text_encoder=te, device=dev)
    res = {
        "fp32_dynamic_vs_fp32_pad512": diff(ref, ref512),
        "bf16_baseline_pad512_vs_fp32": diff(base, ref),
        "bf16_dynamic_vs_fp32": diff(dyn, ref),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
