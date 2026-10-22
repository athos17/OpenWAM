"""Prompt-keyed UMT5 embedding cache for training (frozen text encoder).

The online training path (``wan.encode.encode_text``) returns a ``(B, 512, D)``
context with every position past ``seq_len`` zeroed, so storing only the valid
rows of each prompt is lossless; :meth:`TextEmbedCache.lookup` rebuilds the padded
tensor with a single gather.

File format (safetensors)::

    embeds   (sum_L, D) bf16   valid rows of every prompt, packed in prompt order
    offsets  (N + 1,)   int64  prompt i owns rows offsets[i]:offsets[i+1]
    metadata["openwam_text_embed_cache"] = json {version, prompts, max_length,
                                                 encode_batch_size, source}

Built by ``scripts/precompute/build_text_embed_cache.py`` with the training call
shape (``encode_batch_size`` prompts per call, padded to ``max_length``), which
makes cached rows bitwise identical to online encoding of a same-size batch.
"""

from __future__ import annotations

import json
from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor

FORMAT_KEY = "openwam_text_embed_cache"
FORMAT_VERSION = 1


class TextEmbedCache:
    def __init__(
        self,
        prompts: Sequence[str],
        embeds: Tensor,
        offsets: Tensor,
        *,
        max_length: int = 512,
        meta: Optional[dict] = None,
    ):
        if len(set(prompts)) != len(prompts):
            raise ValueError("TextEmbedCache: duplicate prompts")
        if offsets.numel() != len(prompts) + 1 or int(offsets[-1]) != embeds.shape[0]:
            raise ValueError("TextEmbedCache: offsets do not match prompts / embeds")
        self.prompts = list(prompts)
        self.max_length = int(max_length)
        self.meta = dict(meta or {})
        # One trailing zero row: every padded position gathers it.
        self._rows = torch.cat([embeds, embeds.new_zeros(1, embeds.shape[1])], dim=0)
        self._zero_row = embeds.shape[0]
        starts, lens = offsets[:-1].tolist(), (offsets[1:] - offsets[:-1]).tolist()
        if max(lens) > self.max_length:
            raise ValueError(f"TextEmbedCache: a prompt has more than max_length={self.max_length} rows")
        self._index = {p: (s, n) for p, s, n in zip(self.prompts, starts, lens)}
        # Online-fallback entries (prompts first seen during training).
        self._extra: dict[str, Tensor] = {}

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> "TextEmbedCache":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            raw = (f.metadata() or {}).get(FORMAT_KEY)
            if raw is None:
                raise ValueError(f"{path}: not an OpenWAM text embed cache (missing {FORMAT_KEY} metadata)")
            meta = json.loads(raw)
            if int(meta.get("version", -1)) != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported cache version {meta.get('version')}")
            embeds = f.get_tensor("embeds")
            offsets = f.get_tensor("offsets")
        prompts = meta.pop("prompts")
        return cls(prompts, embeds.to(device), offsets, max_length=meta["max_length"], meta=meta)

    @staticmethod
    def save(path: str, prompts: Sequence[str], rows: Sequence[Tensor], *, max_length: int, meta: dict) -> None:
        from safetensors.torch import save_file

        if len(prompts) != len(rows):
            raise ValueError("save: prompts / rows length mismatch")
        lens = torch.tensor([0] + [int(r.shape[0]) for r in rows], dtype=torch.int64)
        payload = dict(meta, version=FORMAT_VERSION, prompts=list(prompts), max_length=int(max_length))
        save_file(
            {"embeds": torch.cat([r.cpu() for r in rows], dim=0).contiguous(), "offsets": lens.cumsum(0)},
            path,
            metadata={FORMAT_KEY: json.dumps(payload, ensure_ascii=False)},
        )

    # ------------------------------------------------------------------ query

    def __len__(self) -> int:
        return len(self._index) + len(self._extra)

    def __contains__(self, prompt: str) -> bool:
        return prompt in self._index or prompt in self._extra

    @property
    def device(self) -> torch.device:
        return self._rows.device

    def missing(self, prompts: Iterable[str]) -> list:
        """Unique prompts (first-seen order) that are not cached."""
        return [p for p in dict.fromkeys(prompts) if p not in self]

    def add(self, prompt: str, row: Tensor) -> None:
        self._extra[prompt] = row.to(device=self._rows.device, dtype=self._rows.dtype)

    def rows(self, prompt: str) -> Tensor:
        if prompt in self._extra:
            return self._extra[prompt]
        start, n = self._index[prompt]
        return self._rows[start : start + n]

    def lookup(self, prompts: Sequence[str]) -> Tuple[Tensor, Tensor]:
        """``(context (B, max_length, D), seq_lens (B,))`` exactly as ``encode_text`` returns."""
        missing = self.missing(prompts)
        if missing:
            raise KeyError(f"{len(missing)} prompt(s) not in text embed cache, e.g. {missing[0]!r}")
        device = self._rows.device
        seq_lens = torch.tensor([self.rows(p).shape[0] for p in prompts], dtype=torch.long)
        if any(p in self._extra for p in prompts):
            context = self._rows.new_zeros(len(prompts), self.max_length, self._rows.shape[1])
            for i, p in enumerate(prompts):
                context[i, : int(seq_lens[i])] = self.rows(p)
            return context, seq_lens.to(device)
        pos = torch.arange(self.max_length)
        starts = torch.tensor([self._index[p][0] for p in prompts], dtype=torch.long)
        idx = torch.where(pos[None, :] < seq_lens[:, None], starts[:, None] + pos[None, :], self._zero_row)
        return self._rows[idx.to(device)], seq_lens.to(device)


@torch.no_grad()
def encode_prompt_rows(prompts: Sequence[str], *, tokenizer, text_encoder, device, batch_size: int) -> list:
    """Valid rows of each prompt, encoded ``batch_size`` at a time through the online
    ``encode_text`` call (padding to 512). The last batch is padded with repeats of
    its final prompt so every call has the training shape."""
    from openwam.model.video_backbone.wan import encode as wan_encode

    rows = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i : i + batch_size])
        n_real = len(chunk)
        chunk += [chunk[-1]] * (batch_size - n_real)
        context, seq_lens = wan_encode.encode_text(chunk, tokenizer=tokenizer, text_encoder=text_encoder, device=device)
        rows.extend(context[j, : int(seq_lens[j])].clone() for j in range(n_real))
    return rows
