"""Offline text-embedding / VAE-latent caches for frozen-encoder outputs.

The caches hold exactly what the online training path would hand the DiT:

* ``text``   — UMT5 ``context`` rows (bf16) produced by ``wan.encode.encode_text``
  with the training call shape (B=48, ``padding="max_length"`` to 512). Only the
  first ``max_tokens`` positions are stored: the online path zeroes every
  position past ``seq_len`` (``context[i, v:] = 0``), so trimming is lossless and
  the reader zero-pads back to 512.
* ``latent`` — Wan VAE ``input_latents`` rows (bf16) produced by
  ``wan.encode.preprocess_video`` + ``wan.encode.encode_video`` with B=48.

Layout of one cache directory::

    meta.json              signature + shard table (rows, bytes, sha256)
    VERIFIED-<host>.json   written by ``verify_cache`` after a full sha256 pass
    windows.npy            latent only: (N, 3) int32 (sub_dataset, episode, start)
    prompts.json           text only: row -> prompt string
    seq_lens.npy           text only: (N,) int32
    rows-XX-of-YY.bin      raw little-endian bf16 bits, C order

Readers open the shard memmaps lazily per process (DataLoader workers fork).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

FORMAT_VERSION = 1
TEXT_MAX_LENGTH = 512  # matches wan.encode.encode_text


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256_file(path, chunk: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def shard_name(index: int, count: int) -> str:
    return f"rows-{index:02d}-of-{count:02d}.bin"


def shard_bounds(num_rows: int, index: int, count: int) -> tuple[int, int]:
    """Contiguous, near-equal row range of shard ``index``."""
    return num_rows * index // count, num_rows * (index + 1) // count


MANIFEST_KEYS = (
    "format_version",
    "kind",
    "signature",
    "num_rows",
    "row_shape",
    "dtype",
    "shards",
    "side_files",
    "encoder_fingerprint",
    "tokenizer_sha256",
)


def manifest_digest(meta: dict) -> str:
    """Digest of everything a reader trusts: signature, geometry, weights and file hashes."""
    return digest({k: meta.get(k) for k in MANIFEST_KEYS})


# ----------------------------------------------------------------------------
# Signatures (what the cached values depend on)
# ----------------------------------------------------------------------------


def module_fingerprint(module: torch.nn.Module, extra: Optional[dict] = None) -> str:
    """sha256 over every parameter/buffer (name, dtype, shape, raw bytes)."""
    h = hashlib.sha256()
    items = sorted(module.state_dict().items())
    for name, value in (extra or {}).items():
        items.append((f"__extra__.{name}", value))
    for name, tensor in items:
        t = tensor.detach().to("cpu").contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def text_encoder_fingerprint(video_backbone) -> str:
    return module_fingerprint(video_backbone.text_encoder)


def vae_fingerprint(video_backbone) -> str:
    vae = video_backbone.vae
    extra = {k: getattr(vae, k) for k in ("mean", "std") if isinstance(getattr(vae, k, None), torch.Tensor)}
    return module_fingerprint(vae, extra)


def tree_digest(root, suffixes: Optional[Iterable[str]] = None) -> str:
    """Content digest of every file under ``root`` (relative path + sha256)."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and (suffixes is None or p.suffix in suffixes))
    return digest([(str(p.relative_to(root)), sha256_file(p)) for p in files])


def episode_inventory(sub_datasets) -> list:
    """(sub-dataset, relative episode path, size) for every episode, in index order."""
    out = []
    for ds_idx, ds in enumerate(sub_datasets):
        root = Path(ds.data_root)
        for path in ds._episode_files:
            out.append([ds_idx, str(Path(path).relative_to(root.parent.parent)), os.path.getsize(path)])
    return out


def instructions_digest(sub_datasets) -> str:
    """Digest of the loaded instruction tables (the only prompt source)."""
    return digest([[ds_idx, ds._instructions] for ds_idx, ds in enumerate(sub_datasets)])


# Datasets other than MultiTaskRoboTwinDataset opt in by implementing the
# ``encode_cache_*`` protocol (see LeRobotV3Reader); the functions below dispatch.


def latent_signature(dataset) -> dict:
    """Everything that determines the VAE input pixels of every window."""
    if hasattr(dataset, "encode_cache_latent_signature"):
        return dataset.encode_cache_latent_signature()
    subs = dataset._sub_datasets
    first = subs[0]
    return {
        "dataset": "MultiTaskRoboTwinDataset",
        "tasks": [ds.data_root.rsplit("/", 3)[-3] + "/" + ds.variant for ds in subs],
        "embodiment": first.embodiment,
        "num_frames": first.num_frames,
        "video_sample_indices": list(first._video_sample_indices),
        "height": first.height,
        "width": first.width,
        "multiview": first.multiview,
        "camera_layout": list(first.camera_layout or []),
        "target_camera": first.target_camera,
        "window_stride": first.window_stride,
        "split": first.split,
        "color_jitter": any(ds._color_jitter is not None for ds in subs),
        "episode_inventory_sha256": digest(episode_inventory(subs)),
        "num_windows": len(dataset),
    }


def text_signature(dataset) -> dict:
    if hasattr(dataset, "encode_cache_text_signature"):
        return dataset.encode_cache_text_signature()
    subs = dataset._sub_datasets
    return {
        "dataset": "MultiTaskRoboTwinDataset",
        "tasks": [ds.data_root.rsplit("/", 3)[-3] + "/" + ds.variant for ds in subs],
        "split": subs[0].split,
        "instructions_sha256": instructions_digest(subs),
        "max_length": TEXT_MAX_LENGTH,
    }


def window_table(dataset) -> np.ndarray:
    if hasattr(dataset, "encode_cache_window_table"):
        return dataset.encode_cache_window_table()
    rows = [(d, e, s) for d, ds in enumerate(dataset._sub_datasets) for e, s in ds._window_index]
    return np.asarray(rows, dtype=np.int32).reshape(-1, 3)


def candidate_prompt_list(dataset) -> list:
    """Every prompt string ``_get_prompt`` can return for this dataset, sorted."""
    if hasattr(dataset, "encode_cache_prompts"):
        return dataset.encode_cache_prompts()
    from openwam.dataloader.robotwin import candidate_prompts

    out = set()
    for ds in dataset._sub_datasets:
        for path in ds._episode_files:
            out.update(candidate_prompts(ds._instructions, path, ds.split, ds.task_name))
    return sorted(out)


# ----------------------------------------------------------------------------
# Writing / verification
# ----------------------------------------------------------------------------


# Shard-record fields every shard (on every node) must agree on.
SHARD_CONSENSUS_KEYS = ("signature_sha256", "encoder_fingerprint", "tokenizer_sha256", "row_shape", "num_rows")


def write_shard_record(cache_dir, index: int, count: int, row_start: int, row_end: int, path, extra=None) -> dict:
    record = {
        "index": index,
        "file": Path(path).name,
        "row_start": row_start,
        "row_end": row_end,
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
        "host": socket.gethostname(),
        **(extra or {}),
    }
    (Path(cache_dir) / f"shard-{index:02d}-of-{count:02d}.json").write_text(json.dumps(record, indent=2))
    return record


def load_shard_records(cache_dir, count: int) -> list:
    """All shard records, checked to agree on signature, weights and geometry."""
    cache_dir = Path(cache_dir)
    records = [json.loads((cache_dir / f"shard-{i:02d}-of-{count:02d}.json").read_text()) for i in range(count)]
    for key in SHARD_CONSENSUS_KEYS:
        values = {json.dumps(r.get(key), sort_keys=True) for r in records}
        if len(values) != 1:
            raise ValueError(f"shards disagree on {key}: {sorted(values)}")
    return records


def finalize_meta(
    cache_dir, *, kind: str, signature: dict, num_rows: int, row_shape, count: int, side_files: dict, extra=None
) -> dict:
    cache_dir = Path(cache_dir)
    records = load_shard_records(cache_dir, count)
    if records[0]["signature_sha256"] != digest(signature):
        raise ValueError("shard signature differs from the finalizing host's dataset signature")
    shards = []
    for i, record in enumerate(records):
        expected = shard_bounds(num_rows, i, count)
        if (record["row_start"], record["row_end"]) != expected:
            raise ValueError(f"shard {i} rows {record['row_start']}..{record['row_end']} != {expected}")
        shards.append({k: record[k] for k in ("index", "file", "row_start", "row_end", "bytes", "sha256", "host")})
    row_bytes = int(np.prod(row_shape)) * 2
    for record in shards:
        if record["bytes"] != (record["row_end"] - record["row_start"]) * row_bytes:
            raise ValueError(f"shard {record['file']} has {record['bytes']} bytes, wrong for its rows")
    meta = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "signature": signature,
        "signature_sha256": digest(signature),
        "num_rows": num_rows,
        "row_shape": list(row_shape),
        "dtype": "bfloat16",
        "shards": shards,
        "side_files": side_files,
        "encoder_fingerprint": records[0]["encoder_fingerprint"],
        "tokenizer_sha256": records[0].get("tokenizer_sha256"),
        **(extra or {}),
    }
    meta["manifest_sha256"] = manifest_digest(meta)
    tmp = cache_dir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    tmp.replace(cache_dir / "meta.json")
    return meta


def verify_cache(cache_dir, workers: int = 16) -> dict:
    """Full sha256 pass over every shard + side file; writes VERIFIED-<host>.json."""
    from concurrent.futures import ThreadPoolExecutor

    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    if manifest_digest(meta) != meta["manifest_sha256"]:
        raise ValueError("meta.json manifest digest mismatch")
    files = [(r["file"], r["sha256"]) for r in meta["shards"]] + list(meta.get("side_files", {}).items())
    with ThreadPoolExecutor(workers) as pool:
        actual = list(pool.map(lambda f: sha256_file(cache_dir / f[0]), files))
    bad = [name for (name, want), got in zip(files, actual) if want != got]
    if bad:
        raise ValueError(f"sha256 mismatch in {cache_dir}: {bad}")
    stamp = {"manifest_sha256": meta["manifest_sha256"], "host": socket.gethostname(), "files": len(files)}
    (cache_dir / f"VERIFIED-{socket.gethostname()}.json").write_text(json.dumps(stamp, indent=2))
    return stamp


def load_verified_meta(cache_dir, kind: str) -> dict:
    """meta.json, refusing caches that were not fully verified on THIS host."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    if meta.get("format_version") != FORMAT_VERSION or meta.get("kind") != kind:
        raise ValueError(f"{cache_dir}: not a v{FORMAT_VERSION} {kind} cache")
    if manifest_digest(meta) != meta["manifest_sha256"]:
        raise ValueError(f"{cache_dir}: meta.json manifest digest mismatch")
    stamp_path = cache_dir / f"VERIFIED-{socket.gethostname()}.json"
    if not stamp_path.is_file():
        raise ValueError(f"{cache_dir}: no {stamp_path.name}; run verify on this host first")
    if json.loads(stamp_path.read_text()).get("manifest_sha256") != meta["manifest_sha256"]:
        raise ValueError(f"{cache_dir}: {stamp_path.name} was written for a different manifest")
    for record in meta["shards"]:
        if os.path.getsize(cache_dir / record["file"]) != record["bytes"]:
            raise ValueError(f"{cache_dir}: {record['file']} size changed since verification")
    return meta


def check_signature(meta: dict, expected: dict, what: str) -> None:
    if meta["signature"] != expected:
        diff = sorted(k for k in set(meta["signature"]) | set(expected) if meta["signature"].get(k) != expected.get(k))
        raise ValueError(f"{what} cache does not match this dataset configuration; differing keys: {diff}")


# ----------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------


class _ShardedRows:
    """Row-addressable view over the shard memmaps, opened lazily per process."""

    def __init__(self, cache_dir, meta: dict):
        self._dir = Path(cache_dir)
        self._shards = meta["shards"]
        self._starts = np.asarray([r["row_start"] for r in self._shards], dtype=np.int64)
        self._row_shape = tuple(meta["row_shape"])
        self._pid = None
        self._maps = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_pid"], state["_maps"] = None, None
        return state

    def _open(self):
        if self._pid != os.getpid():
            # Empty shards (more shards than rows, e.g. one prompt over 8 GPUs) map to
            # None; searchsorted below always resolves to the last shard starting at or
            # before the row, which is the non-empty one holding it.
            self._maps = [
                np.memmap(
                    self._dir / r["file"],
                    dtype=np.int16,
                    mode="r",
                    shape=(r["row_end"] - r["row_start"], *self._row_shape),
                )
                if r["row_end"] > r["row_start"]
                else None
                for r in self._shards
            ]
            self._pid = os.getpid()
        return self._maps

    def row(self, index: int, prefix: Optional[int] = None) -> torch.Tensor:
        maps = self._open()
        s = int(np.searchsorted(self._starts, index, side="right") - 1)
        src = maps[s][index - self._starts[s]]
        if prefix is not None:
            src = src[:prefix]
        # np.array copies out of the mapping so the tensor owns pinnable memory.
        return torch.from_numpy(np.array(src)).view(torch.bfloat16)


class LatentCache:
    """Training-side reader: global window index -> (C, T, H, W) bf16 latent."""

    def __init__(self, cache_dir, dataset):
        meta = load_verified_meta(cache_dir, "latent")
        check_signature(meta, latent_signature(dataset), "latent")
        windows = np.load(Path(cache_dir) / "windows.npy")
        if not np.array_equal(windows, window_table(dataset)):
            raise ValueError(f"{cache_dir}: cached window table differs from this dataset's enumeration")
        self.meta = meta
        self._rows = _ShardedRows(cache_dir, meta)

    def get(self, index: int) -> torch.Tensor:
        return self._rows.row(index)


class TextCache:
    """Training-side reader: prompt string -> ((seq_len, dim) bf16 context, seq_len)."""

    def __init__(self, cache_dir, dataset):
        meta = load_verified_meta(cache_dir, "text")
        check_signature(meta, text_signature(dataset), "text")
        prompts = json.loads((Path(cache_dir) / "prompts.json").read_text())
        self._row = {p: i for i, p in enumerate(prompts)}
        missing = [p for p in candidate_prompt_list(dataset) if p not in self._row]
        if missing:
            raise ValueError(f"{cache_dir}: {len(missing)} reachable prompts are not cached, e.g. {missing[:3]}")
        self._seq_lens = np.load(Path(cache_dir) / "seq_lens.npy")
        self.meta = meta
        self._rows = _ShardedRows(cache_dir, meta)

    def get(self, prompt: str) -> tuple[torch.Tensor, int]:
        row = self._row.get(prompt)
        if row is None:
            raise KeyError(f"prompt not in text cache: {prompt!r}")
        n = int(self._seq_lens[row])
        return self._rows.row(row, prefix=n), n


def verify_encoder_weights(cfg, architecture) -> None:
    """Refuse a training run whose frozen encoders differ from the ones that built the caches.

    Runs on LOCAL_RANK 0 of every node (each node loads its own checkpoint copy).
    """
    dl = cfg.get("dataloader", {})
    caches = [("text", dl.get("text_cache_dir")), ("latent", dl.get("latent_cache_dir"))]
    caches = [(k, d) for k, d in caches if d]
    if not caches or int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return
    vb = architecture.video_backbone
    for kind, cache_dir in caches:
        meta = json.loads((Path(cache_dir) / "meta.json").read_text())
        actual = text_encoder_fingerprint(vb) if kind == "text" else vae_fingerprint(vb)
        if actual != meta["encoder_fingerprint"]:
            raise ValueError(f"{kind} cache {cache_dir} was built with different encoder weights")
        if kind == "text":
            tok_dir = Path(cfg.training.finetune_ckpt_path) / "tokenizer"
            if tree_digest(tok_dir) != meta["tokenizer_sha256"]:
                raise ValueError(f"text cache {cache_dir} was built with a different tokenizer")


def offload_cached_encoders(cfg, architecture) -> list:
    """Move encoders whose outputs come from caches off the GPU. Returns moved names."""
    dl = cfg.get("dataloader", {}) if cfg is not None else {}
    moved = []
    if dl.get("text_cache_dir"):
        architecture.video_backbone.text_encoder.to("cpu")
        moved.append("text_encoder")
    if dl.get("latent_cache_dir"):
        architecture.video_backbone.vae.to("cpu")
        moved.append("vae")
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved
