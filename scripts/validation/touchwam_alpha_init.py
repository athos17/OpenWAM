#!/usr/bin/env python3
"""Bounded-memory full Alpha tensor audit against the actual TouchWAM topology.

Builds the complete architecture on meta, validates every pretrained key/shape,
then assigns each tensor through load_state_dict(assign=True), compares its raw
bytes to the checkpoint, and releases it before proceeding. This proves every
V/A/proprio/preprocessor tensor can be reused bit-for-bit; it is deliberately
not a full resident-model forward or a production training run.
"""

import argparse
import ctypes
import hashlib
import json
import os
import struct
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors import safe_open

from openwam.model import build_architecture, resolve_architecture_config
from openwam.train.utils.ckpt_model_loader import merge_ckpt_model_cfg


def read_header(path):
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        length = struct.unpack("<Q", prefix)[0]
        raw = handle.read(length)
    return json.loads(raw), prefix + raw


def _drop_readonly_pages(tensor):
    """Release only fully covered file-backed pages after hashing a tensor."""
    page = os.sysconf("SC_PAGE_SIZE")
    start, stop = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
    first = (start + page - 1) // page * page
    last = stop // page * page
    mappings = Path("/proc/self/maps").read_text().splitlines()
    file_backed = any(".safetensors" in line and
                      int(line.split()[0].split("-")[0], 16) <= start and
                      int(line.split()[0].split("-")[1], 16) >= stop for line in mappings)
    if last > first and file_backed:
        # safetensors CPU tensors are read-only in this audit. DONTNEED discards
        # resident file pages; it neither writes the checkpoint nor drops system caches.
        result = ctypes.CDLL(None, use_errno=True).madvise(ctypes.c_void_p(first), ctypes.c_size_t(last-first), 4)
        if result:
            raise OSError(ctypes.get_errno(), "madvise on safetensors mapping failed")


def assign_and_hash(model, name, tensor, source_handle, offset, nbytes, *, chunk_bytes=8*1024**2, file_hash=None):
    result = model.load_state_dict({name: tensor}, strict=False, assign=True)
    if result.unexpected_keys or name in result.missing_keys:
        raise RuntimeError(f"Tensor assignment rejected: {name}")
    assigned = model.state_dict()[name]
    if assigned.is_meta or assigned.dtype != tensor.dtype or tuple(assigned.shape) != tuple(tensor.shape):
        raise RuntimeError(f"Tensor assignment changed metadata: {name}")
    raw = assigned.detach().reshape(-1).view(torch.uint8).numpy()
    view = memoryview(raw)
    target_hash, source_hash = hashlib.sha256(), hashlib.sha256()
    for start in range(0, nbytes, chunk_bytes):
        count = min(chunk_bytes, nbytes - start)
        source_handle.seek(offset + start)
        block = source_handle.read(count)
        if len(block) != count:
            raise EOFError(name)
        source_hash.update(block)
        if file_hash is not None:
            file_hash.update(block)
        target_hash.update(view[start:start+count])
    if target_hash.digest() != source_hash.digest():
        raise RuntimeError(f"Pretrained bits changed during assignment: {name}")
    digest = target_hash.hexdigest()
    _drop_readonly_pages(assigned)
    path, _, leaf = name.rpartition(".")
    parent = model.get_submodule(path) if path else model
    empty = torch.empty(assigned.shape, device="meta", dtype=assigned.dtype)
    if leaf in parent._parameters:
        parent._parameters[leaf] = torch.nn.Parameter(empty, requires_grad=parent._parameters[leaf].requires_grad)
    else:
        parent._buffers[leaf] = empty
    return digest


def run(checkpoint, output, config="train_touchwam_pretrain", *, header_only=False, live_cfg=None):
    torch.set_num_threads(1)
    checkpoint, output = Path(checkpoint).resolve(strict=True), Path(output)
    source_cfg = OmegaConf.load(checkpoint.parent / "config.yaml")
    if OmegaConf.select(source_cfg, "model.architecture.variant") != "joint_self_attn":
        raise ValueError("Expected an OpenWAM joint_self_attn checkpoint, never a random-action or TouchWAM source")
    root = Path(__file__).resolve().parents[2]
    if live_cfg is None:
        with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
            live = compose(config_name=config)
    else:
        live = OmegaConf.create(OmegaConf.to_container(live_cfg, resolve=True))
    checkpoint_stat = checkpoint.stat()
    model_cfg = merge_ckpt_model_cfg(source_cfg, live, str(checkpoint.parent))
    resolved = resolve_architecture_config(model_cfg)
    params = dict(resolved.params)
    params["video_backbone"] = OmegaConf.to_container(model_cfg.video_backbone, resolve=True)
    params["video_backbone"].update(_source=OmegaConf.to_container(model_cfg.video_backbone, resolve=True),
                                     _ckpt_dir=str(checkpoint.parent), _materialize_weights=False)
    started = time.time()
    with init_empty_weights(include_buffers=False):
        model = build_architecture(resolved.registry_name, params)
    audit = model.audit_openwam_checkpoint(str(checkpoint))
    header, header_bytes = read_header(checkpoint)
    weights = {k: v for k, v in header.items() if k != "__metadata__"}
    expected = model.state_dict()
    new = {k for k in expected if k.startswith(model.new_weight_prefixes)}
    if set(expected) - set(weights) != new:
        raise RuntimeError("Only tactile parameters may be new")
    model_config_hash = hashlib.sha256(json.dumps(OmegaConf.to_container(model_cfg, resolve=True),
                                                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = dict(schema="touchwam_alpha_bitwise_assignment_audit_v1", checkpoint=str(checkpoint),
                  checkpoint_stat={"size": checkpoint_stat.st_size, "mtime_ns": checkpoint_stat.st_mtime_ns},
                  resolved_model_config_sha256=model_config_hash,
                  checkpoint_bytes=checkpoint.stat().st_size, checkpoint_header_sha256=hashlib.sha256(header_bytes).hexdigest(),
                  architecture=resolved.registry_name, config=config, meta_topology_audit=audit,
                  validation_mode="header_only" if header_only else "bounded_per_tensor_assignment",
                  full_resident_model_forward=False, pretrained_tensor_hashes={}, new_tactile_keys=sorted(new),
                  parameter_bytes_by_group={group: sum(v["data_offsets"][1]-v["data_offsets"][0]
                      for k, v in weights.items() if k.startswith(group))
                      for group in ("video_backbone.", "action_backbone.", "proprio_encoder.")})
    if not header_only:
        ordered = sorted(weights, key=lambda k: weights[k]["data_offsets"][0])
        checkpoint_hash = hashlib.sha256(header_bytes)
        cursor = 0
        with safe_open(str(checkpoint), framework="pt", device="cpu") as source, open(checkpoint, "rb") as raw:
            for index, name in enumerate(ordered):
                begin, end = weights[name]["data_offsets"]
                if begin != cursor:
                    raise ValueError("Checkpoint data is not a contiguous safe tensor payload")
                cursor = end
                value = source.get_tensor(name)
                digest = assign_and_hash(model, name, value, raw, len(header_bytes)+begin, end-begin,
                                         file_hash=checkpoint_hash)
                report["pretrained_tensor_hashes"][name] = digest
                del value
                if index % 100 == 0:
                    print(json.dumps({"verified_tensors": index+1, "total_tensors": len(ordered)}), flush=True)
        if len(header_bytes) + cursor != checkpoint.stat().st_size:
            raise ValueError("Checkpoint contains unexpected trailing bytes")
        report["checkpoint_sha256"] = checkpoint_hash.hexdigest()
    final_stat = checkpoint.stat()
    if (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError("Checkpoint changed during validation; do not publish this audit")
    report.update(status="passed", verified_tensor_count=len(report["pretrained_tensor_hashes"]),
                  elapsed_s=time.time()-started, all_pretrained_bits_preserved=not header_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("pretrained_tensor_hashes", "new_tactile_keys")}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="train_touchwam_pretrain")
    p.add_argument("--header-only", action="store_true")
    args = p.parse_args()
    run(args.checkpoint, args.output, args.config, header_only=args.header_only)


if __name__ == "__main__":
    main()
