"""Stage a new batch-size phase without copying or mutating trained tensors.

Supported migration: same world size, accumulation=1, epoch zero, during
linear warmup. Adam states remain byte-identical hardlinks. Phase step starts
at zero; provenance retains the original update count. Source bucket prefixes
are removed before regrouping, preserving all remaining virtual windows.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openwam.dataloader.modality_sampler import GlobalModalitySampler
from openwam.train.utils.training_utils import build_cosine_scheduler


def make_plan(contract, meta, scheduler, target_batch):
    layout = contract["training_layout"]
    if (contract["sampling_algorithm"] != "full_passes_random_partial_v1"
            or layout["batching"] != "global_modality_codec_observation_v4"
            or layout.get("rebatch") or layout["gradient_accumulation"] != 1
            or meta["epoch"] != 0 or meta["global_step"] != meta["opt_step"]
            or not meta.get("optimizer_boundary")):
        raise ValueError("Unsupported source continuation layout")
    world = int(layout["world_size"])
    old_batch = int(layout["microbatch"])
    if target_batch <= old_batch or target_batch % old_batch:
        raise ValueError("Target batch must be an integer multiple of the source batch")
    ratio = target_batch // old_batch
    names = contract["source_order"]
    sources = contract["sources"]
    total_real = sum(s["windows"] for s in sources.values())
    counts = {name: max(1, round(total_real * contract["weights"][name]))
              if sources[name]["windows"] and contract["weights"][name] > 0 else 0 for name in names}
    if sum(counts.values()) != contract["virtual_windows"]:
        raise ValueError("Source virtual-window counts do not reproduce the saved contract")
    modes = [sources[name]["training_modality"] for name in names]
    dataset = SimpleNamespace(names=names, _seed=contract["seed"], source_modalities=modes,
        source_tactile_codecs=[sources[name].get("tactile_codec", "tactile_intensity_v1")
                               if "tactile" in mode else None for name, mode in zip(names, modes)],
        source_tactile_observation_shapes=[sources[name].get("tactile_observation_shape") for name in names],
        dataset_sample_counts=lambda: counts)
    sampler = GlobalModalitySampler(dataset, old_batch, world)
    bucket_counts = sampler._counts()
    block = old_batch * world
    schedule = np.concatenate([np.full(math.ceil(n / block), i, dtype=np.int64)
                               for i, n in enumerate(bucket_counts)])
    if len(schedule) != meta["batches_per_epoch"]:
        raise ValueError("Source epoch length does not reproduce the checkpoint")
    step = int(meta["global_step"])
    if not 0 < step < len(schedule):
        raise ValueError("Migration requires an unfinished source epoch")
    np.random.RandomState(dataset._seed ^ 0x4D4F44).shuffle(schedule)
    used = np.bincount(schedule[:step], minlength=len(bucket_counts))
    consumed = [min(n, int(k) * block) for n, k in zip(bucket_counts, used)]
    target_block = target_batch * world
    remaining = sum(math.ceil((n - c) / target_block) for n, c in zip(bucket_counts, consumed))
    warmup, cosine = scheduler["_schedulers"]
    milestone, = scheduler["_milestones"]
    tick = scheduler["last_epoch"]
    if tick != step * world or not 0 < tick < milestone:
        raise ValueError("Only in-warmup source schedulers with the expected Accelerate tick count are supported")
    if (milestone - tick) % ratio or cosine["T_max"] % ratio:
        raise ValueError("Schedule cannot be migrated exactly at this batch ratio")
    base_lrs = warmup["base_lrs"]
    if len(set(base_lrs)) != 1 or len(set(scheduler["_last_lr"])) != 1:
        raise ValueError("Migration currently requires equal group learning rates")
    return dict(schema_version=1, from_step=step, world_size=world, seed=dataset._seed,
        source_global_batch=block, target_global_batch=target_block,
        source_total_steps=len(schedule), remaining_steps=remaining,
        bucket_counts={str(k): n for k, n in zip(sampler._modalities, bucket_counts)},
        consumed_by_bucket={str(k): n for k, n in zip(sampler._modalities, consumed)},
        consumed_unique_virtual_windows=sum(consumed), source_processed_samples=step * block,
        remaining_unique_virtual_windows=sum(bucket_counts)-sum(consumed),
        remaining_warmup_ticks=(milestone-tick)//ratio, cosine_ticks=cosine["T_max"]//ratio,
        start_lr_factor=scheduler["_last_lr"][0]/base_lrs[0], peak_lr=base_lrs[0],
        lr_min_ratio=cosine["eta_min"]/base_lrs[0])


def stage(source, destination, step, target_batch=48, workers=8):
    source, destination = Path(source), Path(destination)
    state = source/f"accel_state_step_{step}"
    meta = json.loads((state/"trainer_state.json").read_text())
    if meta["global_step"] != step:
        raise ValueError("Checkpoint identity mismatch")
    contract_bytes = (source/"training_data_contract.json").read_bytes()
    contract = json.loads(contract_bytes)
    saved_scheduler = torch.load(state/"scheduler.bin", map_location="cpu", weights_only=False)
    plan = make_plan(contract, meta, saved_scheduler, target_batch)
    plan.update(source_run=str(source), source_contract_sha256=hashlib.sha256(contract_bytes).hexdigest())
    shards = list((state/"pytorch_model").glob("*optim_states.pt"))
    if len(shards) != plan["world_size"] or len(list(state.glob("random_states_*.pkl"))) != plan["world_size"]:
        raise ValueError("Checkpoint does not contain all optimizer/RNG shards")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    cfg = OmegaConf.load(source/"config.yaml")
    t = cfg.training
    t.update(batch_size=target_batch, dataset_num_workers=workers, use_gradient_checkpointing=True,
        gradient_checkpointing_layers=list(range(20)), gradient_checkpointing_layers_video_only=None,
        use_gradient_checkpointing_offload=False, num_epochs=1, max_steps=None, stop_after_steps=None,
        save_steps=500, save_full_states_for_resume=True, finetune_ckpt_path=None,
        resume_ckpt_path=str(destination), initialize_touch_from_openwam=False, output_path=str(destination.parent))
    t.rebatch = plan
    t.step_timing.record_steps = 1000
    cfg.model.architecture.mot_checkpoint_mixed_attn = True
    cfg.project.name = "touchwam-six-source-64gpu-bs48-20261030"
    OmegaConf.save(cfg, destination/"config.yaml")
    for name in ("tokenizer", "normalization"):
        shutil.copytree(source/name, destination/name, copy_function=os.link)
    for name in ("normalization_sources.json", "initialization.json"):
        if (source/name).is_file(): os.link(source/name, destination/name)
    new_contract = copy.deepcopy(contract)
    new_contract["training_layout"].update(microbatch=target_batch, rebatch=plan)
    (destination/"training_data_contract.json").write_text(json.dumps(new_contract, indent=2)+"\n")
    new_state = destination/"accel_state_step_0"
    shutil.copytree(state, new_state, copy_function=os.link)
    # Atomic replacement breaks the hardlink, never overwriting source metadata.
    groups = [{"params": [torch.nn.Parameter(torch.zeros(()))], "lr": plan["peak_lr"]}
              for _ in saved_scheduler["_last_lr"]]
    optimizer = torch.optim.AdamW(groups, lr=plan["peak_lr"])
    scheduler = build_cosine_scheduler(optimizer, total_opt_steps=plan["remaining_steps"],
                                      cfg=cfg, num_processes=plan["world_size"])
    if not all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(scheduler.get_last_lr(), saved_scheduler["_last_lr"])):
        raise ValueError("Migrated initial LR does not equal the saved optimizer LR")
    temporary = new_state/"scheduler.bin.tmp"
    torch.save(scheduler.state_dict(), temporary)
    temporary.replace(new_state/"scheduler.bin")
    new_meta = dict(global_step=0, opt_step=0, epoch=0, checkpoint_format_version=2,
        optimizer_boundary=True, batches_per_epoch=plan["remaining_steps"], gradient_accumulation_steps=1,
        rebatch_from_step=step, rebatch_source=str(state))
    temporary = new_state/"trainer_state.json.tmp"
    temporary.write_text(json.dumps(new_meta)+"\n")
    temporary.replace(new_state/"trainer_state.json")
    # The phase-0 model is also independently deployable; storage is shared safely.
    os.link(source/f"checkpoint_step_{step}.safetensors", destination/"checkpoint_step_0.safetensors")
    proof = dict(status="staged", plan=plan, source_state=str(state), destination=str(destination),
        tensor_storage="hardlinks; model, Adam and RNG bytes unchanged", phase_step=0,
        initial_lr=scheduler.get_last_lr(), source_scheduler=saved_scheduler,
        target_scheduler=scheduler.state_dict())
    (destination/"rebatch_migration.json").write_text(json.dumps(proof,indent=2)+"\n")
    return proof


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--step", required=True, type=int)
    args=parser.parse_args()
    result=stage(args.source,args.destination,args.step)
    print(json.dumps({k:v for k,v in result.items() if "scheduler" not in k},indent=2))
