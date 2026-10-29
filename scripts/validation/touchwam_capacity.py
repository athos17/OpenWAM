#!/usr/bin/env python3
"""Full-model capacity check with distinct real windows and explicit source coverage.

Uses the production trainer/model/optimizer and uint8 collator. A deterministic
validation-only loader replaces population sampling, so every source executes.
Fixtures repeat across ranks/cycles; this is not a full-population I/O benchmark.
Checkpoint writing is disabled for the initial capacity probe.
"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.validation.touchwam_full_recipe_smoke import compose_config, sha256


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def prepare(args):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from openwam.dataloader.registry import build_dataset
    from scripts.preflight_touchwam import validate_sample, validate_source_contract
    torch.set_num_threads(2)
    cfg = compose_config(args.config)
    source = OmegaConf.create(OmegaConf.to_container(cfg.dataloader.datasets[args.prepare_source], resolve=True))
    if not source.get('enabled', True):
        raise ValueError('Disabled source')
    source.seed = int(cfg.project.seed)
    ds = build_dataset(source, split='train')
    leaves = getattr(ds, 'buckets', [ds])
    samples, proofs = [], []
    count = int(cfg.training.batch_size)
    for j, bi in enumerate(np.linspace(0, len(leaves)-1, count, dtype=int)):
        leaf = leaves[int(bi)]
        index = min(len(leaf)-1, max(0, int(len(leaf) * (j+1)/(count+1))))
        sample = getattr(leaf, '_getitem_impl', leaf.__getitem__)(index)
        mode = getattr(leaf, 'training_modality', getattr(ds, 'training_modality', 'video_action'))
        check = validate_sample(sample, mode, tactile_spec=cfg.model.architecture.tactile,
                                **validate_source_contract(args.prepare_source, source))
        samples.append(sample)
        proofs.append(dict(bucket=str(getattr(leaf, '_dataset_dir', getattr(leaf, 'root', ''))),
                           index=index, validation=check))
    plain = OmegaConf.to_container(source, resolve=True)
    expand = getattr(ds, 'normalization_sources_for_config', None)
    norms = expand(plain) if expand else {args.prepare_source: dict(config=plain,
        training_modality=mode, stats_path=getattr(ds, 'normalization_stats_path', None))}
    shape = tuple(samples[0]['tactile_observation_maps'].shape[-2:]) if 'tactile_observation_maps' in samples[0] else None
    record = dict(source=args.prepare_source, source_config=plain, mode=mode,
                  codec=samples[0].get('tactile_codec'), observation_shape=shape,
                  normalization_sources=norms, samples=samples, windows=proofs)
    folder = Path(args.fixtures); folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{args.prepare_source}.pt'
    if path.exists():
        raise FileExistsError(path)
    torch.save(record, path)
    write(path.with_suffix('.json'), dict(source=args.prepare_source, sha256=sha256(path),
          windows=proofs, distinct_windows=len({(p['bucket'], p['index']) for p in proofs})))
    print(json.dumps(dict(prepared=args.prepare_source, windows=count)), flush=True)


def train(args):
    import torch
    import torch.distributed as dist
    import torch.utils.checkpoint as checkpoint_module
    # Install before importing architecture modules that bind checkpoint aliases.
    checkpoint_calls = 0
    original_checkpoint = checkpoint_module.checkpoint
    def traced_checkpoint(*a, **kw):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if args.recompute_policy == 'forbid':
            raise RuntimeError('Activation recomputation was invoked in a no-recompute capacity run')
        return original_checkpoint(*a, **kw)
    checkpoint_module.checkpoint = traced_checkpoint
    from omegaconf import OmegaConf
    from openwam.dataloader.mixture import MixtureDataset
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.training_utils import collate_samples_with_video_uint8
    from scripts.train import _build_accelerator

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    cfg = compose_config(args.config)
    t = cfg.training
    microbatch = int(t.batch_size)
    assert microbatch > 0 and t.gradient_accumulation_steps == 1
    assert not t.use_gradient_checkpointing_offload
    if args.recompute_policy == 'forbid':
        assert not t.use_gradient_checkpointing
        assert not cfg.model.architecture.mot_checkpoint_mixed_attn
    names = args.sources.split(',') if args.sources else [k for k,v in cfg.dataloader.datasets.items() if v.get('enabled', True)]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate source')
    stop = len(names) * args.cycles
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    # Keep a long schedule while stopping the capacity run after explicit coverage.
    # Its horizon represents the requested 64-card run approximately;
    # production modality-group padding is not part of this capacity fixture.
    horizon = math.ceil(106277944 / (64 * microbatch))
    t.update(dataset_num_workers=0, max_steps=horizon, num_epochs=1, stop_after_steps=stop,
             save_steps=0, save_full_states_for_resume=False, output_path=str(output/'run'),
             log_flush_every=1)
    t.step_timing.summary_every = 1

    class Source(torch.utils.data.Dataset):
        action_dim = 80
        def __init__(self, name):
            p = Path(args.fixtures)/f'{name}.pt'
            proof = json.loads(p.with_suffix('.json').read_text())
            if sha256(p) != proof['sha256']:
                raise ValueError('Fixture changed')
            self.record = torch.load(p, map_location='cpu', weights_only=False)
            if len(self.record['samples']) != microbatch or proof['distinct_windows'] != microbatch:
                raise ValueError(f'Expected {microbatch} distinct fixture windows: {name}')
            live = OmegaConf.to_container(cfg.dataloader.datasets[name], resolve=True)
            frozen = dict(self.record['source_config']); frozen.pop('seed', None)
            live.pop('seed', None)
            if live != frozen:
                raise ValueError(f'Fixture configuration drift: {name}')
            self.training_modality = self.record['mode']
            self.tactile_codec = self.record['codec']
            self.tactile_observation_shape = self.record['observation_shape']
            self.normalization_stats_path = None
            self.training_data_contract = dict(schema='capacity_fixture_v1', sha256=proof['sha256'])
        def __len__(self): return len(self.record['samples'])
        def __getitem__(self, i): return dict(self.record['samples'][i % len(self)])
        def normalization_sources_for_config(self, config): return self.record['normalization_sources']

    class Coverage(MixtureDataset):
        def __len__(self): return horizon * world * int(t.batch_size)
        def set_epoch(self, epoch): pass
        def __getitem__(self, i):
            source = (i // (world * int(t.batch_size))) % len(names)
            sample = self._datasets[source][i % int(t.batch_size)]
            sample.update(_dataset_index=source, _dataset_name=names[source])
            return sample

    sources = [Source(name) for name in names]
    dataset = Coverage(sources, names=names, seed=42, strict_action_dim=True,
                       source_configs=[s.record['source_config'] for s in sources])
    dataset.training_layout = dict(batching='validation_source_round_robin', world_size=world,
                                   microbatch=microbatch, gradient_accumulation=1)
    accelerator = _build_accelerator(cfg)
    records, pending = [], {}

    class TracedTrainer(OpenWAMTrainer):
        def build_dataloader(self, batch_size):
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True, collate_fn=collate_samples_with_video_uint8)
        def compute_loss(self, batch):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            routes = [row['_dataset_name'] for row in batch]
            if len(routes) != microbatch or len(set(routes)) != 1:
                raise ValueError(f'Wrong per-rank capacity batch: {routes}')
            pending.update(source=routes[0], start=time.monotonic())
            print(json.dumps(dict(rank=rank, stage='forward', source=routes[0], batch_size=len(routes))), flush=True)
            return super().compute_loss(batch)
        def log_step(self, **kw):
            torch.cuda.synchronize()
            metrics = {k:float(v) for k,v in kw['metrics'].items()}
            if not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError(f'Nonfinite metrics: {metrics}')
            rec = dict(rank=rank, source=pending['source'], step=kw['global_step'],
                opt_step=kw['opt_step'], metrics=metrics, lr=kw['lr'],
                seconds=time.monotonic()-pending['start'],
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            records.append(rec)
            with (output/f'steps-rank{rank}.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
            super().log_step(**kw)

    trainer = TracedTrainer(cfg, accelerator=accelerator, dataset=dataset)
    init = getattr(trainer.architecture, 'initialization_provenance', {})
    assert init.get('openwam_tensors_preserved') is True
    assert init.get('tactile_observation_encoder', {}).get('initialization') == 'ftp1_pretrained'
    if rank == 0:
        write(output/'initialization.json', init)
        OmegaConf.save(cfg, output/'capacity_config.yaml')
    trainer.train()
    if [r['source'] for r in records] != names * args.cycles or [r['opt_step'] for r in records] != list(range(1, stop+1)):
        raise ValueError('Source coverage or real optimizer updates incomplete')
    if args.recompute_policy == 'config' and t.use_gradient_checkpointing and checkpoint_calls == 0:
        raise ValueError('Configured activation checkpointing was never observed')
    write(output/f'rank-{rank}.json', dict(status='passed', records=records,
          explicit_activation_checkpoint_calls=checkpoint_calls))
    accelerator.wait_for_everyone()
    if rank == 0:
        ranks = [json.loads((output/f'rank-{i}.json').read_text()) for i in range(world)]
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        write(output/'report.json', dict(status='passed', ranks=ranks, per_gpu_batch=microbatch,
              gpu_name=device.name, gpu_total_memory_gib=device.total_memory/1024**3,
              world_size=world, recompute_policy=args.recompute_policy,
              explicit_activation_checkpoint_calls=sum(r['explicit_activation_checkpoint_calls'] for r in ranks),
              sources=names,
              steps=stop, real_reader_windows_per_source=microbatch, repeated_across_ranks=True,
              full_population_io_benchmarked=False, checkpoint_io=False,
              validation_scope='full_model_capacity_and_optimizer_updates'))
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='train_touchwam_six_source_bs8_norecompute_20261029')
    p.add_argument('--fixtures', required=True)
    p.add_argument('--prepare-source')
    p.add_argument('--output')
    p.add_argument('--sources')
    p.add_argument('--cycles', type=int, default=4)
    p.add_argument('--recompute-policy', choices=['forbid', 'config'], default='forbid',
                   help='Forbid checkpoint calls, or exercise the configured recomputation policy')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.prepare_source: prepare(args)
    else:
        if not args.output or args.cycles < 1: p.error('Positive cycles and output required')
        try: train(args)
        except Exception as exc:
            import traceback
            import torch
            out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
            write(out/f'failure-rank{os.environ.get("RANK", "0")}.json',
                  dict(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_initialized() else None,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_initialized() else None))
            raise


if __name__ == '__main__': main()
