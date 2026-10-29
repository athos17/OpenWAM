#!/usr/bin/env python3
"""Bounded full-Alpha distributed training on provenance-bound reader fixtures.

Prepare each source in a separate CPU process, then run once with torchrun.
The fixture repeats one real, already-normalized window per rank. Model sizes,
T5/VAE preprocessing, tactile encoders, loss, optimizer and trainer are real.
This is a short integration check, not a representative training run or quality
evaluation. Checkpoints are disabled; reports retain the source provenance.
"""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def compose_config(name):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        return compose(config_name=name)


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    from openwam.dataloader.registry import build_dataset
    from openwam.dataloader.utils.normalization_package import build_normalization_manifest
    from scripts.preflight_touchwam import validate_sample, validate_source_contract

    torch.set_num_threads(2)
    cfg = compose_config(args.config)
    source = OmegaConf.create(OmegaConf.to_container(cfg.dataloader.datasets[args.prepare_source], resolve=True))
    if not source.get('enabled', True):
        raise ValueError('A disabled source cannot enter the full-model smoke')
    source.seed = int(cfg.project.seed)
    dataset = build_dataset(source, split='train')
    leaves = getattr(dataset, 'buckets', [dataset])
    leaf = leaves[len(leaves) // 2]
    index = len(leaf) // 2
    sample = getattr(leaf, '_getitem_impl', leaf.__getitem__)(index)
    mode = getattr(leaf, 'training_modality', getattr(dataset, 'training_modality', 'video_action'))
    checked = validate_sample(sample, mode, tactile_spec=cfg.model.architecture.tactile,
                              **validate_source_contract(args.prepare_source, source))
    observation_shape = getattr(leaf, 'tactile_observation_shape', None)
    if 'tactile_observation_maps' in sample:
        measured_shape = tuple(sample['tactile_observation_maps'].shape[-2:])
        if observation_shape is not None and tuple(observation_shape) != measured_shape:
            raise ValueError('Reader observation shape differs from its emitted sample')
        observation_shape = measured_shape
    plain = OmegaConf.to_container(source, resolve=True)
    expand = getattr(dataset, 'normalization_sources_for_config', None)
    norms = expand(plain) if expand else {args.prepare_source: {
        'config': plain, 'training_modality': mode,
        'stats_path': getattr(dataset, 'normalization_stats_path', None)}}
    norm_manifest = build_normalization_manifest(norms)
    record = {
        'source': args.prepare_source, 'source_config': plain,
        'source_publication': getattr(dataset, 'publication_contract', None),
        'sample_validation': checked, 'window_index': index,
        'bucket': str(getattr(leaf, '_dataset_dir', getattr(leaf, 'root', source.dataset_dir))),
        'mode': mode, 'codec': sample.get('tactile_codec'),
        'observation_shape': observation_shape,
        'normalization_manifest': norm_manifest,
        'normalization_sources': norms, 'sample': sample,
    }
    folder = Path(args.fixtures)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{args.prepare_source}.pt'
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(f'Fixture already exists; use a new evidence directory: {path}')
    torch.save(record, path)
    proof = {k: v for k, v in record.items() if k not in ('sample', 'normalization_sources')}
    proof.update(fixture_sha256=sha256(path), fixture_bytes=path.stat().st_size)
    path.with_suffix('.json').write_text(json.dumps(proof, indent=2, default=str) + '\n')
    print(json.dumps({'source': args.prepare_source, 'status': 'passed', 'fixture': str(path)}), flush=True)


def train(args):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from openwam.dataloader.mixture import MixtureDataset
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from scripts.train import _build_accelerator

    torch.set_num_threads(2)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world != 8:
        raise ValueError('This full-model validation is budgeted for exactly eight GPUs')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.cuda.reset_peak_memory_stats()
    names = args.sources.split(',')
    if not names or len(names) != len(set(names)):
        raise ValueError('Select distinct source fixture names')
    cfg = compose_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg.training.update(batch_size=1, gradient_accumulation_steps=1, dataset_num_workers=0,
                        num_epochs=1, max_steps=len(names), save_steps=0,
                        save_full_states_for_resume=False, output_path=str(output / 'run'), debug=False)
    cfg.project.wandb = None
    cfg.project.seed = 42
    if not cfg.training.initialize_touch_from_openwam or cfg.training.action_backbone_init != 'pretrained':
        raise ValueError('The smoke must use complete pretrained OpenWAM initialization')

    class Source(torch.utils.data.Dataset):
        action_dim = 80

        def __init__(self, name):
            path = Path(args.fixtures) / f'{name}.pt'
            proof = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != proof['fixture_sha256']:
                raise ValueError(f'Fixture changed: {name}')
            self.record = torch.load(path, map_location='cpu', weights_only=False)
            if self.record['source'] != name:
                raise ValueError('Fixture source mismatch')
            live_source = OmegaConf.to_container(cfg.dataloader.datasets[name], resolve=True)
            frozen_source = dict(self.record['source_config'])
            live_source.pop('seed', None)
            frozen_source.pop('seed', None)
            if live_source != frozen_source:
                raise ValueError(f'Fixture source config no longer matches the selected recipe: {name}')
            self.training_modality = self.record['mode']
            self.tactile_codec = self.record['codec']
            self.tactile_observation_shape = self.record['observation_shape']
            self.publication_contract = self.record['source_publication']
            self.training_data_contract = dict(schema='real_reader_repeated_smoke_window_v1',
                fixture_sha256=proof['fixture_sha256'], bucket=self.record['bucket'],
                window_index=self.record['window_index'], original_source_config=self.record['source_config'])
            self.normalization_stats_path = None

        def __len__(self):
            return world

        def __getitem__(self, index):
            if not 0 <= index < world:
                raise IndexError(index)
            return dict(self.record['sample'])

        def normalization_sources_for_config(self, config):
            return self.record['normalization_sources']

    sources = [Source(name) for name in names]
    dataset = MixtureDataset(sources, names=names, seed=42, strict_action_dim=True,
                             source_configs=[s.record['source_config'] for s in sources])
    accelerator = _build_accelerator(cfg)
    seen = []
    graph_schedule = []
    graph_by_source = {name: [s.training_modality, s.tactile_codec, s.tactile_observation_shape]
                       for name, s in zip(names, sources)}

    class TracedTrainer(OpenWAMTrainer):
        def compute_loss(self, batch):
            rows = batch if isinstance(batch, list) else [batch]
            route = [item['_dataset_name'] for item in rows]
            seen.extend(route)
            graph_schedule.extend(graph_by_source[name] for name in route)
            print(json.dumps({'rank': rank, 'stage': 'compute_loss', 'sources': route}), flush=True)
            losses = super().compute_loss(batch)
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite loss {key} on {route}')
            return losses

    started = time.time()
    trainer = TracedTrainer(cfg, accelerator=accelerator, dataset=dataset)
    initialization = getattr(trainer.architecture, 'initialization_provenance', {})
    if initialization.get('openwam_tensors_preserved') is not True:
        raise ValueError('Complete pretrained OpenWAM initialization was not preserved')
    if initialization.get('tactile_observation_encoder', {}).get('initialization') != 'ftp1_pretrained':
        raise ValueError('The FTP tactile observation encoder was not initialized from pretrained weights')
    if rank == 0:
        (output / 'initialization.json').write_text(json.dumps(
            initialization, indent=2, default=str) + '\n')
    trainer.train()
    if len(seen) != len(names):
        raise ValueError(f'Incomplete batch count on rank {rank}: {seen}')
    report = dict(status='passed', rank=rank, world_size=world, seen_sources=seen,
                  graph_schedule=graph_schedule,
                  elapsed_seconds=time.time()-started, source_steps=len(seen),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                  full_model=True, real_t5_vae=True, real_reader_fixtures=True,
                  repeated_window_per_source=True, checkpoint_io=False)
    (output / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
    accelerator.wait_for_everyone()
    if rank == 0:
        files = list((output / 'run').glob('*/training_metrics.jsonl'))
        if len(files) != 1:
            raise ValueError('Expected one production trainer metrics file')
        metrics = [json.loads(line) for line in files[0].read_text().splitlines()]
        if len(metrics) != len(names):
            raise ValueError('Production trainer did not complete the requested update budget')
        if [row.get('opt_step') for row in metrics] != list(range(1, len(names) + 1)):
            raise ValueError('A real optimizer update was skipped during the smoke')
        for row in metrics:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f'Nonfinite training metric: {key}')
        ranks = [json.loads((output/f'rank-{i}.json').read_text()) for i in range(world)]
        if any(r['graph_schedule'] != ranks[0]['graph_schedule'] for r in ranks):
            raise ValueError('Ranks did not follow the same modality/codec/shape schedule')
        from collections import Counter
        coverage = Counter(name for r in ranks for name in r['seen_sources'])
        if dict(coverage) != {name: world for name in names}:
            raise ValueError(f'Incomplete global fixture coverage: {coverage}')
        summary = dict(status='passed', config=args.config, steps=len(names), world_size=world,
                       ranks=ranks, metrics=metrics, global_source_counts=dict(coverage),
                       validation_scope='full_model_real_data_bounded_smoke',
                       full_population_training=False, long_run_stability_verified=False)
        (output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'status':'passed','report':str(output/'report.json')}),flush=True)
    accelerator.wait_for_everyone()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='train_touchwam_six_source_20261029')
    parser.add_argument('--fixtures', required=True)
    parser.add_argument('--output')
    parser.add_argument('--prepare-source')
    parser.add_argument('--sources', default='egodex,touchscale,agibot,robocoin,sharpadex,ftp_freetacman,ftp_rdp_bimanual')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.prepare_source:
        prepare(args)
    else:
        if not args.output:
            parser.error('--output is required for the GPU run')
        train(args)


if __name__ == '__main__':
    main()
