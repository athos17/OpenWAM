"""Coverage, distributed sharding, and strict mixed-source checkpoint contracts."""
from copy import deepcopy
from itertools import zip_longest

import numpy as np
import pytest
import torch
from accelerate.data_loader import prepare_data_loader
from omegaconf import OmegaConf

from openwam.dataloader.mixture import MixtureDataset, _plain_config
from openwam.dataloader.modality_sampler import GlobalModalitySampler
from openwam.dataloader.utils.normalization_package import (
    save_normalization_package, read_normalization_package, verify_normalization_package,
)
from openwam.train.utils.checkpointing import save_normalization_stats, verify_resume_normalization_stats
from openwam.deploy.model_loader import _build_normalizer


def make_mixture(factory, counts=(21, 13, 17), **kwargs):
    sources = [factory(n) for n in counts]
    for ds, modality in zip(sources, ('video_only', 'video_action', 'video_action')):
        ds.training_modality = modality
        ds.epoch = 0
        ds.set_epoch = lambda epoch, ds=ds: setattr(ds, 'epoch', epoch)
    return MixtureDataset(sources, names=['ego', 'agi', 'sharpa'], **kwargs)


@pytest.mark.parametrize('batch,world', [(1, 4), (3, 4), (2, 1)])
def test_sampler_coverage_and_accelerate_sharding(fake_dataset_factory, batch, world):
    ds = make_mixture(fake_dataset_factory)
    sampler = GlobalModalitySampler(ds, batch, world)
    order = list(sampler)
    assert set(order) == set(range(len(ds)))
    assert len(order) == len(sampler)
    assert len(order) - len(ds) < 2 * batch * world
    assert order == list(sampler)
    loaders = []
    for rank in range(world):
        loader = torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler, collate_fn=list)
        loaders.append(prepare_data_loader(loader, device=torch.device('cpu'), num_processes=world,
                                          process_index=rank, split_batches=False, put_on_device=False))
    for global_batches in zip_longest(*loaders):
        assert all(b is not None and len(b) == batch for b in global_batches)
        modalities = {ds.source_modalities[x['_dataset_index']] for b in global_batches for x in b}
        assert len(modalities) == 1
    sampler.set_epoch(1)
    assert all(child.epoch == 1 for child in ds.datasets)
    assert list(sampler) != order
    again = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) == again


def test_partial_source_draw_changes_and_reaches_tail(fake_dataset_factory):
    ds = MixtureDataset([fake_dataset_factory(100), fake_dataset_factory(10)], weights=[1, 1])
    visited = set()
    for epoch in range(5):
        ds.set_epoch(epoch)
        indices = ds._index_map[ds._index_map[:, 0] == 0, 1]
        assert len(set(indices)) == 55
        visited.update(indices.tolist())
    assert len(visited) > 90 and max(visited) == 99


def package_sources(tmp_path):
    sources = {}
    for name, mode, offset in [('agi', 'quantile', 1), ('sharpa', 'min-max', 5)]:
        stats = {k: np.full(20, v, np.float32) for k, v in
                 dict(mean=offset, std=1, min=offset-2, max=offset+2, q01=offset-2, q99=offset+2).items()}
        state = {k: v + 3 if k != 'std' else v for k, v in stats.items()}
        path = tmp_path / (name + '.npy')
        np.save(path, {'eef': stats, 'eef_state': state})
        sources[name] = dict(config=dict(type=name, normalize_mode=mode, action_mode='eef', unify_action=False),
                             stats_path=str(path), training_modality='video_action')
    sources['ego'] = dict(config={'type': 'egodex_video'}, stats_path=None, training_modality='video_only')
    return sources


def test_package_independent_state_transforms_and_explicit_selection(tmp_path):
    sources = package_sources(tmp_path)
    output = tmp_path / 'checkpoint'
    save_normalization_package(output, sources)
    verify_normalization_package(output, sources)
    save_normalization_package(output, sources)
    cfg = OmegaConf.create({'dataloader': {'type': 'mixture'}, 'project': {'output_dir': '${train_only_clock:now}'}})
    with pytest.raises(ValueError, match='normalization_source'):
        _build_normalizer(cfg, str(output))
    for name in ('agi', 'sharpa'):
        mixed = _build_normalizer(cfg, str(output), normalization_source=name)
        entry = read_normalization_package(output)['sources'][name]
        standalone = _build_normalizer(OmegaConf.create({'dataloader': sources[name]['config']}),
                                      str((output / entry['stats_file']).parent))
        x = np.full((1,20), 4., np.float32)
        np.testing.assert_array_equal(mixed.normalize(x), standalone.normalize(x))
        np.testing.assert_array_equal(mixed.unnormalize(x / 4), standalone.unnormalize(x / 4))
    with pytest.raises(ValueError, match='no action supervision'):
        _build_normalizer(cfg, str(output), normalization_source='ego')
    with pytest.raises(ValueError):
        _build_normalizer(cfg, str(output), normalization_source='absent')
    drift = deepcopy(sources)
    drift['agi']['config']['normalize_mode'] = 'min-max'
    with pytest.raises(ValueError, match='differ'):
        verify_normalization_package(output, drift)
    entry = read_normalization_package(output)['sources']['agi']
    (output / entry['stats_file']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        read_normalization_package(output)


@pytest.mark.parametrize('change', ['index', 'stats', 'layout', 'config', 'manifest'])
def test_resume_rejects_source_drift(tmp_path, fake_dataset_factory, change):
    sources = package_sources(tmp_path)
    configs = [sources[n]['config'] for n in ('ego','agi','sharpa')]
    configs[0]['dataset_dir'] = str(tmp_path)
    (tmp_path/'dataset_manifest.json').write_text('{}')
    ds = make_mixture(fake_dataset_factory, source_configs=configs)
    ds.datasets[0].training_data_contract = {'index_sha256': 'original'}
    ds.datasets[1].normalization_stats_path = sources['agi']['stats_path']
    ds.datasets[2].normalization_stats_path = sources['sharpa']['stats_path']
    ds.training_layout = {'world_size':4, 'microbatch':1, 'gradient_accumulation':2}
    output = tmp_path/'checkpoint'
    save_normalization_stats(str(output), ds)
    verify_resume_normalization_stats(str(output), ds)
    if change == 'index': ds.datasets[0].training_data_contract['index_sha256'] = 'changed'
    if change == 'stats': (tmp_path/'agi.npy').write_bytes(b'changed')
    if change == 'layout': ds.training_layout['world_size'] = 8
    if change == 'config': ds.source_configs[1]['normalize_mode'] = 'min-max'
    if change == 'manifest': (tmp_path/'dataset_manifest.json').write_text('{"revision":2}')
    with pytest.raises(ValueError, match='contract differs'):
        verify_resume_normalization_stats(str(output), ds)


def test_plain_source_config_resolves_nested_omegaconf():
    nested = {'color_jitter': OmegaConf.create({'enabled': False})}
    assert type(_plain_config(nested)['color_jitter']) is dict


def test_deploy_cli_selects_source():
    from openwam.deploy.server import _build_argparser, resolve_deploy_args
    args = _build_argparser().parse_args(['--ckpt-dir', '/tmp/model', '--normalization-source', 'agi'])
    cfg, _ = resolve_deploy_args(args)
    assert cfg.normalization_source == 'agi'
