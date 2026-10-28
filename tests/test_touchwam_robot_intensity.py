"""Robot actions remain actions through real v3 video/time/normalization reads."""
import copy
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openwam.dataloader.robot_intensity import LeRobotRobotIntensityDataset, _robot_table
from openwam.dataloader.touchwam_pretrain import LeRobotVideoIntensityDataset, HomogeneousPublishedBuckets
from tests.test_touchwam_publication_reader import make_video_dataset


def robot_bucket(root, *, tactile=False, semantics="recorded_command"):
    make_video_dataset(root)
    info = json.loads((root / 'meta/info.json').read_text())
    slots = [34, 35, 43]  # Synthetic canonical position components and gripper, not arm joints.
    for column in ('action', 'observation.state'):
        info['features'][column] = {'dtype': 'float32', 'shape': [3]}
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/source_definition.json').write_text(json.dumps({
        'source_id': 'ftp1/Synthetic', 'source_kind': 'robot', 'physical_time_verified': True}))
    table_path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(table_path)
    state = np.tile(np.array([1., 2., .5], np.float32), (40, 1))
    action = np.tile(np.array([2., 4., 1.], np.float32), (40, 1))
    action[:, 0] = np.arange(40) / 10
    state[:, 0] = np.arange(40) / 20
    for name, values in [('action', action), ('observation.state', state)]:
        table = table.append_column(name, pa.array(values.tolist()))
        table = table.append_column(name + '.valid', pa.array([[True] * 3] * 40))
    pq.write_table(table, table_path)
    stats = {}
    for key, bound in [('eef', 4.), ('eef_state', 2.)]:
        stats[key] = dict(min=np.full(80, -1.), max=np.full(80, 1.),
                          mean=np.zeros(80), std=np.ones(80), q01=np.full(80, -1.), q99=np.full(80, 1.))
        for name in ('min', 'q01'):
            stats[key][name][slots] = -bound
        for name in ('max', 'q99'):
            stats[key][name][slots] = bound
    stats_path = root / 'normalization_stats.npy'
    np.save(stats_path, stats)
    spec = dict(slots=slots, units=['m', 'm', 'aperture_fraction'], coordinate_frame='synthetic_robot_base',
                representation='openwam_canonical_components_v1')
    contract = dict(schema='touchwam_robot_action_v1', source_kind='robot', source_id='ftp1/Synthetic',
                    dataset_info_sha256=hashlib.sha256((root / 'meta/info.json').read_bytes()).hexdigest(),
                    verification={'status': 'verified', 'evidence': ['synthetic known geometry and controls']},
                    action_semantics=semantics, terminal_action_valid=False,
                    action=dict(spec, column='action', validity_column='action.valid'),
                    proprio=dict(spec, column='observation.state', validity_column='observation.state.valid'),
                    normalization=dict(mode='min-max', population='train', file='normalization_stats.npy',
                        sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(), action_key='eef', proprio_key='eef_state'))
    if semantics == 'future_state_target':
        contract['action'] = copy.deepcopy(contract['proprio'])
    (root / 'meta/action_contract.json').write_text(json.dumps(contract))
    if tactile:
        t = root / 'tactile'
        t.mkdir()
        np.save(t / 'maps.npy', np.ones((20, 1, 32, 32), np.float32) * .4)
        np.save(t / 'valid.npy', np.ones((20, 1), bool))
        np.save(t / 'times.npy', np.arange(20) / 10)
        (t / 'manifest.json').write_text(json.dumps(dict(schema='tactile_intensity_v1', codec='tactile_intensity_v1',
            dataset_info_sha256=contract['dataset_info_sha256'], fps=10,
            region_names=['right.gripper.jaw_a.surface'], sensor_names=['unit'],
            episodes={'1': dict(maps='maps.npy', valid='valid.npy', timestamps='times.npy', layout_present=[True])})))
    return root, contract


def reader(root, **kwargs):
    return LeRobotRobotIntensityDataset(root, height=48, width=32, **kwargs)


@pytest.mark.parametrize('tactile', [False, True])
def test_robot_actions_time_and_separate_normalization(tmp_path, tactile):
    root, _ = robot_bucket(tmp_path / 'robot', tactile=tactile)
    ds = reader(root, tactile_mode='required' if tactile else 'off')
    sample = ds[0]
    assert sample['episode_id'] == 1
    assert sample['training_modality'] == ('video_action_tactile' if tactile else 'video_action')
    assert sample['action'].shape == (32, 80) and sample['proprio'].shape == (1, 80)
    assert sample['action_mask'].sum() == 32 * 3
    torch.testing.assert_close(sample['action'][:4, 34], torch.tensor([.5, .5, .5, .525]))
    torch.testing.assert_close(sample['proprio'][0, [34, 35, 43]], torch.tensor([.5, 1., .25]))
    assert not sample['action'][:, :34].any()
    assert not sample['action_mask'][:, :34].any()
    if tactile:
        assert sample['tactile_valid'][-1].all()
        assert sample['future_tactile_valid'].all()
    assert ds.normalization_stats is None  # no double normalization by the mixture
    assert HomogeneousPublishedBuckets([ds]).normalization_stats_path == ds.normalization_stats_path


def test_robot_episode_tail_and_missing_components_are_masked(tmp_path):
    root, _ = robot_bucket(tmp_path / 'robot')
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).to_pydict()
    table['action.valid'][20][1] = False
    table['action'][20][1] = float('nan')
    table['action'][39] = [float('nan')] * 3  # terminal command intentionally has no target
    pq.write_table(pa.table(table), path)
    ds = reader(root, tactile_mode='off')
    sample = ds[0]
    assert torch.isfinite(sample['action']).all()
    assert not sample['action_mask'][:3, 35].any()
    tail = ds[len(ds) - 1]
    assert tail['episode_start_frame'] == 17
    assert tail['action_mask'][:6, 34].all()  # native commands 17 and 18; terminal 19 excluded
    assert not tail['action_mask'][6:].any()
    assert not tail['action'][6:].any()


def test_measured_future_state_requires_explicit_semantics_and_never_uses_current_as_future(tmp_path):
    root, _ = robot_bucket(tmp_path / 'robot', semantics='future_state_target')
    sample = reader(root, tactile_mode='off')[0]
    assert sample['action_semantics'] == 'future_state_target'
    assert not sample['action_mask'][0].any()  # 10 Hz stream cannot label a 33 ms future as current state
    assert sample['action_mask'][1].any()
    assert (sample['action_label_time_offsets_s'][sample['action_mask'].any(-1)] > 0).all()
    assert sample['action'][1, 34] == pytest.approx(1.05 / 4)


def test_ftp_cannot_be_loaded_as_ego_video(tmp_path):
    root, _ = robot_bucket(tmp_path / 'robot')
    with pytest.raises(ValueError, match='action reader'):
        LeRobotVideoIntensityDataset(root)
    (root / 'meta/action_contract.json').unlink()
    with pytest.raises(FileNotFoundError, match='excluded'):
        reader(root)


@pytest.mark.parametrize('mutation,match', [
    (lambda c: c['verification'].update(status='unverified'), 'verified'),
    (lambda c: c.update(action_semantics='guessed_from_joints'), 'semantics'),
    (lambda c: c.update(dataset_info_sha256='wrong'), 'fingerprint'),
    (lambda c: c['action'].update(slots=[34, 34, 43]), 'more than one'),
    (lambda c: c['action'].update(representation='native_joints'), 'native geometry'),
    (lambda c: c['normalization'].update(population='all_splits'), 'train-population'),
    (lambda c: c['normalization'].update(sha256='wrong'), 'fingerprinted'),
    (lambda c: c['proprio'].update(validity_column='missing'), 'columns'),
])
def test_unverified_or_mismatched_contract_fails(tmp_path, mutation, match):
    root, contract = robot_bucket(tmp_path / 'robot')
    mutation(contract)
    (root / 'meta/action_contract.json').write_text(json.dumps(contract))
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        reader(root)


def test_no_valid_robot_actions_never_silently_downgrade(tmp_path):
    root, _ = robot_bucket(tmp_path / 'robot')
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).to_pydict()
    table['action.valid'] = [[False] * 3] * 40
    pq.write_table(pa.table(table), path)
    with pytest.raises(ValueError, match='fallback is forbidden'):
        reader(root)[0]


def test_from_config_exports_bound_normalizer_and_contract(tmp_path):
    from openwam.dataloader.registry import build_dataset
    root, contract = robot_bucket(tmp_path / 'robot', tactile=True)
    cfg = dict(type='lerobot_robot_intensity', source_kind='robot', source_family='ftp1',
               expected_modalities=['video_action_tactile'], dataset_dir=str(root), tactile_mode='required',
               normalize_mode='min-max', action_mode='eef', height=48, width=32)
    ds = build_dataset(cfg)
    assert ds.training_data_contract['action_contract'] == contract
    assert ds.normalization_stats_path.endswith('normalization_stats.npy')
    cfg['expected_modalities'] = ['video_action']
    with pytest.raises(ValueError, match='fallback is forbidden'):
        build_dataset(cfg)


def test_float32_clock_keeps_current_tactile_observation(tmp_path):
    root, _ = robot_bucket(tmp_path / 'robot', tactile=True)
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path)
    table = table.set_column(table.column_names.index('timestamp'), 'timestamp', table['timestamp'].cast(pa.float32()))
    pq.write_table(table, path)
    ds = reader(root, tactile_mode='required')
    for i in (3, 7, 9, 11):
        sample = ds[i]
        assert sample['tactile_valid'][-1].all()
        assert sample['tactile_history_source_times_s'][-1] == pytest.approx(0)
    assert ds[11]['future_tactile_valid'].all()  # +0.8s is the actual terminal 1.9s frame


@pytest.mark.parametrize('malformation', ['flat', 'missing_state', 'incomplete', 'unused_nonidentity'])
def test_normalization_rejects_packages_that_cannot_roundtrip_to_deploy(tmp_path, malformation):
    root, contract = robot_bucket(tmp_path / 'robot')
    path = root / 'normalization_stats.npy'
    stats = np.load(path, allow_pickle=True).item()
    if malformation == 'flat':
        stats = stats['eef']
    elif malformation == 'missing_state':
        del stats['eef_state']
    elif malformation == 'incomplete':
        del stats['eef']['mean']
    else:
        stats['eef_state']['min'][0] = stats['eef_state']['max'][0] = 0
    np.save(path, stats)
    contract['normalization']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'meta/action_contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='complete action/proprio|identity normalization'):
        reader(root)


def test_missing_entire_validity_vector_is_missing_not_a_shape_error(tmp_path):
    root, _ = robot_bucket(tmp_path / 'robot')
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).to_pydict()
    table['action'][21] = table['action.valid'][21] = None
    pq.write_table(pa.table(table), path)
    sample = reader(root)[0]
    assert not sample['action_mask'][3:6].any()
    assert not sample['action'][3:6].any()
    assert sample['action_mask'][6].any()


def test_packaged_normalizer_matches_reader_action_and_proprio(tmp_path):
    from omegaconf import OmegaConf
    from openwam.dataloader.utils.normalization_package import save_normalization_package
    from openwam.deploy.model_loader import _build_normalizer
    root, _ = robot_bucket(tmp_path / 'robot', tactile=True)
    ds = reader(root, tactile_mode='required')
    config = dict(type='lerobot_robot_intensity', normalize_mode='min-max', action_mode='eef', unify_action=False)
    output = tmp_path / 'checkpoint'
    save_normalization_package(output, {'ftp': dict(config=config, training_modality='video_action_tactile',
                                                  stats_path=ds.normalization_stats_path)})
    normalizer = _build_normalizer(OmegaConf.create({'dataloader': {'type': 'mixture'}}), str(output), 'ftp')
    sample = ds[0]
    physical = np.zeros((1, 80), np.float32)
    physical[0, [34, 35, 43]] = [1., 2., .5]
    np.testing.assert_allclose(normalizer.normalize(physical), sample['proprio'].numpy())
    recovered = normalizer.unnormalize(sample['action'].numpy())
    np.testing.assert_allclose(recovered[0, [34, 35, 43]], [2., 4., 1.])
    assert np.all(recovered[:, :34] == 0)


def test_robot_reader_reaches_real_wan_action_and_tactile_backward(tmp_path, monkeypatch):
    from scripts.validation.touchwam_multisource_gpu import tiny_model
    from tests.test_touchwam import open_gates

    root, _ = robot_bucket(tmp_path / 'robot', tactile=True)
    sample = reader(root, tactile_mode='required')[0]
    model = tiny_model(torch.device('cpu'), torch.float32)
    # Tiny production modules, native 80-D actions and 32x32 tactile maps.
    # Only expensive text/image encoders are replaced by synthetic latents.
    model.tactile_spec['sensor_names'] = ['unit', 'bench2dex_sim']
    monkeypatch.setattr(model, 'preprocess', lambda **kw: dict(
        input_latents=torch.randn(1, 4, 3, 4, 4), context=torch.randn(1, 4, 32),
        first_frame_latents=torch.zeros(1, 4, 1, 4, 4), context_mask=torch.ones(1, 4, dtype=torch.bool),
        fuse_vae_embedding_in_latents=True))
    prepared = model.prepare_inputs([sample])
    open_gates(model)
    prepared['use_gradient_checkpointing'] = True
    losses = model.compute_loss(**prepared)
    losses['loss'].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses['loss_action'] > 0 and losses['loss_tactile'] > 0
    assert model.action_backbone.action_encoder.weight.grad.abs().sum() > 0
    assert model.touch_v2.input_projection[0].weight.grad.abs().sum() > 0
