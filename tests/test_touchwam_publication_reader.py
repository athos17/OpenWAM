import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openwam.tactile.intensity_store import IntensityStore
from openwam.dataloader.touchwam_pretrain import LeRobotVideoIntensityDataset
from openwam.dataloader.mixture import MixtureDataset
from openwam.dataloader.modality_sampler import GlobalModalitySampler, batch_modality_code


def sidecar(tmp_path, *, fps=10, timestamps=None):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text('{}')
    t = root / "tactile"
    t.mkdir()
    times = np.arange(12, dtype=np.float64) / fps if timestamps is None else timestamps
    maps = np.zeros((len(times), 2, 8, 8), dtype=np.float32)
    maps[:, 1] = np.arange(len(times))[:, None, None] / len(times)
    valid = np.ones((len(times), 2), dtype=bool)
    np.save(t / "maps.npy", maps)
    np.save(t / "valid.npy", valid)
    np.save(t / "timestamps.npy", times)
    manifest = dict(schema="tactile_regions_v1", codec="tactile_intensity_v1", fps=fps,
        dataset_info_sha256=hashlib.sha256((root / 'meta/info.json').read_bytes()).hexdigest(),
        region_names=["right.gripper.jaw_a.surface", "right.gripper.jaw_b.surface"],
        sensor_names=["unit", "unit"], episodes={"0": {
            "maps": "maps.npy", "valid": "valid.npy", "timestamps": "timestamps.npy",
            "layout_present": [True, True]}})
    (t / "manifest.json").write_text(json.dumps(manifest))
    return root, t


def test_causal_history_and_future_are_distinct_measurements(tmp_path):
    root, t = sidecar(tmp_path)
    store = IntensityStore(t / "manifest.json", root, size=4)
    sample = store.window(0, .2)
    # Low-rate sample-and-hold cannot borrow the next native measurement.
    assert sample['tactile_maps'][:, 1, 0, 0, 0].tolist() == pytest.approx([1/12, 1/12, 1/12, 2/12])
    assert sample['tactile_valid'].all()
    assert sample['future_tactile_valid'].all()
    assert sample['future_tactile_maps'][:, 1, 0, 0, 0].tolist() == pytest.approx(np.arange(3, 11)/12)
    assert sample['tactile_valid'][:, 0].all()  # real zero values are observed
    assert sample['tactile_maps'][:, 0].count_nonzero() == 0


def test_missing_values_never_become_labels_or_cross_segments(tmp_path):
    root, t = sidecar(tmp_path)
    maps, valid = np.load(t / 'maps.npy'), np.load(t / 'valid.npy')
    valid[3, 1] = False
    maps[3, 1] = np.nan
    np.save(t / 'maps.npy', maps)
    np.save(t / 'valid.npy', valid)
    store = IntensityStore(t / 'manifest.json', root)
    sample = store.window(0, .2, segment_start_s=.2, segment_end_s=.4)
    assert not sample['tactile_valid'][:3].any()
    assert not sample['future_tactile_valid'][0, 1]
    assert sample['future_tactile_valid'][1].all()
    assert not sample['future_tactile_valid'][2:].any()
    assert torch.isfinite(sample['future_tactile_maps']).all()
    assert sample['tactile_query_mask'].all()  # query and label are independent


def test_manifest_binds_the_dataset_and_forbids_ambiguous_regions(tmp_path):
    root, t = sidecar(tmp_path)
    meta = json.loads((t / 'manifest.json').read_text())
    meta['region_names'][1] = meta['region_names'][0]
    (t / 'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='unique'):
        IntensityStore(t / 'manifest.json', root)
    (root / 'meta/info.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='fingerprint'):
        IntensityStore(t / 'manifest.json', root)


def make_video_dataset(root, *, fps=10):
    av = pytest.importorskip('av')
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'videos/observation.images.camera/chunk-000').mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'fps': fps, 'total_episodes': 2, 'total_frames': 40,
            'splits': {'train': '1:2'},
            'features': {'observation.images.camera': {'dtype': 'video', 'shape': [16, 16, 3],
                                                       'info': {'video.fps': fps}}}}
    (root / 'meta/info.json').write_text(json.dumps(info))
    pq.write_table(pa.table({'episode_index': np.repeat([0, 1], 20),
                            'timestamp': np.tile(np.arange(20)/fps, 2), 'task_index': [0]*40}),
                   root / 'data/chunk-000/file-000.parquet')
    pq.write_table(pa.table({'task_index': [0], 'task': ['Move the object.']}), root / 'meta/tasks.parquet')
    pq.write_table(pa.Table.from_pylist([{'episode_index': i, 'length': 20,
        'data/chunk_index': 0, 'data/file_index': 0, 'dataset_from_index': i*20,
        'dataset_to_index': (i+1)*20, 'videos/observation.images.camera/chunk_index': 0,
        'videos/observation.images.camera/file_index': 0,
        'videos/observation.images.camera/from_timestamp': i*20/fps,
        'videos/observation.images.camera/to_timestamp': (i+1)*20/fps} for i in range(2)]),
        root / 'meta/episodes/chunk-000/file-000.parquet')
    path = root / 'videos/observation.images.camera/chunk-000/file-000.mp4'
    with av.open(str(path), mode='w') as out:
        stream = out.add_stream('libx264', rate=fps)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '0', 'preset': 'ultrafast'}
        for i in range(40):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i*5, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return root


def test_real_v3_video_offset_after_split_and_no_fake_actions(tmp_path):
    root = make_video_dataset(tmp_path / 'data')
    ds = LeRobotVideoIntensityDataset(root, height=48, width=32, tactile_mode='off')
    sample = ds[0]
    assert sample['episode_id'] == 1
    assert sample['training_modality'] == 'video_only'
    assert 'action' not in sample and 'proprio' not in sample
    assert sample['prompt'] == 'Move the object.'
    assert sample['video_mask'].all()
    assert np.asarray(sample['video'][0])[:30].mean() == pytest.approx(100, abs=2)
    assert np.asarray(sample['video'][-1])[:30].mean() == pytest.approx(155, abs=2)
    assert ds.training_data_contract['selected_episodes'] == [1]
    assert not ds[len(ds)-1]['video_mask'][-1]


def test_unverified_fps_is_rejected_before_training(tmp_path):
    root = make_video_dataset(tmp_path / 'data')
    (root / 'meta/source_definition.json').write_text(json.dumps({'physical_time_verified': False}))
    with pytest.raises(ValueError, match='unresolved'):
        LeRobotVideoIntensityDataset(root)


def test_missing_camera_chunks_are_never_observed_zero_targets(tmp_path):
    root = make_video_dataset(tmp_path / 'data')
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path)
    observed = np.ones(40, dtype=bool)
    observed[20:22] = False  # Missing prefix of the selected episode.
    observed[30:32] = False  # Missing interior chunk.
    pq.write_table(table.append_column('observation.images.camera.valid', pa.array(observed)), path)
    info = json.loads((root / 'meta/info.json').read_text())
    info['features']['observation.images.camera.valid'] = {'dtype': 'bool', 'shape': [1]}
    (root / 'meta/info.json').write_text(json.dumps(info))
    ds = LeRobotVideoIntensityDataset(root, height=48, width=32, tactile_mode='off')
    assert len(ds) > 0
    for i in range(len(ds)):
        sample = ds[i]
        start = sample['episode_start_frame']
        requested = start + np.rint(ds.video_offsets_s * ds.fps).astype(np.int64)
        assert observed[20 + requested[requested < 20]].all()
        assert sample['video_mask'][0] and sample['video_mask'][1:].any()


def test_publication_gate_rejects_partial_and_changed_populations(tmp_path):
    from openwam.dataloader.publication import publication_contract
    root = make_video_dataset(tmp_path / 'data')
    report = {'status': 'pending_copy', 'episodes': 2, 'frames': 40}
    pub = root / 'publication.json'
    pub.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not complete'):
        publication_contract(root)
    report['status'] = 'complete'
    pub.write_text(json.dumps(report))
    assert publication_contract(root)['frames'] == 40
    report['frames'] = 39
    pub.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='frames count'):
        publication_contract(root)


class Source(torch.utils.data.Dataset):
    action_dim = 80
    def __init__(self, mode, n):
        self.training_modality, self.n = mode, n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return {'training_modality': self.training_modality, 'value': i}


def test_four_modalities_global_batching_coverage_and_resume():
    modes = ['video_only', 'video_action', 'video_tactile', 'video_action_tactile']
    ds = MixtureDataset([Source(m, n) for m, n in zip(modes, [9, 13, 7, 15])],
                        names=modes, strict_action_dim=True)
    sampler = GlobalModalitySampler(ds, batch_size=2, world_size=2)
    sampler.set_epoch(3)
    indices = list(sampler)
    assert len(indices) == len(sampler)
    assert set(indices) == set(range(len(ds)))
    for start in range(0, len(indices), 4):
        samples = [ds[i] for i in indices[start:start+4]]
        assert batch_modality_code(samples) >= 0
        assert batch_modality_code(samples[:2]) == batch_modality_code(samples[2:])
    sampler.set_epoch(3)
    assert list(sampler)[12:] == indices[12:]
    sampler.set_epoch(4)
    assert list(sampler) != indices
    assert batch_modality_code([ds._datasets[0][0], ds._datasets[1][0]]) == -1


@pytest.mark.parametrize("weights,expected", [(None, [0.1, 0.3, 0.6]), ([2, 1], [1/6, 1/2, 1/3])])
def test_partial_tactile_source_splits_without_changing_parent_weight(weights, expected):
    from openwam.dataloader.touchwam_pretrain import PublishedModalityBuckets
    buckets = PublishedModalityBuckets([Source('video_only', 2), Source('video_tactile', 6)])
    buckets.publication_contract = {'fingerprint': 'checked-full-population'}
    ds = MixtureDataset([buckets, Source('video_action', 12)], weights=weights,
                        names=['optical', 'robot'], strict_action_dim=True,
                        source_configs=[{'dataset_dir': '/optical'}, {'dataset_dir': '/robot'}])
    assert ds.source_modalities == ['video_only', 'video_tactile', 'video_action']
    assert ds.weights == pytest.approx(expected)
    assert ds.datasets[0].publication_contract == buckets.publication_contract
    assert ds.datasets[1].publication_contract == buckets.publication_contract
    assert ds.source_configs[1]['selected_training_modality'] == 'video_tactile'
    sampler = GlobalModalitySampler(ds, batch_size=1, world_size=2)
    indices = list(sampler)
    for start in range(0, len(indices), 2):
        assert batch_modality_code([ds[i] for i in indices[start:start+2]]) >= 0


def test_hour_budgets_use_selected_split_capacity(tmp_path, monkeypatch):
    import openwam.dataloader.utils.lerobotv3 as util
    root = tmp_path / 'published'
    make_video_dataset(root / 'first', fps=10)
    make_video_dataset(root / 'second', fps=20)
    seen = []
    original = util.water_fill_hours
    def capture(capacities, budget):
        seen.extend(capacities)
        return original(capacities, budget)
    monkeypatch.setattr(util, 'water_fill_hours', capture)
    ds = LeRobotVideoIntensityDataset.from_config({'dataset_dir': str(root), 'total_hours': .1,
                                                   'tactile_mode': 'off', 'height': 48, 'width': 32})
    assert seen == pytest.approx([2/3600, 1/3600])
    assert len(ds.buckets) == 2
