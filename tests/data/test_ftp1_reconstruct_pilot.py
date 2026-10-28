"""Exercise the video/metadata/experimental-output contract, not calibration."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def module():
    pytest.importorskip('av')
    pytest.importorskip('pyarrow')
    pytest.importorskip('scipy')
    path = Path(__file__).resolve().parents[2] / 'scripts/data/ftp1/reconstruct_pilot.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('ftp_reconstruct_test', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fixture_bucket(tmp_path, m):
    bucket = tmp_path / 'bucket'
    (bucket / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (bucket / 'data' / 'chunk-000').mkdir(parents=True)
    feature = 'observation.tactile_rgb.right_tactile_data_gripper.0'
    video = bucket / 'videos' / feature / 'chunk-000' / 'file-000.mp4'
    video.parent.mkdir(parents=True)
    with m.av.open(str(video), 'w') as out:
        stream = out.add_stream('libx264rgb', rate=10)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = 'rgb24'
        stream.options = {'crf': '0', 'preset': 'ultrafast'}
        for i in range(6):
            frame = m.av.VideoFrame.from_ndarray(np.full((240, 320, 3), 100 + i, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    info = {'source_id': 'ftp1/Unit', 'fps': 10, 'codebase_version': 'v3.0',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'features': {feature: {'dtype': 'video', 'shape': [240, 320, 3]}}}
    (bucket / 'meta' / 'info.json').write_text(json.dumps(info))
    (bucket / 'meta' / 'conversion_complete.json').write_text('{}')
    surface = {'raw_field': 'right_tactile_data_gripper', 'raw_sensor_index': 0,
               'canonical_region': 'right.gripper.jaw_a.surface', 'sensor': 'GelSightMini', 'expected_ftp_area_id': 0}
    definition = {'tactile_region_definition': {'registry_version': m.REGISTRY_VERSION, 'surfaces': [surface]}}
    (bucket / 'meta' / 'source_definition.json').write_text(json.dumps(definition))
    data = {'index': list(range(6)), 'episode_index': [0, 0, 0, 1, 1, 1],
            'frame_index': [0, 1, 2, 0, 1, 2], 'timestamp': [0., .1, .2, 0., .1, .2],
            feature + '.valid': [True, False, True, False, True, True],
            'observation.ftp1.source_frame_index': [20, 21, 22, 90, 91, 92]}
    for kind, value in [('area', [0]), ('sensor', 'GelSightMini'), ('type', 'image')]:
        field = 'observation.ftp1.right_tactile_' + kind + '_gripper'
        data[field] = [value] * 6
        data[field + '.valid'] = [True] * 6
    m.pq.write_table(m.pa.Table.from_pydict(data), bucket / 'data' / 'chunk-000' / 'file-000.parquet')
    episodes = []
    for i in range(2):
        episodes.append({'episode_index': i, 'length': 3, 'data/chunk_index': 0, 'data/file_index': 0,
                         'source/episode_index': i + 4, 'source/group': 'test',
                         f'videos/{feature}/chunk_index': 0, f'videos/{feature}/file_index': 0,
                         f'videos/{feature}/from_timestamp': i * .3})
    m.pq.write_table(m.pa.Table.from_pylist(episodes), bucket / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    return bucket


def args_for(bucket):
    return argparse.Namespace(bucket=str(bucket), output=None, weights='unused', episodes=5,
                              reference_policy='episode-first-candidate', channel_order='RGB',
                              xy_mode='normalized_row_col', preview_scale=20.)


def test_full_stream_masks_origin_and_safe_resume(tmp_path, module, monkeypatch):
    m = module
    bucket = fixture_bucket(tmp_path, m)
    monkeypatch.setattr(m, 'load_weights', lambda root: ({}, {'fixture': True}))
    def surface(image, weights, options):
        yy, xx = np.indices((240, 320), dtype=np.float32)
        height = (xx / 319 - .5) * (float(image[0, 0, 0]) - 100)
        return height, {'marker_fraction': 0.}
    monkeypatch.setattr(m.mini, 'surface', surface)
    result = m.run(args_for(bucket))
    assert result['episode_count'] == 2
    assert result['surface_frames'] == 6
    assert result['reconstruction_valid_surface_frames'] == 2
    assert result['train_valid'] is False
    root = bucket / 'experimental_deform' / 'mini_v1'
    one = root / 'episode-000000' / 'right.gripper.jaw_a.surface'
    two = root / 'episode-000001' / 'right.gripper.jaw_a.surface'
    np.testing.assert_array_equal(np.load(one / 'source_valid.npy'), [True, False, True])
    np.testing.assert_array_equal(np.load(one / 'reconstruction_valid.npy'), [True, False, True])
    assert not np.load(one / 'train_valid.npy').any()
    maps = np.load(one / 'signed_height_60x80.npy')
    assert not maps[0].any()
    assert np.isnan(maps[1]).all()
    assert np.isfinite(maps[2]).all() and maps[2].max() > 0
    assert np.isnan(np.load(two / 'positive_preview_32x32.npy')).all()
    complete = json.loads((two / 'complete.json').read_text())
    assert complete['reference_confirmed_unloaded'] is False
    assert complete['source_episode_index'] == 5
    frame = m.pq.read_table(one / 'frame_index.parquet')
    assert frame['observation.ftp1.source_frame_index'].to_pylist() == [20, 21, 22]
    assert not (bucket / 'tactile').exists()
    before = (one / 'signed_height_60x80.npy').stat().st_mtime_ns
    assert m.run(args_for(bucket)) == result
    assert (one / 'signed_height_60x80.npy').stat().st_mtime_ns == before
    with (one / 'source_valid.npy').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        m.run(args_for(bucket))


def test_fixed_scale_does_not_turn_weak_frames_into_strong_frames(module):
    zero = np.zeros((240, 320), np.float32)
    _, weak, _ = module.low_resolution_maps(zero + 2, zero, 20)
    _, strong, _ = module.low_resolution_maps(zero + 10, zero, 20)
    np.testing.assert_allclose(weak, .1, atol=1e-6)
    np.testing.assert_allclose(strong, .5, atol=1e-6)


def test_unsupported_sensor_never_uses_mini_network(tmp_path, module, monkeypatch):
    bucket = fixture_bucket(tmp_path, module)
    path = bucket / 'meta' / 'info.json'
    info = json.loads(path.read_text())
    info['source_id'] = 'ftp1/ViTaMIn'
    path.write_text(json.dumps(info))
    def forbidden(*args):
        raise AssertionError('Mini network must not be loaded for Fin Ray')
    monkeypatch.setattr(module, 'load_weights', forbidden)
    result = module.run(args_for(bucket))
    assert result['status'] == 'unsupported_optical_calibration'
    assert 'Fin Ray' in result['reason']
    assert not result['train_valid']


def test_experiment_cannot_overwrite_formal_tactile_cache(tmp_path, module):
    bucket = fixture_bucket(tmp_path, module)
    args = args_for(bucket)
    args.output = str(bucket / 'tactile')
    with pytest.raises(ValueError, match='formal tactile'):
        module.run(args)


def constant_area_fixture(bucket, m):
    path = bucket / 'meta/archive_index.json'
    index = {'archive_fingerprint': {'name': 'source.tar', 'size': 100, 'mtime_ns': 123},
             'arrays': {'data/right_tactile_area_gripper': {'config': {'zarr_format': 2,
                 'shape': [100, 1], 'chunks': [3, 1], 'dtype': '<i8', 'fill_value': 0}, 'chunks': {}}}}
    path.write_text(json.dumps(index))
    full = bucket / 'meta/full_source_archive_index.json'
    full.write_text(json.dumps(index))
    data_path = bucket / 'data/chunk-000/file-000.parquet'
    data = m.pq.read_table(data_path).to_pydict()
    data['observation.ftp1.right_tactile_area_gripper.valid'] = [False] * 6
    m.pq.write_table(m.pa.Table.from_pydict(data), data_path)
    info = json.loads((bucket / 'meta/info.json').read_text())
    definition = json.loads((bucket / 'meta/source_definition.json').read_text())
    episode = m.episode_rows(bucket)[0]
    surface = definition['tactile_region_definition']['surfaces'][0]
    feature = 'observation.tactile_rgb.right_tactile_data_gripper.0'
    return info, episode, surface, feature, full, data_path


def test_declared_area_preserves_original_validity_and_requires_full_proof(tmp_path, module):
    bucket = fixture_bucket(tmp_path, module)
    info, episode, surface, feature, full, data_path = constant_area_fixture(bucket, module)
    data, _, valid, derived, proof = module.episode_metadata(bucket, info, episode, feature, surface, 'declared_constant_area', full)
    assert valid.tolist() == [True, True, True] and derived.tolist() == [True, True, True]
    assert data['observation.ftp1.right_tactile_area_gripper.valid'] == [False] * 3
    assert proof['full_source_archive_index_sha256'] == module.sha256(full)
    with pytest.raises(ValueError, match='complete source archive index'):
        module.episode_metadata(bucket, info, episode, feature, surface, 'declared_constant_area')
    index = json.loads(full.read_text())
    index['arrays']['data/right_tactile_area_gripper']['chunks'] = {'8.0': {'offset': 100, 'size': 10}}
    full.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='no stored chunks'):
        module.episode_metadata(bucket, info, episode, feature, surface, 'declared_constant_area', full)


def test_declared_area_rejects_observed_wrong_area_and_other_profile(tmp_path, module):
    bucket = fixture_bucket(tmp_path, module)
    info, episode, surface, feature, full, data_path = constant_area_fixture(bucket, module)
    data = module.pq.read_table(data_path).to_pydict()
    data['observation.ftp1.right_tactile_area_gripper.valid'][0] = True
    data['observation.ftp1.right_tactile_area_gripper'][0] = [1]
    module.pq.write_table(module.pa.Table.from_pydict(data), data_path)
    with pytest.raises(ValueError, match='area disagrees'):
        module.episode_metadata(bucket, info, episode, feature, surface, 'declared_constant_area', full)
    with pytest.raises(ValueError, match='not registered'):
        module.declared_area_evidence(bucket, info, {**surface, 'canonical_region': 'right.gripper.jaw_b.surface'}, full)


def test_declared_area_does_not_repair_missing_sensor_or_type(tmp_path, module):
    bucket = fixture_bucket(tmp_path, module)
    info, episode, surface, feature, full, data_path = constant_area_fixture(bucket, module)
    data = module.pq.read_table(data_path).to_pydict()
    data['observation.ftp1.right_tactile_sensor_gripper.valid'][0] = False
    data['observation.ftp1.right_tactile_type_gripper.valid'][1] = False
    module.pq.write_table(module.pa.Table.from_pydict(data), data_path)
    _, _, valid, derived, _ = module.episode_metadata(bucket, info, episode, feature, surface, 'declared_constant_area', full)
    assert valid.tolist() == [False, False, True] and derived.tolist() == [False, False, True]


def test_saturated_area_resize_enforces_exact_float32_bounds(module):
    zero = np.zeros((240, 320), np.float32)
    _, saturated, _ = module.low_resolution_maps(zero + 40, zero, 20)
    # OpenCV INTER_AREA produces 1.000000119 for this ratio without the
    # post-resize boundary clamp. The representation promises exact [0,1].
    np.testing.assert_array_equal(saturated, np.ones((32, 32), np.float32))
    half = zero.copy()
    half[:, 160:] = 10
    expected = module.cv2.resize(np.clip(half / 20, 0, 1), (32, 32), interpolation=module.cv2.INTER_AREA)
    _, actual, _ = module.low_resolution_maps(half, zero, 20)
    np.testing.assert_array_equal(actual, expected)  # All already in-range values stay bitwise unchanged.
