"""Independent audit detects mislabelled or malformed experimental sidecars."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    pytest.importorskip('av')
    pytest.importorskip('pyarrow')
    pytest.importorskip('scipy')
    sys.path.insert(0, str(ROOT / 'scripts/data/ftp1'))
    reconstruction = load('audit_fixture_reconstruct', ROOT / 'scripts/data/ftp1/reconstruct_pilot.py')
    audit = load('audit_fixture_audit', ROOT / 'scripts/data/ftp1/audit_deform_pilot.py')
    fixtures = load('audit_fixture_helpers', ROOT / 'tests/data/test_ftp1_reconstruct_pilot.py')
    return reconstruction, audit, fixtures


@pytest.mark.parametrize('identity_policy', ['strict', 'declared_constant_area'])
def test_real_contract_audit_and_independent_mask_detection(tmp_path, modules, monkeypatch, identity_policy):
    m, audit, fixtures = modules
    bucket = fixtures.fixture_bucket(tmp_path, m)
    episodes_path = bucket / 'meta/episodes/chunk-000/file-000.parquet'
    episodes = m.pq.read_table(episodes_path).to_pylist()
    for row, start in zip(episodes, [20, 90]):
        row['source/from_frame'] = start
        row['source/to_frame'] = start + 3
    m.pq.write_table(m.pa.Table.from_pylist(episodes), episodes_path)
    data_path = bucket / 'data/chunk-000/file-000.parquet'
    data = m.pq.read_table(data_path).to_pydict()
    data['observation.ftp1.source_episode_index'] = [4, 4, 4, 5, 5, 5]
    m.pq.write_table(m.pa.Table.from_pydict(data), data_path)
    monkeypatch.setattr(m, 'load_weights', lambda root: ({}, {'fixture': True}))
    def surface(image, weights, options):
        yy, xx = np.indices((240, 320), dtype=np.float32)
        return (xx / 319 - .5) * (float(image[0, 0, 0]) - 100), {'marker_fraction': 0.}
    monkeypatch.setattr(m.mini, 'surface', surface)
    args = fixtures.args_for(bucket)
    variant = 'mini_v1'
    if identity_policy == 'declared_constant_area':
        _, _, _, _, full, _ = fixtures.constant_area_fixture(bucket, m)
        args.identity_policy = identity_policy
        args.source_archive_index = str(full)
        variant = 'mini_v2'
    m.run(args)
    path = bucket / f'experimental_deform/{variant}/experiment.json'
    report, tiles = audit.audit_experiment(path)
    assert report['episodes'] == 2 and report['surface_frames'] == 6
    assert report['all_train_valid_false']
    assert report['surfaces'][0]['reconstruction_valid'] == 2
    assert report['surfaces'][1]['reconstruction_valid'] == 0
    figures = audit.montage(tmp_path / 'audit.png', tiles)
    assert len(figures) == 1 and Path(figures[0]).is_file()
    cache = path.parent / 'episode-000000/right.gripper.jaw_a.surface'
    valid = np.load(cache / 'train_valid.npy')
    valid[2] = True
    np.save(cache / 'train_valid.npy', valid)
    with pytest.raises(ValueError, match='enabled for training'):
        audit.audit_experiment(path, verify_hashes=False)


def test_pixel_scan_rejects_missing_data_filled_with_zeros(tmp_path, modules):
    _, audit, _ = modules
    np.save(tmp_path / 'signed_height_60x80.npy', np.zeros((2, 60, 80), np.float32))
    np.save(tmp_path / 'positive_preview_32x32.npy', np.zeros((2, 32, 32), np.float32))
    with pytest.raises(ValueError, match='entirely NaN'):
        audit.scan_maps(tmp_path, 2, np.array([True, False]), np.ones(2, bool), np.array([True, False]))


def test_manifest_path_cannot_escape_artifact_directory(tmp_path, modules):
    _, audit, _ = modules
    with pytest.raises(ValueError, match='outside artifact'):
        audit.safe_path(tmp_path, '../escape.npy')
