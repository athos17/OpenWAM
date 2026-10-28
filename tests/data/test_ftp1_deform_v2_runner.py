"""Minimal-input selection and shared temporary-storage limits for Mini v2."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/data/ftp1'))
    import run_deform_v2
    return run_deform_v2


def test_only_required_small_inputs_are_selected(module):
    desired = ['meta/info.json', 'meta/source_definition.json', 'meta/conversion_complete.json',
               'meta/archive_index.json', 'meta/episodes/chunk-000/file.parquet', 'data/chunk-000/file.parquet',
               'videos/observation.tactile_rgb.right_tactile_data_gripper.0/chunk-000/file.mp4']
    excluded = ['videos/observation.images.camera_main_rgb/chunk-000/file.mp4',
                'experimental_deform/mini_v1/large.npy', 'meta/stats.json']
    manifest = {'files': [{'path': 'bucket/' + p, 'bytes': 1, 'sha256': 'a' * 64} for p in desired + excluded]}
    selected = module.selected_inputs(manifest, 'bucket')
    assert {entry['input_relative'] for entry in selected} == set(desired)


def test_verified_pfs_copy_recovers_partial_but_rejects_final_conflict(module, tmp_path):
    src, dst = tmp_path / 'source', tmp_path / 'input/file'
    src.write_bytes(b'original scientific input')
    size, checksum = module.sha256_file(src)
    dst.parent.mkdir()
    dst.with_name('file.partial').write_bytes(b'old partial')
    module.copy_checked(src, dst, size, checksum)
    assert dst.read_bytes() == src.read_bytes()
    dst.write_bytes(b'!' * size)
    with pytest.raises(ValueError, match='conflicts'):
        module.copy_checked(src, dst, size, checksum)
    assert src.read_bytes() == b'original scientific input'


def test_budget_records_observed_peak_and_rejects_combined_work(module, tmp_path):
    setup, work, main = tmp_path / 'setup', tmp_path / 'work', tmp_path / 'main'
    main.mkdir()
    plan = {'work': str(work), 'main_work': str(main), 'max_work_bytes': 10000, 'max_combined_work_bytes': 10000}
    token = {'schema': module.VERSION, 'plan_sha256': module.digest(plan), 'work': str(work)}
    module.atomic_json(setup / 'control/plan.json', plan)
    module.atomic_json(setup / 'control/OWNER.json', token)
    module.atomic_json(work / 'OWNER.json', token)
    module.check_budget(setup)
    with (main / 'growing').open('wb') as handle:
        handle.truncate(10001)
    with pytest.raises(RuntimeError, match='Combined'):
        module.check_budget(setup)
    observed = json.loads((setup / 'control/budget_observations.json').read_text())
    assert observed['peak_combined_bytes'] > 10000
    assert observed['peak_v2_bytes'] < 10000


def test_ownership_token_change_blocks_work_access(module, tmp_path):
    setup, work = tmp_path / 'setup', tmp_path / 'work'
    plan = {'work': str(work)}
    token = {'schema': module.VERSION, 'plan_sha256': module.digest(plan), 'work': str(work)}
    module.atomic_json(setup / 'control/plan.json', plan)
    module.atomic_json(setup / 'control/OWNER.json', token)
    module.atomic_json(work / 'OWNER.json', dict(token, plan_sha256='wrong'))
    with pytest.raises(ValueError, match='ownership is missing or changed'):
        module.owner(setup)


def test_retry_reuses_verified_inputs_without_copying_or_changing_old_files(module, tmp_path):
    source = tmp_path / 'old/input.bin'
    source.parent.mkdir()
    source.write_bytes(b'unchanged original converted bytes')
    target = tmp_path / 'retry/input.bin'
    size, checksum = module.sha256_file(source)
    module.copy_checked(source, target, size, checksum, hardlink=True)
    assert source.stat().st_ino == target.stat().st_ino
    source.unlink()
    assert module.sha256_file(target) == (size, checksum)
