"""Ownership, post-publication eviction, and CPU budget supervisor fixtures."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/data/ftp1'))
    import coordinate_pilot as coordinator
    import publish_bos
    import run_pilot_worker as worker

    control = tmp_path / 'control'
    control.mkdir()
    plan = {'run_id': 'tiny-test', 'worker_count': 1,
            'output': str(tmp_path / 'work' / 'output'),
            'definitions': str(tmp_path / 'definitions'), 'weights': str(tmp_path / 'weights'),
            'jobs': [{'source': 'Tiny', 'group': {'group': 'Tiny/task.zarr', 'frames': 2, 'episode_ends': [2]},
                      'reuse_bucket': str(tmp_path / 'reuse')}]}
    args = SimpleNamespace(control=str(control), raw=str(tmp_path / 'raw'),
                           bos_output=str(tmp_path / 'bos'), max_work_bytes=1024**2, wait_seconds=1)
    coordinator.atomic_json(control / 'worker_plan.json', plan)
    for name in ('raw', 'reuse'):
        path = tmp_path / name
        path.mkdir()
        (path / 'untouched').write_text('original')
    coordinator.initialize_ownership(args, plan)
    return coordinator, publish_bos, worker, args, plan


def make_publication(owned, *, staged=False):
    coordinator, publisher, _, args, plan = owned
    row = plan['jobs'][0]
    source_dir = Path(plan['output']) / row['source']
    source_dir.mkdir(parents=True)
    (source_dir / 'fixture.bin').write_bytes(b'scientific-fixture')
    receipt = publisher.publish(source_dir, Path(args.bos_output) / row['source'], max_bytes=args.max_work_bytes)
    record = {'source': row['source'], 'plan_sha256': coordinator.plan_hash(plan),
              'output': str(source_dir), 'frames': 2, 'episodes': 1,
              'bos': receipt, 'temporary_cleanup_completed': False, 'published_at': 1.0}
    if staged:
        folder = Path(plan['output']).parent / 'staging' / row['source'] / 'selection'
        folder.mkdir(parents=True)
        segments = []
        for i in range(2):
            path = folder / f'segment-{i:05d}.bin'
            path.write_bytes(b'packed-source')
            segments.append({'file': path.name})
        manifest = folder / 'manifest.json'
        coordinator.atomic_json(manifest, {'source': row['source'], 'group': row['group'], 'segments': segments})
        record.update(staged_manifest=str(manifest), staged_manifest_sha256=publisher.sha256_file(manifest)[1])
    coordinator.atomic_json(Path(args.control) / 'published' / 'Tiny.json', record)
    return record


def test_owner_tokens_match_and_nonempty_unknown_work_is_rejected(owned, tmp_path):
    coordinator, _, _, args, plan = owned
    owner = coordinator.load_owned_work(args.control, plan)
    assert owner['schema'] == 'touchwam.ftp1.pilot_work_owner.v1'
    assert json.loads((Path(owner['work']) / 'OWNER.json').read_text()) == owner
    other = tmp_path / 'unknown' / 'output'
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match='nonempty work directory'):
        coordinator.initialize_ownership(args, dict(plan, output=str(other)))


@pytest.mark.parametrize('protected', ['control', 'raw', 'reuse', 'definitions', 'weights', 'bos'])
def test_owner_refuses_cleanup_scope_overlapping_protected_inputs(owned, tmp_path, protected):
    coordinator, _, _, args, plan = owned
    overlapping = tmp_path / protected / 'nested' / 'output'
    with pytest.raises(ValueError, match='must not overlap'):
        coordinator.initialize_ownership(args, dict(plan, output=str(overlapping)))


def test_cleanup_resumes_after_one_sparse_segment_was_already_evicted(owned):
    coordinator, _, _, args, plan = owned
    record = make_publication(owned, staged=True)
    manifest = Path(record['staged_manifest'])
    (manifest.parent / 'segment-00000.bin').unlink()
    result = coordinator.cleanup_published(args.control, plan, plan['jobs'][0], record)
    assert result['temporary_cleanup_completed']
    assert not Path(record['output']).exists()
    assert not (manifest.parent / 'segment-00001.bin').exists()
    assert manifest.exists()
    assert (Path(args.raw) / 'untouched').read_text() == 'original'
    assert (Path(plan['jobs'][0]['reuse_bucket']) / 'untouched').read_text() == 'original'
    coordinator.cleanup_published(args.control, plan, plan['jobs'][0], result)


def test_bad_bos_payload_prevents_local_eviction(owned):
    coordinator, _, _, args, plan = owned
    record = make_publication(owned, staged=True)
    payload = Path(args.bos_output) / 'Tiny' / 'fixture.bin'
    payload.write_bytes(b'!' * payload.stat().st_size)
    with pytest.raises(ValueError, match='checksum conflict'):
        coordinator.cleanup_published(args.control, plan, plan['jobs'][0], record)
    assert Path(record['output']).exists()
    assert (Path(record['staged_manifest']).parent / 'segment-00000.bin').exists()


def test_bad_owner_or_output_symlink_prevents_eviction(owned, tmp_path):
    coordinator, _, _, args, plan = owned
    record = make_publication(owned)
    owner_path = Path(plan['output']).parent / 'OWNER.json'
    original = owner_path.read_text()
    owner_path.write_text('{}')
    with pytest.raises(ValueError, match='ownership tokens differ'):
        coordinator.cleanup_published(args.control, plan, plan['jobs'][0], record)
    assert Path(record['output']).exists()
    owner_path.write_text(original)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'sentinel').write_text('keep')
    (Path(record['output']) / 'linked').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        coordinator.cleanup_published(args.control, plan, plan['jobs'][0], record)
    assert (outside / 'sentinel').read_text() == 'keep'


def test_coordinator_resumes_published_receipt_and_catalog_without_restaging(owned, monkeypatch):
    coordinator, _, _, args, plan = owned
    record = make_publication(owned, staged=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('Published recovery must not restage source payload')

    monkeypatch.setitem(sys.modules, 'stage', SimpleNamespace(stage_pilot_group=forbidden))
    coordinator.coordinate(args)
    assert not Path(record['output']).exists()
    done = json.loads((Path(args.control) / 'published' / 'Tiny.json').read_text())
    assert done['temporary_cleanup_completed']
    catalog = json.loads((Path(args.control) / 'catalog.json').read_text())
    assert catalog['total_episodes'] == 1 and catalog['total_frames'] == 2
    coordinator.coordinate(args)
    assert json.loads((Path(args.control) / 'catalog.json').read_text()) == catalog


def test_worker_ignores_stale_ready_after_published_source_eviction(owned, monkeypatch):
    coordinator, _, worker, args, plan = owned
    record = make_publication(owned)
    coordinator.cleanup_published(args.control, plan, plan['jobs'][0], record)
    coordinator.atomic_json(Path(args.control) / 'ready' / 'Tiny.json',
                            {'source': 'Tiny', 'plan_sha256': coordinator.plan_hash(plan)})
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '1')

    def forbidden(*args, **kwargs):
        raise AssertionError('Stale ready must never start a conversion subprocess')

    monkeypatch.setattr(worker.subprocess, 'Popen', forbidden)
    worker.worker(SimpleNamespace(control=args.control, encoder_threads=1, wait_seconds=1))
    assert not Path(record['output']).exists()


@pytest.mark.parametrize('cause', ['budget', 'coordinator_error'])
def test_worker_supervisor_stops_real_child_process_on_budget_or_error(owned, cause):
    coordinator, _, worker, args, plan = owned
    if cause == 'budget':
        (Path(plan['output']).parent / 'too-large.bin').write_bytes(b'x' * (args.max_work_bytes + 1))
    else:
        coordinator.atomic_json(Path(args.control) / 'stage_error.json', {'plan_sha256': coordinator.plan_hash(plan)})
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    begin = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match='byte budget|Coordinator failed'):
            worker.monitor_process(process, args.control, plan, interval=0.01)
        assert process.poll() is not None
        assert time.monotonic() - begin < 10
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 9)
            process.wait()


def test_mount_root_alias_preserves_portable_owner_and_cleanup_on_another_host(owned, tmp_path, monkeypatch):
    coordinator, publisher, worker, _, _ = owned
    first_host = tmp_path / 'host-a'
    first_host.mkdir()
    mount = tmp_path / 'shared-mount'
    mount.symlink_to(first_host, target_is_directory=True)
    control = mount / 'control'
    control.mkdir()
    plan = {'run_id': 'portable', 'worker_count': 1,
            'output': str(mount / 'work' / 'output'), 'definitions': str(mount / 'definitions'),
            'weights': str(mount / 'weights'),
            'jobs': [{'source': 'Tiny', 'group': {'group': 'Tiny/task.zarr', 'frames': 2, 'episode_ends': [2]},
                      'reuse_bucket': str(mount / 'reuse')}]}
    args = SimpleNamespace(control=str(control), raw=str(mount / 'raw'), bos_output=str(mount / 'bos'),
                           max_work_bytes=1024**2, wait_seconds=1)
    coordinator.atomic_json(control / 'worker_plan.json', plan)
    owner = coordinator.initialize_ownership(args, plan)
    assert owner['work'] == str(mount / 'work')
    assert owner['control'] == str(control)
    portable = coordinator, publisher, worker, args, plan
    record = make_publication(portable, staged=True)
    record['bos']['destination'] = str(mount / 'bos' / 'Tiny')
    coordinator.atomic_json(control / 'published' / 'Tiny.json', record)
    # The same portable mount spelling resolves to a different backing path
    # on the worker host; no host-a pathname is valid as shared ownership.
    second_host = tmp_path / 'host-b'
    shutil.copytree(first_host, second_host)
    mount.unlink()
    mount.symlink_to(second_host, target_is_directory=True)
    assert coordinator.load_owned_work(control, plan) == owner
    coordinator.cleanup_published(control, plan, plan['jobs'][0], record)
    assert not (mount / 'work' / 'output' / 'Tiny').exists()
    assert (first_host / 'work' / 'output' / 'Tiny').exists()
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '1')
    worker.worker(SimpleNamespace(control=str(control), encoder_threads=1, wait_seconds=1))
