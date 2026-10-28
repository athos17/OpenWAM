#!/usr/bin/env python3
"""Independent, bounded three-Mini rerun from immutable published v3 inputs.

The dev host prepares/publishes via its existing BOS mount. One CPU job runs
``worker`` on PFS, without a BOS mount or credentials. This never alters the
main pilot's code, plan, source v3, masks, or mini_v1 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from coordinate_pilot import atomic_json, bytes_under, logical_path
from publish_bos import checked_path, publish, sha256_file, verify_published

SOURCES = ('Unit', 'Unit_Bimanual', 'VLA_touch')
BUDGET = 2 * 1024**3
VERSION = 'touchwam.ftp1.mini_v2_rerun.v1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def selected_inputs(manifest, bucket):
    prefix = bucket + '/'
    exact = {'meta/info.json', 'meta/source_definition.json', 'meta/conversion_complete.json', 'meta/archive_index.json'}
    result = []
    for entry in manifest['files']:
        if not entry['path'].startswith(prefix):
            continue
        relative = entry['path'][len(prefix):]
        if relative in exact or relative.startswith(('meta/episodes/', 'data/', 'videos/observation.tactile_rgb.')):
            result.append(dict(entry, input_relative=relative))
    if not exact.issubset({entry['input_relative'] for entry in result}):
        raise ValueError('Published Mini bucket lacks required metadata')
    return result


def copy_checked(source, target, size, checksum, *, hardlink=False):
    if target.exists():
        if sha256_file(target) != (size, checksum):
            raise ValueError(f'Existing minimal input conflicts: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        if sha256_file(source) != (size, checksum):
            raise ValueError('Reusable minimal input changed')
        os.link(source, target)
        if sha256_file(target) != (size, checksum):
            raise ValueError('Reused minimal input failed readback')
        return
    partial = target.with_name(target.name + '.partial')
    with source.open('rb') as src, partial.open('wb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024**2)
    if sha256_file(partial) != (size, checksum):
        raise ValueError(f'Minimal input checksum mismatch: {target}')
    partial.replace(target)  # PFS only; never used to publish BOS objects.


def owner(setup):
    plan = read(setup / 'control/plan.json')
    token = read(setup / 'control/OWNER.json')
    work = logical_path(plan['work'])
    if token != {'schema': VERSION, 'plan_sha256': digest(plan), 'work': str(work)}:
        raise ValueError('Rerun ownership differs from immutable plan')
    if work.is_symlink() or read(work / 'OWNER.json') != token:
        raise ValueError('Rerun work ownership is missing or changed')
    checked_path(work, 'input')
    checked_path(work, 'output')
    return plan, work


def check_budget(setup):
    plan, work = owner(setup)
    size = bytes_under(work)
    if plan.get('prior_work'):
        prior = Path(plan['prior_work'])
        if prior.is_symlink() or read(prior / 'OWNER.json') != plan['prior_work_owner']:
            raise ValueError('Prior attempt work ownership changed')
        size += bytes_under(prior)
    main_bytes = 0
    def scan_error(error):
        if not isinstance(error, FileNotFoundError):
            raise error
    for folder, _, files in os.walk(plan['main_work'], onerror=scan_error, followlinks=False):
        for name in files:
            try:
                main_bytes += (Path(folder) / name).stat().st_size
            except FileNotFoundError:
                pass  # The main coordinator concurrently evicts published sources.
    telemetry_path = setup / 'control/budget_observations.json'
    old = read(telemetry_path) if telemetry_path.exists() else {}
    atomic_json(telemetry_path, {'observed_at': time.time(), 'v2_bytes': size, 'main_bytes': main_bytes,
                                'combined_bytes': size + main_bytes,
                                'peak_v2_bytes': max(size, old.get('peak_v2_bytes', 0)),
                                'peak_combined_bytes': max(size + main_bytes, old.get('peak_combined_bytes', 0)),
                                'check_interval_seconds': 1})
    if size > plan['max_work_bytes']:
        raise RuntimeError(f'Independent v2 work exceeds 2 GiB: {size}')
    if size + main_bytes > plan['max_combined_work_bytes']:
        raise RuntimeError(f'Combined main+v2 work exceeds 10 GiB: {size + main_bytes}')
    return size


def prepare(args):
    setup, main, bos, work = map(logical_path, (args.setup, args.main_setup, args.bos_root, args.work))
    if any(work.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(work.resolve())
           for p in (setup, main, bos)):
        raise ValueError('Independent work must not overlap setup, main pilot, or BOS')
    if not (setup / 'code_manifest.json').exists():
        raise ValueError('Freeze code_v2 and code_manifest.json before preparing inputs')
    main_plan = read(main / 'control/worker_plan.json')
    jobs = {row['source']: row for row in main_plan['jobs']}
    sources = []
    for source in SOURCES:
        published = read(main / 'control/published' / f'{source}.json')
        if published['plan_sha256'] != digest(main_plan) or published['source'] != source:
            raise ValueError('Primary source receipt differs from the selected pilot')
        manifest_path = bos / source / '_BOS_PUBLICATION.json'
        if sha256_file(manifest_path)[1] != published['bos']['manifest_sha256']:
            raise ValueError('BOS source publication manifest differs from trusted receipt')
        manifest = read(manifest_path)
        if manifest.get('complete') is not True:
            raise ValueError('Source v3 publication is incomplete')
        entries = selected_inputs(manifest, published['bucket'])
        if published.get('staged_manifest'):
            full_index = Path(published['staged_manifest']).parent / 'source_index.json'
        else:
            full_index = Path(jobs[source]['reuse_bucket']) / 'meta/archive_index.json'
        index_bytes, index_sha = sha256_file(full_index)
        entries.append({'path': str(full_index), 'input_relative': 'meta/full_source_archive_index.json',
                        'bytes': index_bytes, 'sha256': index_sha, 'full_archive_evidence': True})
        sources.append({'source': source, 'bucket': published['bucket'], 'frames': published['frames'],
                        'episodes': published['episodes'], 'source_publication_sha256': published['bos']['manifest_sha256'],
                        'source_publication_id': manifest['publication_id'], 'inputs': entries})
    total = sum(entry['bytes'] for source in sources for entry in source['inputs'])
    if total > BUDGET:
        raise ValueError('Minimal inputs exceed the independent 2 GiB work budget')
    plan = {'schema': VERSION, 'run_id': 'ftp1-mini-v2-declared-area-20261028', 'work': str(work),
            'bos_root': str(bos), 'sources': sources, 'identity_policy': 'declared_constant_area',
            'weights': str(setup / 'assets/numeric_weights'), 'max_work_bytes': BUDGET,
            'main_work': str(logical_path(main_plan['output']).parent), 'max_combined_work_bytes': 10 * 1024**3,
            'input_bytes': total, 'code_manifest_sha256': sha256_file(setup / 'code_manifest.json')[1],
            'source_plan_sha256': digest(main_plan), 'training_enabled': False}
    if args.reuse_input_root:
        reusable = logical_path(args.reuse_input_root)
        prior = reusable.parent
        if reusable.name != 'input' or prior.resolve().is_relative_to(work.resolve()) or work.resolve().is_relative_to(prior.resolve()):
            raise ValueError('Reusable input must belong to a separate prior attempt work')
        prior_owner = read(prior / 'OWNER.json')
        if prior_owner.get('schema') != VERSION or prior_owner.get('work') != str(prior):
            raise ValueError('Reusable input has no matching prior work ownership')
        plan.update(prior_work=str(prior), prior_work_owner=prior_owner, reuse_input_root=str(reusable))
    token = {'schema': VERSION, 'plan_sha256': digest(plan), 'work': str(work)}
    if work.exists() and any(work.iterdir()) and not (work / 'OWNER.json').exists():
        raise ValueError('Refusing nonempty v2 work without ownership')
    for path, value in ((setup / 'control/plan.json', plan), (setup / 'control/OWNER.json', token), (work / 'OWNER.json', token)):
        if path.exists() and read(path) != value:
            raise ValueError(f'Existing v2 identity differs: {path}')
        atomic_json(path, value)
    for source in sources:
        input_root = checked_path(work, 'input/' + source['source'])
        for entry in source['inputs']:
            src = Path(entry['path']) if entry.get('full_archive_evidence') else checked_path(bos / source['source'], entry['path'])
            if plan.get('reuse_input_root'):
                src = checked_path(Path(plan['reuse_input_root']), source['source'] + '/' + entry['input_relative'])
            target = checked_path(input_root, entry['input_relative'])
            copy_checked(src, target, entry['bytes'], entry['sha256'], hardlink=bool(plan.get('reuse_input_root')))
            check_budget(setup)
        atomic_json(setup / 'control/inputs' / (source['source'] + '.json'), source)
    atomic_json(setup / 'control/prepared.json', {'plan_sha256': digest(plan), 'input_bytes': total, 'complete': True})
    print(json.dumps({'event': 'mini_v2_prepared', 'input_bytes': total, 'plan_sha256': digest(plan)}), flush=True)


def verify_snapshot(setup, plan):
    if sha256_file(setup / 'code_manifest.json')[1] != plan['code_manifest_sha256']:
        raise ValueError('Frozen v2 code manifest changed')
    for entry in read(setup / 'code_manifest.json')['files']:
        if sha256_file(checked_path(setup / 'code_v2', entry['path'])) != (entry['bytes'], entry['sha256']):
            raise ValueError('Frozen v2 code snapshot changed: ' + entry['path'])


def worker(args):
    setup = logical_path(args.setup)
    plan, work = owner(setup)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or int(os.environ.get('RANK', '0')) != 0:
        raise ValueError('The v2 rerun requires exactly one CPU replica')
    verify_snapshot(setup, plan)
    for source in plan['sources']:
        name = source['source']
        completed_path = setup / 'control/completed' / (name + '.json')
        if completed_path.exists():
            if read(completed_path)['plan_sha256'] != digest(plan):
                raise ValueError('Completed v2 source belongs to another plan')
            continue
        input_root = checked_path(work, 'input/' + name)
        output_root = checked_path(work, 'output/' + name)
        for entry in source['inputs']:
            if sha256_file(checked_path(input_root, entry['input_relative'])) != (entry['bytes'], entry['sha256']):
                raise ValueError('Staged v2 input changed')
        command = [sys.executable, str(setup / 'code_v2/scripts/data/ftp1/reconstruct_pilot.py'),
                   '--bucket', str(input_root), '--output', str(output_root), '--weights', plan['weights'],
                   '--episodes', '5', '--reference-policy', 'episode-first-candidate', '--channel-order', 'RGB',
                   '--xy-mode', 'normalized_row_col', '--preview-scale', '20',
                   '--identity-policy', 'declared_constant_area',
                   '--source-archive-index', str(input_root / 'meta/full_source_archive_index.json')]
        check_budget(setup)
        print(json.dumps({'event': 'mini_v2_source_started', 'source': name}), flush=True)
        process = subprocess.Popen(command, start_new_session=True)
        try:
            while process.poll() is None:
                check_budget(setup)
                time.sleep(1)
            check_budget(setup)
            if process.returncode:
                raise RuntimeError(f'Reconstruction failed: {name} / {process.returncode}')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        from audit_deform_pilot import audit_experiment
        report, _ = audit_experiment(output_root / 'experiment.json', True, bucket=input_root)
        if not report['validated'] or not report['all_train_valid_false']:
            raise ValueError('Independent representation audit failed')
        atomic_json(output_root / 'representation_audit.json', report)
        atomic_json(output_root / 'execution_provenance.json', {
            'schema': VERSION, 'plan_sha256': digest(plan), 'source': source,
            'code_manifest_sha256': plan['code_manifest_sha256'],
            'code_snapshot': read(setup / 'code_manifest.json'), 'training_enabled': False})
        summary = read(output_root / 'experiment.json')
        complete = {'plan_sha256': digest(plan), 'source': name, 'output': str(output_root),
                    'frames': summary['frames'], 'episodes': summary['episode_count'],
                    'surface_frames': summary['surface_frames'],
                    'reconstruction_valid_surface_frames': summary['reconstruction_valid_surface_frames'],
                    'training_enabled': False, 'representation_audit_passed': True}
        atomic_json(completed_path, complete)
        print(json.dumps({'event': 'mini_v2_source_complete', **complete}), flush=True)
    atomic_json(setup / 'control/worker_complete.json', {'plan_sha256': digest(plan), 'complete': True})


def publish_results(args):
    setup = logical_path(args.setup)
    plan, work = owner(setup)
    verify_snapshot(setup, plan)
    if read(setup / 'control/worker_complete.json') != {'plan_sha256': digest(plan), 'complete': True}:
        raise ValueError('CPU v2 job has not completed this exact plan')
    records = []
    for source in plan['sources']:
        name = source['source']
        output = checked_path(work, 'output/' + name)
        destination = Path(plan['bos_root']) / '_deform/mini_v2' / name
        receipt_path = setup / 'control/published' / (name + '.json')
        if receipt_path.exists():
            record = read(receipt_path)
            if record['plan_sha256'] != digest(plan):
                raise ValueError('Published v2 source belongs to another plan')
        else:
            complete = read(setup / 'control/completed' / (name + '.json'))
            if complete['plan_sha256'] != digest(plan) or complete['frames'] != source['frames']:
                raise ValueError('CPU source result identity mismatch')
            receipt = publish(output, destination, max_bytes=BUDGET)
            record = dict(complete, bos=receipt, cleaned=False)
            atomic_json(receipt_path, record)
        verify_published(destination, expected_manifest_sha256=record['bos']['manifest_sha256'], max_bytes=BUDGET)
        owner(setup)
        for relative in ('input/' + name, 'output/' + name):
            path = checked_path(work, relative)
            if path.exists():
                bytes_under(path)
                shutil.rmtree(path)
        record['cleaned'] = True
        atomic_json(receipt_path, record)
        records.append(record)
    catalog = {'schema': VERSION, 'status': 'complete_unverified_experiment', 'training_enabled': False,
               'plan': plan, 'sources': records, 'code_manifest_sha256': plan['code_manifest_sha256'],
               'verification': 'BOS POSIX mount reopen SHA256; no independent API or cache bypass',
               'budget_observations': read(setup / 'control/budget_observations.json')}
    folder = setup / 'catalog'
    atomic_json(folder / 'catalog.json', catalog)
    shutil.copyfile(setup / 'code_manifest.json', folder / 'code_manifest.json')
    shutil.copyfile(setup / 'code_v2.tar.gz', folder / 'code_v2.tar.gz')
    if plan.get('prior_work'):
        prior = Path(plan['prior_work'])
        for path in (prior / 'output').rglob('*.json'):
            if path.name in {'experiment.json', 'complete.json', 'plan.json', 'representation_audit.json', 'execution_provenance.json'}:
                target = folder / 'prior_attempt_evidence' / path.relative_to(prior)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        prior_setup = setup.parent
        for name in ('control/error.json', 'control/plan.json', 'cpu_job.json', 'job_status_latest.json', 'code_manifest.json'):
            source_path = prior_setup / name
            if source_path.exists():
                target = folder / 'prior_attempt_evidence' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target)
    receipt = publish(folder, Path(plan['bos_root']) / '_deform/mini_v2/_catalog', max_bytes=16 * 1024**2)
    if plan.get('prior_work'):
        prior = Path(plan['prior_work'])
        if read(prior / 'OWNER.json') != plan['prior_work_owner']:
            raise ValueError('Refusing prior attempt cleanup after ownership change')
        for relative in ('input', 'output'):
            path = checked_path(prior, relative)
            if path.exists():
                bytes_under(path)
                shutil.rmtree(path)
    atomic_json(setup / 'control/complete.json', {'plan_sha256': digest(plan), 'catalog_publication': receipt,
                                               'remaining_work_bytes': bytes_under(work), 'complete': True})
    print(json.dumps({'event': 'mini_v2_published', 'sources': records}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'worker', 'publish'])
    parser.add_argument('--setup', required=True)
    parser.add_argument('--main-setup')
    parser.add_argument('--bos-root')
    parser.add_argument('--work')
    parser.add_argument('--reuse-input-root', help='Verified PFS inputs from a retained failed attempt; hardlink without copying')
    args = parser.parse_args()
    try:
        {'prepare': prepare, 'worker': worker, 'publish': publish_results}[args.operation](args)
    except BaseException as error:
        atomic_json(Path(args.setup) / 'control/error.json', {'operation': args.operation, 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
