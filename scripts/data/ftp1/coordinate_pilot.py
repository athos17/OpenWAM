#!/usr/bin/env python3
"""Bounded one-source-at-a-time staging, CPU-worker handoff and BOS publish.

Run on the dev host with the existing BOS mount. Source data is read-only.
Only this run's temporary work files are removed, after BOS reread checks pass.
Control receipts, selected census and original inputs always remain intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

from publish_bos import checked_path, publish, raise_walk_error, sha256_file, verify_published


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def log(event, **kwargs):
    print(json.dumps({'time': time.time(), 'event': event, **kwargs}), flush=True)


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def logical_path(path):
    """Keep the shared mount spelling; host-specific realpaths are not portable."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def validate_work_scope(control, plan, work, raw, bos_output):
    if work.is_symlink():
        raise ValueError('The owned work directory itself must not be a symlink')
    checked_path(work, 'output')
    checked_path(work, 'staging')
    resolved_work = work.resolve()
    protected = [Path(path).resolve() for path in (control, raw, bos_output, plan['definitions'], plan['weights'])]
    protected.extend(Path(row['reuse_bucket']).resolve() for row in plan['jobs'] if row.get('reuse_bucket'))
    if any(resolved_work.is_relative_to(path) or path.is_relative_to(resolved_work) for path in protected):
        raise ValueError('Work must not overlap control, original/reused input, definitions, weights or BOS')


def source_name(source):
    if not isinstance(source, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', source):
        raise ValueError('Unsafe pilot source name')
    return source


def bytes_under(root):
    total = 0
    for folder, dirs, files in os.walk(root, followlinks=False, onerror=raise_walk_error):
        for name in dirs + files:
            path = Path(folder) / name
            if path.is_symlink():
                raise ValueError(f'Symlink in owned work directory: {path}')
            if name in files:
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    # A worker's atomic metadata replacement may remove .tmp.
                    pass
    return total


def initialize_ownership(args, plan):
    control = logical_path(args.control)
    output = logical_path(plan['output'])
    if output.name != 'output':
        raise ValueError('A dedicated work/output directory is required')
    work = output.parent
    run_id = plan.get('run_id')
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('An explicit immutable run_id is required')
    sources = [source_name(row['source']) for row in plan['jobs']]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError('Pilot sources must be nonempty and unique')
    if args.max_work_bytes <= 0:
        raise ValueError('A positive work byte budget is required')
    validate_work_scope(control, plan, work, args.raw, args.bos_output)
    owner = {'schema': 'touchwam.ftp1.pilot_work_owner.v1', 'run_id': run_id,
             'plan_sha256': plan_hash(plan), 'control': str(control), 'work': str(work),
             'output': str(output), 'raw': str(logical_path(args.raw)),
             'bos_output': str(logical_path(args.bos_output)), 'max_work_bytes': args.max_work_bytes}
    work_owner = work / 'OWNER.json'
    if work.exists() and not work_owner.exists() and any(work.iterdir()):
        raise ValueError('Refusing a nonempty work directory without OWNER.json')
    for path in (work_owner, control / 'OWNER.json'):
        if path.is_symlink():
            raise ValueError('Refusing symlink ownership token')
        if path.exists() and json.loads(path.read_text()) != owner:
            raise ValueError('Work ownership token does not match this run/plan')
    atomic_json(work_owner, owner)
    atomic_json(control / 'OWNER.json', owner)
    return owner


def load_owned_work(control, plan):
    control = logical_path(control)
    path = checked_path(control, 'OWNER.json')
    owner = json.loads(path.read_text())
    output = logical_path(plan['output'])
    work = output.parent
    if (owner.get('plan_sha256') != plan_hash(plan)
            or owner.get('run_id') != plan.get('run_id') or owner.get('control') != str(control)
            or owner.get('work') != str(work) or owner.get('output') != str(output)):
        raise ValueError('Work ownership does not match the fixed plan')
    validate_work_scope(control, plan, work, owner['raw'], owner['bos_output'])
    if json.loads(checked_path(work, 'OWNER.json').read_text()) != owner:
        raise ValueError('Work/control ownership tokens differ')
    return owner


def check_work_budget(control, plan, *, check_error=True):
    owner = load_owned_work(control, plan)
    if check_error and (Path(control) / 'stage_error.json').exists():
        raise RuntimeError('Coordinator failed; stopping the CPU source subprocess')
    size = bytes_under(owner['work'])
    if size > owner['max_work_bytes']:
        raise RuntimeError(f'Temporary work exceeded byte budget: {size} > {owner["max_work_bytes"]}')
    return size


def validate_completion(complete, row, plan):
    source = source_name(row['source'])
    source_dir = Path(plan['output']) / source
    if (complete.get('plan_sha256') != plan_hash(plan) or complete.get('source') != source
            or Path(complete['output']).absolute() != source_dir.absolute()
            or complete.get('frames') != row['group']['frames']
            or complete.get('episodes') != len(row['group']['episode_ends'])):
        raise ValueError('Worker completion does not match the fixed pilot')
    return source_dir


def cleanup_published(control, plan, row, record):
    """Idempotently evict only this run's generated files after BOS verification."""
    owner = load_owned_work(control, plan)
    source = source_name(row['source'])
    validate_completion(record, row, plan)
    expected_bos = Path(owner['bos_output']) / source
    if Path(record['bos']['destination']).resolve() != expected_bos.resolve():
        raise ValueError('Published receipt points outside the owned BOS namespace')
    verify_published(expected_bos, expected_manifest_sha256=record['bos']['manifest_sha256'],
                     max_bytes=owner['max_work_bytes'])
    work = Path(owner['work'])
    manifest_name = record.get('staged_manifest')
    if manifest_name:
        manifest = Path(manifest_name)
        source_staging = checked_path(work, f'staging/{source}')
        if not manifest.is_relative_to(source_staging):
            raise ValueError('Staged cleanup manifest escapes the owned source')
        manifest = checked_path(work, manifest.relative_to(work).as_posix())
        if sha256_file(manifest)[1] != record['staged_manifest_sha256']:
            raise ValueError('Staged cleanup manifest changed')
        cached = json.loads(manifest.read_text())
        if cached['source'] != source or cached['group'] != row['group']:
            raise ValueError('Refusing cleanup of a different sparse source')
        for segment in cached['segments']:
            name = segment['file']
            if not re.fullmatch(r'segment-[0-9]+\.bin', name):
                raise ValueError('Unsafe sparse cleanup segment name')
            path = checked_path(work, (manifest.parent / name).relative_to(work).as_posix())
            path.unlink(missing_ok=True)
        atomic_json(manifest.parent / 'payload_evicted.json', {
            'reason': 'Selected v3/deform publication verified through BOS mount',
            'publication': str(expected_bos / '_BOS_PUBLICATION.json'), 'time': time.time()})
    source_dir = checked_path(work, f'output/{source}')
    # Check every descendant before rmtree: no linked tree can enter cleanup.
    if source_dir.exists():
        bytes_under(source_dir)
        shutil.rmtree(source_dir)
    record['temporary_cleanup_completed'] = True
    atomic_json(Path(control) / 'published' / f'{source}.json', record)
    return record


def _coordinate(args):
    from stage import stage_pilot_group

    control = logical_path(args.control)
    plan = json.loads((control / 'worker_plan.json').read_text())
    fingerprint = plan_hash(plan)
    owner = initialize_ownership(args, plan)
    work = Path(owner['work'])
    staging = work / 'staging'
    # A fresh coordinator resumes this immutable run; previous errors remain
    # in history and cannot poison all future worker attempts through stale ready.
    old_error = control / 'stage_error.json'
    if old_error.exists():
        error = json.loads(old_error.read_text())
        if error.get('plan_sha256') != fingerprint:
            raise ValueError('Stale coordinator error belongs to another plan')
        atomic_json(control / 'error_history' / f'{time.time_ns()}.json', error)
        old_error.unlink()
    records = []
    for row in plan['jobs']:
        source = source_name(row['source'])
        done_path = control / 'published' / f'{source}.json'
        if done_path.exists():
            previous = json.loads(done_path.read_text())
            if previous['plan_sha256'] != fingerprint:
                raise ValueError('Published source belongs to another selection')
            records.append(cleanup_published(control, plan, row, previous))
            continue
        check_work_budget(control, plan)
        ready = {'source': source, 'plan_sha256': fingerprint}
        manifest = None
        try:
            completed_path = control / 'completed' / f'{source}.json'
            ready_path = control / 'ready' / f'{source}.json'
            if completed_path.exists():
                validate_completion(json.loads(completed_path.read_text()), row, plan)
                previous_ready = json.loads(ready_path.read_text())
                if previous_ready.get('plan_sha256') != fingerprint or previous_ready.get('source') != source:
                    raise ValueError('Completion has no matching ready receipt')
                manifest = Path(previous_ready['staged_manifest']) if previous_ready.get('staged_manifest') else None
            else:
                if row.get('reuse_bucket'):
                    ready['reuse_bucket'] = row['reuse_bucket']
                else:
                    manifest = stage_pilot_group(
                        args.raw, staging, source, row['group'],
                        index_path=row.get('index_path'), max_bytes=args.max_work_bytes // 3)
                    ready['staged_manifest'] = str(manifest.absolute())
                error_path = control / 'errors' / f'{source}.json'
                if error_path.exists():
                    error = json.loads(error_path.read_text())
                    if error.get('plan_sha256') != fingerprint:
                        raise ValueError('Worker error belongs to another plan')
                    atomic_json(control / 'error_history' / f'{source}-{time.time_ns()}.json', error)
                    error_path.unlink()
                atomic_json(ready_path, ready)
                log('pilot_input_ready', source=source, temporary_bytes=check_work_budget(control, plan))
            deadline = time.monotonic() + args.wait_seconds
            while not completed_path.exists():
                if (control / 'errors' / f'{source}.json').exists():
                    raise RuntimeError(f'CPU worker failed {source}; inspect control/errors')
                if time.monotonic() > deadline:
                    raise TimeoutError(f'CPU worker timed out: {source}')
                check_work_budget(control, plan)
                time.sleep(5)
            complete = json.loads(completed_path.read_text())
            source_dir = validate_completion(complete, row, plan)
            check_work_budget(control, plan)
            # BOS copies and per-object reread hashes must succeed before any
            # temporary raw payload or generated v3 output is removed.
            receipt = publish(source_dir, Path(args.bos_output) / source,
                              max_bytes=args.max_work_bytes)
            receipt.update(source=str(source_dir), destination=str(logical_path(args.bos_output) / source),
                           manifest=str(logical_path(args.bos_output) / source / '_BOS_PUBLICATION.json'))
            record = {'plan_sha256': fingerprint, **complete, 'bos': receipt,
                      'published_at': time.time(), 'temporary_cleanup_completed': False}
            if manifest:
                record['staged_manifest'] = str(manifest)
                record['staged_manifest_sha256'] = sha256_file(manifest)[1]
            atomic_json(done_path, record)
            records.append(cleanup_published(control, plan, row, record))
            log('pilot_source_published_and_evicted', source=source,
                frames=complete['frames'], temporary_bytes=bytes_under(work))
        except Exception as exc:
            atomic_json(control / 'stage_error.json', {'source': source, 'error': repr(exc),
                                                       'plan_sha256': fingerprint})
            raise
    summary = {'schema': 'touchwam.ftp1.pilot_bos_catalog.v1', 'status': 'complete',
               'scope': 'selected_pilot_episodes', 'excluded_sources': ['sharpa'],
               'plan_sha256': fingerprint, 'sources': records,
               'total_frames': sum(r['frames'] for r in records),
               'total_episodes': sum(r['episodes'] for r in records),
               'deform_training_enabled': False}
    atomic_json(control / 'catalog.json', summary)
    # A standalone small catalog directory gets the same publication protocol.
    catalog_dir = control / 'catalog_publication'
    atomic_json(catalog_dir / 'catalog.json', summary)
    atomic_json(catalog_dir / 'worker_plan.json', plan)
    publish(catalog_dir, Path(args.bos_output) / '_catalog', max_bytes=16 * 1024**2)
    log('pilot_bos_complete', sources=len(records), frames=summary['total_frames'],
        episodes=summary['total_episodes'])


def coordinate(args):
    try:
        return _coordinate(args)
    except (Exception, KeyboardInterrupt) as exc:
        control = logical_path(args.control)
        plan = json.loads((control / 'worker_plan.json').read_text())
        atomic_json(control / 'stage_error.json', {'error': repr(exc), 'plan_sha256': plan_hash(plan)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--bos-output', required=True)
    parser.add_argument('--max-work-bytes', type=int, default=10 * 1024**3)
    parser.add_argument('--wait-seconds', type=int, default=7200)
    coordinate(parser.parse_args())
