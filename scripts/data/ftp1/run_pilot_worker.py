#!/usr/bin/env python3
"""Run a fixed CPU rank's small FTP selection, then reconstruct Mini geometry.

The coordinator provides immutable, bounded episode selections and either a
verified reused v3 bucket or a packed sparse archive. Workers never read BOS
credentials or write the BOS publication. Each source has exactly one rank.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from coordinate_pilot import (
    atomic_json, check_work_budget, load_owned_work, log, logical_path, plan_hash, source_name, validate_completion,
)


def existing_completion(control, plan, row):
    """Persistent completion wins over stale ready, even after local eviction."""
    source = source_name(row['source'])
    for directory in ('published', 'completed'):
        path = Path(control) / directory / f'{source}.json'
        if path.exists():
            value = json.loads(path.read_text())
            validate_completion(value, row, plan)
            if directory == 'completed' and not (Path(plan['output']) / source).is_dir():
                raise ValueError('Completed output is missing without a published receipt')
            return value
    return None


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def monitor_process(process, control, plan, *, interval=1.0):
    """Parent controls the whole conversion/ffmpeg process group and its budget."""
    try:
        while process.poll() is None:
            check_work_budget(control, plan)
            time.sleep(interval)
        check_work_budget(control, plan)
        if process.returncode:
            raise RuntimeError(f'CPU source subprocess exited with {process.returncode}')
    except BaseException:
        stop_process(process)
        raise


def execute_source(args, plan, row, ready, rank):
    from convert import convert_group
    from reconstruct_pilot import run as reconstruct
    from validate import validate_bucket

    control = logical_path(args.control)
    source = source_name(row['source'])
    fingerprint = plan_hash(plan)
    check_work_budget(control, plan)
    if existing_completion(control, plan, row):
        return
    result = convert_group((
        '/raw-archive-not-mounted', plan['output'],
        str(Path(plan['definitions']) / f'{source}.json'), row['group'],
        args.encoder_threads, ready.get('staged_manifest'),
        {'reuse_bucket': ready['reuse_bucket']} if ready.get('reuse_bucket') else {}))
    bucket = Path(result['bucket'])
    validation = validate_bucket(bucket, Path(plan['definitions']) / 'regions')
    experiment = reconstruct(SimpleNamespace(
        bucket=str(bucket), output=None, weights=plan['weights'], episodes=10,
        reference_policy='episode-first-candidate', channel_order='RGB',
        xy_mode='normalized_row_col', preview_scale=20.0))
    source_census = {'dataset': source, 'groups': [row['group']],
                     'frames': row['group']['frames'], 'episodes': len(row['group']['episode_ends'])}
    source_dir = Path(plan['output']) / source
    record = {
        'schema_version': 'touchwam.ftp1.pilot_publication.v1',
        'scope': 'selected_pilot_episodes', 'source': source, 'status': 'complete', 'ready': True,
        'training_ready': validation['training_ready'],
        'expected_groups': 1, 'completed_groups': 1, 'validated_groups': 1,
        'frames': validation['frames'], 'episodes': validation['episodes'],
        'physical_time_verified': validation['physical_time_verified'],
        'training_modalities': ['video'], 'tactile_enabled': False,
        'deform_status': experiment['status'], 'deform_training_enabled': False,
        'selected_census': source_census, 'buckets': [bucket.name], 'errors': {}}
    atomic_json(source_dir / 'publication.json', record)
    complete = {'source': source, 'rank': rank, 'plan_sha256': fingerprint,
                'output': str(source_dir), 'bucket': bucket.name,
                'frames': validation['frames'], 'episodes': validation['episodes'],
                'deform_status': experiment['status'], 'completed_at': time.time()}
    validate_completion(complete, row, plan)
    check_work_budget(control, plan)
    # Only the monitoring parent may expose completion to the coordinator,
    # after this process exits and the final budget/error check succeeds.
    atomic_json(control / 'computed' / f'{source}.json', complete)
    log('pilot_source_computed', **complete)


def worker(args):
    control = logical_path(args.control)
    plan = json.loads((control / 'worker_plan.json').read_text())
    rank = int(os.environ.get('RANK', '-1'))
    size = int(os.environ.get('WORLD_SIZE', '0'))
    if size != plan['worker_count'] or not 0 <= rank < size:
        raise ValueError('CPU rank topology differs from the fixed worker plan')
    fingerprint = plan_hash(plan)
    owned = [row for index, row in enumerate(plan['jobs']) if index % size == rank]
    deadline = time.monotonic() + args.wait_seconds
    while not (control / 'OWNER.json').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Coordinator did not initialize work ownership')
        time.sleep(1)
    load_owned_work(control, plan)
    for row in owned:
        source = source_name(row['source'])
        if getattr(args, 'execute_source', None) and args.execute_source != source:
            continue
        if existing_completion(control, plan, row):
            log('pilot_source_resume_skip', source=source, rank=rank)
            continue
        ready_path = control / 'ready' / f'{source}.json'
        deadline = time.monotonic() + args.wait_seconds
        while not ready_path.exists() or (control / 'errors' / f'{source}.json').exists():
            if (control / 'stage_error.json').exists():
                raise RuntimeError('Coordinator staging failed; inspect stage_error.json')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'No bounded input arrived for {source}')
            time.sleep(5)
        ready = json.loads(ready_path.read_text())
        if ready['plan_sha256'] != fingerprint or ready['source'] != source:
            raise ValueError('Staged input identity differs from the worker plan')
        log('pilot_source_started', rank=rank, source=source)
        try:
            if getattr(args, 'execute_source', None):
                execute_source(args, plan, row, ready, rank)
            else:
                check_work_budget(control, plan)
                command = [sys.executable, str(Path(__file__).resolve()), '--control', str(control),
                           '--encoder-threads', str(args.encoder_threads), '--wait-seconds', str(args.wait_seconds),
                           '--execute-source', source]
                process = subprocess.Popen(command, start_new_session=True)
                monitor_process(process, control, plan)
                complete = json.loads((control / 'computed' / f'{source}.json').read_text())
                validate_completion(complete, row, plan)
                atomic_json(control / 'completed' / f'{source}.json', complete)
                log('pilot_source_complete', **complete)
        except Exception as exc:
            atomic_json(control / 'errors' / f'{source}.json', {
                'source': source, 'rank': rank, 'error': repr(exc), 'plan_sha256': fingerprint})
            raise
    log('pilot_rank_complete', rank=rank, sources=len(owned))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True)
    parser.add_argument('--encoder-threads', type=int, default=2)
    parser.add_argument('--wait-seconds', type=int, default=7200)
    parser.add_argument('--execute-source', help=argparse.SUPPRESS)
    worker(parser.parse_args())
