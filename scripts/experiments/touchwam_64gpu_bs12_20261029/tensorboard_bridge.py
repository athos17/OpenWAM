"""CPU sidecar: append TensorBoard events from production JSONL, without trainer edits.

The train run uses optimizer/microstep axes; the system run uses elapsed seconds.
Only complete JSONL records are consumed. A monitor directory is bound to one run.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def rows(path, state):
    if not path.is_file(): return []
    key = str(path)
    stat = path.stat()
    cursor = state['files'].setdefault(key, dict(inode=stat.st_ino, offset=0))
    if cursor['inode'] != stat.st_ino or stat.st_size < cursor['offset']:
        raise RuntimeError(f'Log replaced/truncated: {path}; use a new monitor directory')
    out = []
    with path.open('rb') as stream:
        stream.seek(cursor['offset'])
        while True:
            line = stream.readline()
            if not line or not line.endswith(b'\n'): break
            out.append(json.loads(line))
            cursor['offset'] = stream.tell()
    return out


def emit(writer, values, step, state):
    summaries = []
    for tag, value in values.items():
        if not isinstance(value, (float, int)): continue
        if not math.isfinite(value):
            state['nonfinite_values'] += 1
            continue
        summaries.append(Summary.Value(tag=tag, simple_value=float(value)))
    summaries.append(Summary.Value(tag='health/nonfinite_values_seen', simple_value=state['nonfinite_values']))
    writer.add_event(Event(wall_time=time.time(), step=int(step), summary=Summary(value=summaries)))


def latest(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        lines = stream.read().splitlines()
    return json.loads(lines[-1]) if lines else None


def run(args):
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir/'bridge.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.state_dir/'state.json'
    identity = dict(output_root=str(args.output_root), reports_dir=str(args.reports_dir),
                    event_dir=str(args.event_dir), global_batch=args.global_batch)
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        identity=identity, run_dir=None, started=time.time(), files={}, last_step=0, nonfinite_values=0)
    assert state['identity'] == identity, 'Use a separate state directory for a different run'
    train = EventFileWriter(str(args.event_dir/'train'), flush_secs=5)
    system = EventFileWriter(str(args.event_dir/'system'), flush_secs=5)
    try:
        while True:
            if state['run_dir'] is None:
                candidates = sorted(args.output_root.glob('*/config.yaml'))
                if len(candidates) > 1: raise RuntimeError('Ambiguous output root; select one run before monitoring')
                if candidates: state['run_dir'] = str(candidates[0].parent)
            if state['run_dir']:
                folder = Path(state['run_dir'])
                for row in rows(folder/'training_metrics.jsonl', state):
                    step = row['step']
                    if step != state['last_step']+1: raise RuntimeError(f'Non-contiguous training step: {step}')
                    values = {'train/learning_rate': row['lr'], 'train/grad_norm': row['grad_norm'],
                              'progress/optimizer_updates': row['opt_step'], 'progress/epoch': row['epoch']}
                    for key, value in row.items():
                        if key.startswith('loss_'): values['loss/'+key[5:]] = value
                        elif key.startswith(('samples_', 'tactile_labeled_')): values['data/'+key] = value
                    if row['steps_per_sec'] > 0:
                        values['speed/seconds_per_step'] = 1/row['steps_per_sec']
                        values['speed/global_samples_per_second'] = args.global_batch*row['steps_per_sec']
                    modes = [k[8:] for k,v in row.items() if k.startswith('samples_video') and v > 0]
                    for mode in modes:
                        for key,value in row.items():
                            if key.startswith('loss_'): values[f'loss_by_modality/{mode}/{key[5:]}'] = value
                    emit(train, values, step, state)
                    state.update(last_step=step, last_metrics=row)
                for row in rows(folder/'step_timing.jsonl', state):
                    values = {}
                    for key, value in row['stages'].items():
                        if key in ['iteration_cpu_s','data_wait_cpu_s','peak_allocated_gib','peak_reserved_gib']:
                            for aggregate, number in value.items(): values[f'timing/{key}/{aggregate}'] = number
                    emit(train, values, row['step'], state)
            resources = []
            for path in args.reports_dir.glob('full_resources_node*.jsonl'):
                try:
                    item = latest(path)
                    if item: resources.append(item)
                except (OSError, ValueError): pass  # A concurrent final line may be incomplete.
            failed_nodes = len(list(args.reports_dir.glob('failure_node*.json')))
            completed_nodes = len(list(args.reports_dir.glob('completed_node*.json')))
            values = {'progress/latest_logged_training_step': state['last_step'],
                      'health/resource_nodes': len(resources),
                      'health/failed_nodes': failed_nodes,
                      'health/completed_nodes': completed_nodes,
                      'health/recent_resource_nodes': sum(time.time()-r['time'] < 60 for r in resources)}
            if state['run_dir']:
                metrics_path = Path(state['run_dir'])/'training_metrics.jsonl'
                if metrics_path.exists():
                    values['health/training_metrics_age_seconds'] = max(0, time.time()-metrics_path.stat().st_mtime)
            for node in resources:
                index = node['node_rank']
                gpus = node.get('gpus', [])
                values[f'node{index}/report_age_seconds'] = max(0, time.time()-node['time'])
                values[f'node{index}/cpu_cores'] = node.get('cpu_cores_used',0)
                if gpus:
                    values[f'node{index}/gpu_memory_max_gib'] = max(g['memory.used'] for g in gpus)/1024
                    values[f'node{index}/gpu_utilization_mean_percent'] = sum(g['utilization.gpu'] for g in gpus)/len(gpus)
            emit(system, values, int(time.time()-state['started']), state)
            train.flush(); system.flush()
            state['updated_at'] = time.time()
            atomic(state_path, state)
            # Publish the terminal health state once; never keep presenting stale
            # GPU observations as an ongoing job after a node reports failure.
            if args.once or failed_nodes or completed_nodes == 8: break
            time.sleep(args.poll_seconds)
    finally:
        train.close(); system.close(); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reports-dir', type=Path, required=True)
    parser.add_argument('--event-dir', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--global-batch', type=int, default=768)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    run(parser.parse_args())
