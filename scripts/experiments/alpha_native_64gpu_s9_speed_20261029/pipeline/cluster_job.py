"""Submit / inspect / stop the speed job with dev-cpu's authenticated AIHC client.

Never exports or prints credentials: every API result goes through api.redact.

    cluster_job.py request|create|status|logs|stop
    cluster_job.py wait-submit [--poll-minutes 5] [--timeout-hours 24]
        waits until the baseline formal job (alpha_native_64gpu_aug) has finished, then
        creates the speed job (same resources as the baseline).
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from runtime import BASELINE_SETUP, ROOT, require_preflight, settings, write_json  # noqa: E402

POOL = 'aihc-serverless'
QUEUE = 'aihcq-7crsdg0y5p20'
FINISHED = {'Succeeded', 'Failed', 'ManualTermination', 'Stopped', 'Terminated', 'Completed'}


def job_request(stamp=None):
    stamp = stamp or datetime.now(timezone.utc).strftime('%m%d-%H%M%S')
    return {
        'resourcePoolId': POOL, 'queueID': QUEUE,
        'name': f'openwam-alpha-native-s9-speed-{settings()["steps"]}s-w{settings()["dataset_num_workers"]}-{stamp}',
        'jobType': 'PyTorchJob',
        'command': f'bash {shlex.quote(str(ROOT / "pipeline/cluster_bootstrap.sh"))} speed',
        # Identical to the baseline's request (setup/sharpa_native_64gpu_aug_20261026/results/formal_request.json).
        'jobSpec': {'replicas': 8, 'image': 'registry.baidubce.com/inference/aibox-ubuntu:v2.0-22.04',
                    'resources': [{'name': 'baidu.com/a800_80g_cgpu', 'quantity': 8},
                                  {'name': 'cpu', 'quantity': 116}, {'name': 'memory', 'quantity': 928},
                                  {'name': 'sharedMemory', 'quantity': 128}],
                    'enableRDMA': True, 'hostNetwork': False},
        'dataSources': [{'type': 'pfs', 'name': 'pfs-jVg0EW', 'mountPath': '/mnt/pfs/v89atp',
                         'sourcePath': '/', 'options': {'readOnly': False}}],
        'priority': 'normal', 'enableBccl': False, 'faultTolerance': False, 'retentionPeriod': '7d',
    }


def describe(api, job_id):
    result = api.call('DescribeJob', resourcePoolId=POOL, queueID=QUEUE, jobId=job_id, needDetail=True)
    return json.loads(api.redact(json.dumps(result)))


def create(api):
    require_preflight()
    record = ROOT / 'results/speed_job.json'
    request_file = ROOT / 'results/speed_request.json'
    # A persistent request marker prevents a double submission after an API timeout;
    # an ambiguous response is inspected on the platform using the saved name.
    with (ROOT / 'results/submission.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if record.exists() or request_file.exists():
            raise SystemExit(f'Submission already recorded or in doubt: {request_file}')
        request = job_request()
        write_json(request_file, request)
        result = json.loads(api.redact(json.dumps(api.call('CreateJob', **request))))
        write_json(record, result)
        if not result.get('jobId'):
            raise RuntimeError(f'CreateJob did not return a jobId; inspect {record}')
        print(json.dumps(result), flush=True)


def wait_submit(api, poll_minutes, timeout_hours):
    require_preflight()
    baseline = json.loads((BASELINE_SETUP / 'results/formal_job.json').read_text())['jobId']
    deadline = time.time() + timeout_hours * 3600
    while True:
        job = describe(api, baseline)
        job = job.get('job', job)
        status = job.get('status')
        print(datetime.now(timezone.utc).isoformat(timespec='seconds'), 'baseline', status, flush=True)
        if status in FINISHED or job.get('finishedAt'):
            break
        if time.time() > deadline:
            raise SystemExit('Baseline job still active at the timeout; nothing submitted')
        time.sleep(poll_minutes * 60)
    if status != 'Succeeded':
        print(f'Baseline ended with status {status}; submitting anyway (the queue is free)', flush=True)
    create(api)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=['request', 'create', 'status', 'logs', 'stop', 'wait-submit'])
    p.add_argument('--poll-minutes', type=float, default=5)
    p.add_argument('--timeout-hours', type=float, default=24)
    args = p.parse_args()
    if args.operation == 'request':
        print(json.dumps(job_request('DRY-RUN'), indent=2))
        return
    sys.path.insert(0, '/root/aihc-gpu-smoke-20261022')
    import api
    if args.operation == 'create':
        return create(api)
    if args.operation == 'wait-submit':
        return wait_submit(api, args.poll_minutes, args.timeout_hours)
    job_id = json.loads((ROOT / 'results/speed_job.json').read_text())['jobId']
    if args.operation == 'stop':
        print(api.redact(json.dumps(api.call('StopJob', resourcePoolId=POOL, queueID=QUEUE, jobId=job_id))))
        return
    result = describe(api, job_id)
    details = result.get('job', result)
    if args.operation == 'status':
        write_json(ROOT / 'results/speed_status.json', result)
        compact = {k: details[k] for k in ('jobId', 'name', 'status', 'createdAt', 'finishedAt', 'message') if k in details}
        compact['pods'] = [{k: v for k, v in pod.items() if k in ('name', 'nodeName', 'status', 'reason')}
                           for pod in details.get('pods', [])]
        print(json.dumps(compact))
    else:
        for pod in details.get('pods', []):
            logs = api.call('DescribeJobLogs', resourcePoolId=POOL, queueID=QUEUE, jobId=job_id,
                            podName=pod['name'], maxLines=1000)
            logs = json.loads(api.redact(json.dumps(logs)))
            write_json(ROOT / 'results' / f"speed_{pod['name']}_logs.json", logs)
            print(json.dumps({'pod': pod['name'], 'last_lines': logs.get('logs', [])[-10:]}))


if __name__ == '__main__':
    main()
