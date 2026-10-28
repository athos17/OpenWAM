#!/usr/bin/env python3
"""Submit/inspect the one-replica Mini v2 CPU job through the existing client.

Run on dev-cpu. Authentication stays inside the existing API module. A durable
submission intent prevents an uncertain API response from creating duplicate
jobs; such an intent must be reconciled externally before another submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['create', 'status', 'logs'])
    parser.add_argument('--setup', required=True)
    args = parser.parse_args()
    setup = Path(args.setup)
    sys.path.insert(0, '/root/aihc-gpu-smoke-20261022')
    import api
    record = setup / 'cpu_job.json'
    request = json.loads((setup / 'cpu_request.json').read_text())
    if args.operation == 'create':
        if record.exists():
            print(json.dumps({'already_submitted': True, 'jobId': json.loads(record.read_text())['jobId']}))
            return
        approval = json.loads((setup / 'control/precheck_approved.json').read_text())
        prepared = json.loads((setup / 'control/prepared.json').read_text())
        if approval['passed'] is not True or approval['plan_sha256'] != prepared['plan_sha256']:
            raise ValueError('A passing real-input precheck for this exact prepared plan is required')
        intent = setup / 'submission_intent.json'
        with intent.open('x') as handle:
            json.dump({'plan_sha256': prepared['plan_sha256'], 'request': request, 'time': time.time(),
                       'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, handle, indent=2)
        result = api.call('CreateJob', **request)
        record.write_text(api.redact(json.dumps(result, indent=2)) + '\n')
        print(api.redact(json.dumps(result)))
        return
    job_id = json.loads(record.read_text())['jobId']
    query = {'resourcePoolId': request['resourcePoolId'], 'queueID': request['queueID'], 'jobId': job_id}
    result = api.call('DescribeJob', **query, needDetail=True)
    job = result.get('job', result)
    if args.operation == 'status':
        (setup / 'job_status_latest.json').write_text(api.redact(json.dumps(result, indent=2)) + '\n')
        summary = {k: v for k, v in job.items() if k in ('jobId', 'name', 'status', 'createdAt', 'finishedAt', 'message')}
        summary['pods'] = [{k: v for k, v in pod.items() if k in ('name', 'status', 'reason')} for pod in job.get('pods', [])]
        print(api.redact(json.dumps(summary)))
    else:
        for pod in job.get('pods', []):
            logs = api.call('DescribeJobLogs', **query, podName=pod['name'], maxLines=100)
            print(api.redact(json.dumps({'pod': pod['name'], 'logs': logs.get('logs', [])[-15:]})))


if __name__ == '__main__':
    main()
