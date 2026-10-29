"""Submit once; authenticated resource-pool calls stay on dev-cpu."""
import argparse
import fcntl
import json
import sys

from runtime import ROOT, require_preflight, write_json

POOL, QUEUE = 'aihc-serverless', 'aihcq-7crsdg0y5p20'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['create','status','logs','stop'])
    args = parser.parse_args()
    sys.path.insert(0,'/root/aihc-gpu-smoke-20261022')
    import api
    def call(operation, **kw):
        return json.loads(api.redact(json.dumps(api.call(operation, **kw))))
    if args.operation == 'create':
        require_preflight()
        request = dict(resourcePoolId=POOL, queueID=QUEUE,
            name=ROOT.name.replace('_','-'), jobType='PyTorchJob',
            command=f'bash {ROOT}/pipeline/cluster_bootstrap.sh',
            jobSpec=dict(replicas=8,image='registry.baidubce.com/inference/aibox-ubuntu:v2.0-22.04',
                resources=[dict(name='baidu.com/a800_80g_cgpu',quantity=8),dict(name='cpu',quantity=116),
                           dict(name='memory',quantity=928),dict(name='sharedMemory',quantity=128)],
                enableRDMA=True,hostNetwork=False),
            dataSources=[dict(type='pfs',name='pfs-jVg0EW',mountPath='/mnt/pfs/v89atp',
                              sourcePath='/',options=dict(readOnly=False))],
            priority='normal',enableBccl=False,faultTolerance=False,retentionPeriod='7d')
        with (ROOT/'results/submission.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            if (ROOT/'results/full_request.json').exists():
                raise RuntimeError('Submission already recorded or ambiguous; inspect before retrying')
            write_json(ROOT/'results/full_request.json',request)
            result=call('CreateJob',**request)
            write_json(ROOT/'results/full_job.json',result)
            if not result.get('jobId'): raise RuntimeError('No jobId returned; inspect saved response')
        print(json.dumps(result)); return
    job_id=json.loads((ROOT/'results/full_job.json').read_text())['jobId']
    if args.operation == 'stop':
        print(json.dumps(call('StopJob',resourcePoolId=POOL,queueID=QUEUE,jobId=job_id))); return
    result=call('DescribeJob',resourcePoolId=POOL,queueID=QUEUE,jobId=job_id,needDetail=True)
    write_json(ROOT/'results/full_status.json',result)
    job=result.get('job',result)
    if args.operation == 'status':
        record={k:job.get(k) for k in ['jobId','name','status','createdAt','finishedAt','message']}
        record['pods']=[{k:p.get(k) for k in ['name','status','reason','nodeName']} for p in job.get('pods',[])]
        print(json.dumps(record)); return
    for pod in job.get('pods',[]):
        logs=call('DescribeJobLogs',resourcePoolId=POOL,queueID=QUEUE,jobId=job_id,podName=pod['name'],maxLines=1000)
        write_json(ROOT/'results'/f"logs_{pod['name']}.json",logs)
        print(json.dumps(dict(pod=pod['name'],logs=logs.get('logs',[])[-8:])))


if __name__ == '__main__': main()
