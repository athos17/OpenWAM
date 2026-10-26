"""Use the existing authenticated AIHC client on dev-cpu. Never export credentials.

One 4-node job per variant (base | s7d); each runs the NCCL probe, then the training run.
"""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,'/root/aihc-gpu-smoke-20261022')
import api
R=Path('/mnt/pfs/v89atp/workspace/setup/accel_cmp_32gpu_20261026')
P='aihc-serverless';Q='aihcq-7crsdg0y5p20'
p=argparse.ArgumentParser();p.add_argument('operation',choices=['create','status','logs','stop']);p.add_argument('variant',choices=['base','s7d']);a=p.parse_args()
if a.operation=='create':
 manifest=json.loads((R/'results/release_manifest.json').read_text())
 for name,expected in manifest['files'].items():
  assert hashlib.sha256((R/'release'/name).read_bytes()).hexdigest()==expected,name
 for name,expected in manifest['runtime_files'].items():
  assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,name
 record=R/'results'/f'{a.variant}_job.json'
 assert not record.exists(),f'Job already recorded: {record}'
 name='openwam-accel-cmp-32gpu-'+a.variant+'-'+datetime.now(timezone.utc).strftime('%m%d-%H%M%S')
 request={'resourcePoolId':P,'queueID':Q,'name':name,'jobType':'PyTorchJob',
  'command':f'bash {R}/pipeline/cluster_bootstrap.sh {a.variant}',
  'jobSpec':{'replicas':4,'image':'registry.baidubce.com/inference/aibox-ubuntu:v2.0-22.04',
     'resources':[{'name':'baidu.com/a800_80g_cgpu','quantity':8},{'name':'cpu','quantity':116},
                  {'name':'memory','quantity':928},{'name':'sharedMemory','quantity':128}],
     'enableRDMA':True,'hostNetwork':False},
  'dataSources':[{'type':'pfs','name':'pfs-jVg0EW','mountPath':'/mnt/pfs/v89atp','sourcePath':'/','options':{'readOnly':False}}],
  'priority':'normal','enableBccl':False,'faultTolerance':False,'retentionPeriod':'7d'}
 (R/'results'/f'{a.variant}_request.json').write_text(json.dumps(request,indent=2)+'\n')
 result=api.call('CreateJob',**request)
 record.write_text(api.redact(json.dumps(result,indent=2))+'\n');print(api.redact(json.dumps(result)))
else:
 job=json.loads((R/'results'/f'{a.variant}_job.json').read_text())['jobId']
 args={'resourcePoolId':P,'queueID':Q,'jobId':job}
 if a.operation=='status':
  result=api.call('DescribeJob',**args,needDetail=True)
  (R/'results'/f'{a.variant}_status.json').write_text(api.redact(json.dumps(result,indent=2))+'\n')
  details=result.get('job',result)
  compact={k:v for k,v in details.items() if k in ('jobId','name','status','createdAt','finishedAt','message','conditions')}
  compact['pods']=[{k:v for k,v in pod.items() if k in ('name','nodeName','status','reason')} for pod in details.get('pods',[])]
  print(api.redact(json.dumps(compact)))
 elif a.operation=='logs':
  result=api.call('DescribeJob',**args,needDetail=True)
  for pod in result.get('job',result).get('pods',[]):
   logs=api.call('DescribeJobLogs',**args,podName=pod['name'],maxLines=1000)
   (R/'results'/f"{a.variant}_{pod['name']}_logs.json").write_text(api.redact(json.dumps(logs,indent=2))+'\n')
   print(api.redact(json.dumps({'pod':pod['name'],'last_lines':logs.get('logs',[])[-10:]})))
 elif a.operation=='stop':print(api.redact(json.dumps(api.call('StopJob',**args))))
