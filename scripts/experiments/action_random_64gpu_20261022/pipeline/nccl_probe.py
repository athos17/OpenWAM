from datetime import timedelta
import json,os,socket,time
from pathlib import Path
import torch
import torch.distributed as dist

R=Path('/mnt/pfs/v89atp/workspace/setup/action_random_64gpu_20261022/results')
torch.set_num_threads(4)
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl',timeout=timedelta(seconds=180))
rank=dist.get_rank();world=dist.get_world_size();assert world==64
hosts=[None]*world;dist.all_gather_object(hosts,socket.gethostname());assert len(set(hosts))==8,hosts
for host in set(hosts):assert hosts.count(host)==8
x=torch.ones(4*1024*1024,device='cuda')*(rank+1)
start=time.monotonic();dist.all_reduce(x);torch.cuda.synchronize()
assert torch.all(x==sum(range(1,world+1)))
seconds=time.monotonic()-start
model=torch.nn.Linear(32,8,device='cuda');model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[int(os.environ['LOCAL_RANK'])])
opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
for _ in range(3):
 opt.zero_grad();loss=model(torch.ones(4,32,device='cuda')).square().mean();loss.backward();opt.step();assert torch.isfinite(loss)
record={'rank':rank,'world_size':world,'hosts':hosts,'device':torch.cuda.get_device_name(),
        'allreduce_bytes':x.numel()*x.element_size(),'allreduce_seconds':seconds,'ddp_optimizer_steps':3,'status':'passed'}
(R/f'nccl_rank{rank}.json').write_text(json.dumps(record,indent=2)+'\n')
print('NCCL_PROBE_PASS',json.dumps(record),flush=True)
dist.barrier();dist.destroy_process_group()
