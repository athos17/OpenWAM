"""64-rank NCCL all-reduce, DDP step and gloo group check before training (fails fast on a bad node)."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from runtime import ROOT, write_json  # noqa: E402

torch.set_num_threads(4)
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl', timeout=timedelta(seconds=180))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 64
hosts = [None] * world
dist.all_gather_object(hosts, socket.gethostname())
assert len(set(hosts)) == 8 and all(hosts.count(h) == 8 for h in set(hosts)), hosts
x = torch.ones(64 * 1024 * 1024, device='cuda') * (rank + 1)
dist.all_reduce(x)
torch.cuda.synchronize()
start = time.monotonic()
for _ in range(5):
    dist.all_reduce(x)
torch.cuda.synchronize()
seconds = (time.monotonic() - start) / 5
model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(32, 8, device='cuda'),
                                                  device_ids=[int(os.environ['LOCAL_RANK'])])
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(3):
    opt.zero_grad()
    loss = model(torch.ones(4, 32, device='cuda')).square().mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
# training.touchwam_p1.deferred_modality_sync needs a gloo group; these pods have not run one before.
gloo = dist.new_group(backend='gloo')
probe = torch.ones(1)
dist.all_reduce(probe, group=gloo)
assert probe.item() == world
nbytes = x.numel() * x.element_size()
record = {'rank': rank, 'world_size': world, 'hosts': hosts, 'device': torch.cuda.get_device_name(),
          'allreduce_bytes': nbytes, 'allreduce_seconds': seconds,
          'busbw_GBps': 2 * (world - 1) / world * nbytes / seconds / 1e9, 'status': 'passed'}
write_json(ROOT / 'results' / f'nccl_rank{rank}.json', record)
if rank == 0:
    print('NCCL_PROBE_PASS', json.dumps(record), flush=True)
dist.barrier()
dist.destroy_process_group()
