"""NCCL bandwidth on 2x8 A800 for the ZeRO-2 vs FSDP2/HSDP extrapolation (P0 step 0).

Launched by run_node.sh with CONFIG=nccl_bench (torchrun, one process per GPU). Groups:
  world  - all 16 ranks (what ZeRO-2 uses today);
  intra  - the 8 ranks of a node (HSDP shard group, NVLink);
  inter  - the 2 ranks with the same local rank (HSDP replicate group, RoCE).
Ops: all_reduce, reduce_scatter_tensor, all_gather_into_tensor on bf16 buffers. busbw uses the
nccl-tests factors (2(n-1)/n for all_reduce, (n-1)/n for the others). Rank 0 writes JSON next to
the node logs.
"""
import json
import os
import time

import torch
import torch.distributed as dist

SIZES_MB = (64, 256, 1024)
ITERS, WARMUP = 10, 3


def bench(op, group, size_mb):
    n = dist.get_world_size(group)
    numel = size_mb * (1 << 20) // 2
    numel -= numel % n
    full = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
    shard = torch.empty(numel // n, dtype=torch.bfloat16, device="cuda")

    def run():
        if op == "all_reduce":
            dist.all_reduce(full, group=group)
        elif op == "reduce_scatter":
            dist.reduce_scatter_tensor(shard, full, group=group)
        else:
            dist.all_gather_into_tensor(full, shard, group=group)

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / ITERS
    nbytes = numel * 2
    factor = 2 * (n - 1) / n if op == "all_reduce" else (n - 1) / n
    return {"op": op, "ranks": n, "size_mb": size_mb, "ms": seconds * 1e3,
            "algbw_gbs": nbytes / seconds / 1e9, "busbw_gbs": nbytes / seconds / 1e9 * factor}


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local)
    groups = {"world": dist.group.WORLD}
    for node in range(world // per_node):  # every rank creates every group, in the same order
        g = dist.new_group(list(range(node * per_node, (node + 1) * per_node)))
        if rank // per_node == node:
            groups["intra"] = g
    for lr in range(per_node):
        g = dist.new_group(list(range(lr, world, per_node)))
        if local == lr:
            groups["inter"] = g
    results = []
    for name, group in groups.items():
        for op in ("all_reduce", "reduce_scatter", "all_gather"):
            for size in SIZES_MB:
                r = bench(op, group, size)
                r["group"] = name
                results.append(r)
                if rank == 0:
                    print("%-6s %-14s %5d MB  %8.2f ms  busbw %6.1f GB/s" % (name, op, size, r["ms"], r["busbw_gbs"]), flush=True)
    if rank == 0:
        out = os.path.join(os.environ.get("PERF_ROOT", "."), os.environ.get("TAG", "nccl_bench") + ".json")
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
