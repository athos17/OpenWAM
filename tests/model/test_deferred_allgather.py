"""training.overlap_param_allgather: async ZeRO-1/2 parameter all-gather, waited before reuse."""

import os
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from openwam.train.utils import deferred_allgather


def _worker(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        group = dist.new_group(ranks=list(range(world)))  # DeepSpeed's cloned world group
        shard = 4
        flat = torch.zeros(world * shard)
        partitions = [flat.narrow(0, r * shard, shard) for r in range(world)]
        partitions[rank].copy_(torch.arange(shard, dtype=torch.float32) + 10 * rank)
        deferred_allgather._all_gather_dp_groups_async([flat], [partitions], [group], 1, 0)
        assert len(deferred_allgather._pending) == 1
        deferred_allgather.wait()
        assert not deferred_allgather._pending
        results[rank] = flat.tolist()
    finally:
        dist.destroy_process_group()


def test_async_all_gather_fills_every_shard():
    world = 2
    results = mp.Manager().dict()
    mp.spawn(_worker, args=(world, 29613, results), nprocs=world)
    expected = [float(i + 10 * r) for r in range(world) for i in range(4)]
    assert results[0] == expected and results[1] == expected


def test_install_swaps_and_restores_zero_all_gather(monkeypatch):
    original = object()
    stage = types.ModuleType("deepspeed.runtime.zero.stage_1_and_2")
    stage.all_gather_dp_groups = original
    zero = types.ModuleType("deepspeed.runtime.zero")
    zero.stage_1_and_2 = stage
    for name, module in (
        ("deepspeed", types.ModuleType("deepspeed")),
        ("deepspeed.runtime", types.ModuleType("deepspeed.runtime")),
        ("deepspeed.runtime.zero", zero),
        ("deepspeed.runtime.zero.stage_1_and_2", stage),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    deferred_allgather.install()
    deferred_allgather.install()  # idempotent
    assert stage.all_gather_dp_groups is deferred_allgather._all_gather_dp_groups_async
    deferred_allgather.uninstall()
    assert stage.all_gather_dp_groups is original


def test_wait_without_pending_is_a_no_op():
    deferred_allgather.wait()
    assert not deferred_allgather._pending


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
