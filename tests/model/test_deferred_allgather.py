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


def _modality_worker(rank, port):
    """Compare gathered parameters across graph switches and inactive Adam groups."""
    from openwam.train.utils.modality_optimizer import ModalityAwareAdamW
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        group = dist.new_group(ranks=[0, 1])
        streams = ("video", "action", "tactile", "tactile_intensity", "tactile_rgb")
        flats = [torch.ones(8) for _ in streams]
        partitions = [[flat[:4], flat[4:]] for flat in flats]
        def make_opt():
            return ModalityAwareAdamW([
                {"params": [torch.nn.Parameter(torch.ones(4))], "touchwam_stream": stream,
                 "touchwam_active_in_window": False} for stream in streams
            ], lr=0.1, weight_decay=0.1)
        reference, fast = make_opt(), make_opt()
        sequence = [("video_action_tactile", "tactile_intensity_v1"),
                    ("video_tactile", "tactile_rgb_v1"), ("video_only", None),
                    ("video_action", None), ("video_action_tactile", "tactile_rgb_v1")]
        for modality, codec in sequence:
            for opt in (reference, fast):
                opt.activate(modality, codec)
                for g in opt.param_groups:
                    g["params"][0].grad = torch.full((4,), float(rank + 1))
                opt.step()
            expected = []
            for ref, actual, pieces in zip(reference.param_groups, fast.param_groups, partitions):
                out = torch.empty(8)
                dist.all_gather_into_tensor(out, ref["params"][0].detach(), group=group)
                expected.append(out)
                pieces[rank].copy_(actual["params"][0].detach())
            deferred_allgather._all_gather_dp_groups_async(flats, partitions, [group] * len(streams), 1, 0)
            assert len(deferred_allgather._pending) == len(streams)
            torch.ones(3, 3) @ torch.ones(3, 3)  # independent frozen preprocessing
            deferred_allgather.wait()
            for actual, ref in zip(flats, expected):
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            for a, b in zip(reference.param_groups, fast.param_groups):
                for key, value in reference.state[a["params"][0]].items():
                    torch.testing.assert_close(value, fast.state[b["params"][0]][key], rtol=0, atol=0)
    finally:
        deferred_allgather.wait()
        dist.destroy_process_group()


def test_modality_switches_preserve_grouped_async_allgather_and_adam_state():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_modality_worker, args=(port,), nprocs=2)
