"""Overlap DeepSpeed ZeRO-1/2's post-step parameter all-gather (``training.overlap_param_allgather``).

After ``optimizer.step`` ZeRO-1/2 all-gathers the updated bf16 parameter shards with a
blocking collective, so the default stream idles for the whole all-gather (~0.3 s/step
on 2x8 A800, all of it exposed). :func:`install` makes DeepSpeed launch it with
``async_op=True`` instead; the trainer calls :func:`wait` right before the parameters
are read again (the next DiT forward, a checkpoint), so the next batch's frozen VAE / T5
encode runs while the shards are still in flight.

The all-gather runs on DeepSpeed's own data-parallel communicator (a clone of the world
group), so the trainer's metric / modality collectives on the default group neither
queue behind it nor wait for it. Parameter values are the same; only the time the
default stream waits for them moves.
"""

from __future__ import annotations

import logging

import torch.distributed as dist

logger = logging.getLogger(__name__)

_pending: list = []
_original = None


def _all_gather_dp_groups_async(
    groups_flat, partitioned_param_groups, dp_process_group, start_alignment_factor, allgather_bucket_size
):
    for group_flat, partitioned_params, group in zip(groups_flat, partitioned_param_groups, dp_process_group):
        if dist.get_world_size(group=group) == 1:
            continue
        rank = dist.get_rank(group=group)
        _pending.append(dist.all_gather_into_tensor(group_flat, partitioned_params[rank], group=group, async_op=True))


def install() -> None:
    """Swap ZeRO-1/2's post-step all-gather for the async one (idempotent)."""
    global _original
    from deepspeed.runtime.zero import stage_1_and_2

    if _original is None:
        _original = stage_1_and_2.all_gather_dp_groups
        stage_1_and_2.all_gather_dp_groups = _all_gather_dp_groups_async
        logger.info("ZeRO-1/2 parameter all-gather: async, waited before the next forward")


def uninstall() -> None:
    global _original
    wait()
    if _original is not None:
        from deepspeed.runtime.zero import stage_1_and_2

        stage_1_and_2.all_gather_dp_groups = _original
        _original = None


def wait() -> None:
    """Make the current stream wait for every in-flight parameter all-gather."""
    while _pending:
        _pending.pop(0).wait()


__all__ = ["install", "uninstall", "wait"]
