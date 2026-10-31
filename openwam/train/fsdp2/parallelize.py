"""FSDP2 / HSDP sharding of a TouchWAM-style architecture (torch.distributed.fsdp.fully_shard).

Layout, following TorchTitan's DiT recipe (``models/flux/model.py`` ``_apply_fsdp``):
one FSDP unit per block-loop layer (a :class:`LayerUnit` holding that layer's video and
action blocks, since the MoT loop reads their parameters without calling them), one unit
per tactile bridge block, and a root unit (:class:`TrainRoot`) for every remaining
trainable parameter, gathered for the whole ``compute_loss`` call.

Frozen parameters (T5 / VAE / frozen encoders) stay outside FSDP: one full copy per GPU,
as under DeepSpeed. Trainable scalar parameters (FSDP2 cannot shard 0-dim tensors) stay
replicated too; :class:`openwam.train.fsdp2.accelerator.FSDP2Accelerator` averages their
gradients. Sharded parameters are fp32 (the optimizer's master copy); forward/backward
compute uses ``param_dtype`` copies gathered per unit.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard, register_fsdp_forward_method

from openwam.model.architectures.utils.layer_units import LayerUnit

logger = logging.getLogger(__name__)


class TrainRoot(nn.Module):
    """FSDP root: owns the architecture's remaining trainable parameters and the layer units.

    ``compute_loss`` is registered as an FSDP forward method, so the root parameters are
    gathered before the architecture's own ``compute_loss`` (which calls the model's
    forward plus parameterized pre/post steps) and gradients reduce after its backward.
    The architecture's ``state_dict`` is unchanged: the units sit under this wrapper.
    """

    def __init__(self, arch: nn.Module, layer_units: list[LayerUnit]):
        super().__init__()
        self.arch = arch
        self.layer_units = nn.ModuleList(layer_units)

    def compute_loss(self, **kwargs):
        return self.arch.compute_loss(**kwargs)


def _upcast_trainable(module: nn.Module, dtype: torch.dtype, skip: set) -> None:
    for param in module.parameters():
        if param.requires_grad and param not in skip and param.dtype != dtype:
            param.data = param.data.to(dtype)


def parallelize_touchwam(
    arch: nn.Module,
    mesh,
    *,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    master_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
    forward_prefetch: bool = False,
) -> tuple[TrainRoot, list[nn.Parameter]]:
    """Shard ``arch`` in place; returns the FSDP root and the replicated trainable params.

    ``arch`` must already sit on its training device with its usual dtypes
    (``set_dtype_device``): ``.to()`` must not be called on it afterwards.
    """
    vb = arch.video_backbone
    ab = getattr(arch, "action_backbone", None)
    video_blocks = vb.dit.blocks
    action_blocks = getattr(ab, "blocks", None)
    if action_blocks is not None and len(action_blocks) != len(video_blocks):
        raise ValueError("FSDP2 layer units need one action block per video block")
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False)

    frozen = {p for p in arch.parameters() if not p.requires_grad}
    replicated = [p for p in arch.parameters() if p.requires_grad and p.dim() == 0]
    keep_out = frozen | set(replicated)
    for param in replicated:
        param.data = param.data.to(master_dtype)

    units = []
    for i, video_block in enumerate(video_blocks):
        blocks = {"video": video_block}
        if action_blocks is not None:
            blocks["action"] = action_blocks[i]
        units.append(LayerUnit(**blocks))
    bridges = list(getattr(getattr(arch, "touch_v2", None), "blocks", None) or [])

    # Units in execution order; the last layer is resharded only in backward (TorchTitan:
    # its backward starts right after the forward ends).
    for i, unit in enumerate(units):
        _upcast_trainable(unit, master_dtype, keep_out)
        fully_shard(unit, mesh=mesh, mp_policy=mp_policy, ignored_params=keep_out,
                    reshard_after_forward=reshard_after_forward and i < len(units) - 1)
    for bridge in bridges:
        _upcast_trainable(bridge, master_dtype, keep_out)
        fully_shard(bridge, mesh=mesh, mp_policy=mp_policy, ignored_params=keep_out,
                    reshard_after_forward=reshard_after_forward)
    if forward_prefetch:
        # Each layer starts the next layer's all-gather when it begins (TorchTitan
        # distributed/fsdp.py), so the gather overlaps this layer's compute even when the
        # host does not run ahead of the GPU. Every modality runs every layer; the bridges
        # run only on tactile batches and are left to gather on demand.
        for unit, next_unit in zip(units, units[1:]):
            unit.set_modules_to_forward_prefetch([next_unit])
    vb._layer_units = units

    root = TrainRoot(arch, units)
    in_units = {p for module in (*units, *bridges) for p in module.parameters()}
    _upcast_trainable(arch, master_dtype, keep_out | in_units)
    fully_shard(root, mesh=mesh, mp_policy=mp_policy, ignored_params=keep_out | in_units,
                reshard_after_forward=False)
    register_fsdp_forward_method(root, "compute_loss")

    def count(params):
        return sum(p.numel() for p in params)

    unit_params = {p for unit in units for p in unit.parameters()} - keep_out
    bridge_params = {p for bridge in bridges for p in bridge.parameters()} - keep_out
    root_params = {p for p in arch.parameters() if p.requires_grad} - unit_params - bridge_params - set(replicated)
    logger.info(
        "FSDP2: mesh=%s | %d layer units %.1fM, %d bridge units %.1fM, root %.1fM, replicated scalars %d, "
        "frozen (per-GPU copy) %.1fM | param_dtype=%s reduce_dtype=%s reshard_after_forward=%s forward_prefetch=%s",
        mesh, len(units), count(unit_params) / 1e6, len(bridges), count(bridge_params) / 1e6,
        count(root_params) / 1e6, len(replicated), count(frozen) / 1e6, param_dtype, reduce_dtype,
        reshard_after_forward, forward_prefetch,
    )
    return root, replicated
