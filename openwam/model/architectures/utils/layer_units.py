"""Optional per-layer modules that make the block loop visible to module-hook sharding.

The block loops read each layer's parameters directly (``pre_attn_at_layer`` /
``post_attn_at_layer``, the fused Wan regions, ``run_block``) instead of calling the
block modules, so a backend that gathers parameters in module forward hooks (FSDP2)
never sees those layers run. ``run_layer`` sends one layer's work through the
``LayerUnit`` the backend installed on the video backbone; the unit owns that layer's
blocks, so its hooks gather them around the call and register gradient reduction on the
tensors passed in. Without installed units ``run_layer`` is a plain call.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class LayerUnit(nn.Module):
    """Holds one layer's blocks (shared with their backbones) as a single hookable call.

    ``fn`` must take the hidden-state tensors that carry gradient into the layer as its
    positional arguments and use those objects (the backend may substitute them), and it
    must return the layer's output tensors.
    """

    def __init__(self, **blocks: nn.Module):
        super().__init__()
        for name, block in blocks.items():
            self.add_module(name, block)

    def forward(self, fn: Callable, *tensors: torch.Tensor):
        return fn(*tensors)


def run_layer(vb, layer_id: int, fn: Callable, *tensors: torch.Tensor):
    units = getattr(vb, "_layer_units", None)
    if units is None:
        return fn(*tensors)
    return units[layer_id](fn, *tensors)
