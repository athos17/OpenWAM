"""training.gradient_checkpointing_layers -> per-layer full recompute in the MoT driver."""

import types

import pytest
import torch

from openwam.model.architectures.dual_system.mot_driver import (
    DualSystemMoTDriver,
    resolve_checkpoint_layers,
    run_video_blocks,
)


@pytest.mark.parametrize(
    "spec, expected",
    [
        (None, None),
        ("all", None),
        (0, frozenset()),
        (1, frozenset(range(6))),
        (2, frozenset({0, 2, 4})),
        (4, frozenset({0, 4})),
        ([5, 1], frozenset({1, 5})),
    ],
)
def test_resolve_checkpoint_layers(spec, expected):
    assert resolve_checkpoint_layers(spec, 6) == expected


@pytest.mark.parametrize("spec", [-1, [6], [-1], True])
def test_resolve_checkpoint_layers_rejects(spec):
    with pytest.raises(ValueError):
        resolve_checkpoint_layers(spec, 6)


class _Train:
    training = True


def test_step_routes_only_selected_layers_to_full_recompute():
    driver = DualSystemMoTDriver.__new__(DualSystemMoTDriver)
    driver.ab = _Train()
    driver.checkpoint_layers = frozenset({0, 2})
    calls = []
    driver._step_checkpointed = lambda layer_id, v, a, **kw: calls.append(("full", layer_id)) or (v, a)
    driver._step_impl = lambda layer_id, v, a, **kw: calls.append(("plain", layer_id)) or (v, a)
    for layer_id in range(4):
        driver.step(layer_id, None, None, use_gradient_checkpointing=True)
    assert calls == [("full", 0), ("plain", 1), ("full", 2), ("plain", 3)]
    calls.clear()
    driver.checkpoint_layers = None
    driver.step(1, None, None, use_gradient_checkpointing=True)
    driver.step(1, None, None, use_gradient_checkpointing=False)
    assert calls == [("full", 1), ("plain", 1)]


class _VideoBlocks:
    """Video backbone stand-in: run_block checkpoints when the state says so."""

    num_layers = 4

    def __init__(self):
        torch.manual_seed(0)
        self.weights = [torch.nn.Parameter(torch.randn(3, 3)) for _ in range(self.num_layers)]
        self.calls = []

    def run_block(self, block_id, state):
        self.calls.append((block_id, state.use_gradient_checkpointing, state.use_gradient_checkpointing_offload))
        fn = lambda x: torch.tanh(x @ self.weights[block_id])  # noqa: E731
        if state.use_gradient_checkpointing:
            state.hidden_states = torch.utils.checkpoint.checkpoint(fn, state.hidden_states, use_reentrant=False)
        else:
            state.hidden_states = fn(state.hidden_states)
        return state


def _video_state(gc=True, offload=False):
    from openwam.model.video_backbone.base import BlockLoopState

    x = torch.ones(2, 3, requires_grad=True)
    return BlockLoopState(hidden_states=x, time_mod=None, rope_freqs=None, context=None,
                          use_gradient_checkpointing=gc, use_gradient_checkpointing_offload=offload)


def test_video_only_blocks_checkpoint_only_selected_layers():
    vb = _VideoBlocks()
    state = run_video_blocks(vb, _video_state(offload=True), frozenset({0, 2}))
    assert vb.calls == [(0, True, True), (1, False, False), (2, True, True), (3, False, False)]
    assert state.use_gradient_checkpointing and state.use_gradient_checkpointing_offload  # restored
    vb.calls.clear()
    run_video_blocks(vb, _video_state(gc=False), frozenset({0}))
    assert [c[1] for c in vb.calls] == [False] * 4  # never turns recompute on


def test_video_only_layer_selection_keeps_gradients():
    grads = []
    for layers in (None, frozenset({1}), frozenset()):
        vb = _VideoBlocks()
        run_video_blocks(vb, _video_state(), layers).hidden_states.sum().backward()
        grads.append([w.grad for w in vb.weights])
    for other in grads[1:]:
        for a, b in zip(grads[0], other):
            assert torch.equal(a, b)


def test_video_only_spec_falls_back_to_shared_layers():
    from openwam.model.architectures.base import BaseWAMArchitecture

    arch = types.SimpleNamespace()  # the methods only touch plain attributes
    spec = lambda: BaseWAMArchitecture.video_only_checkpoint_spec(arch)  # noqa: E731
    assert spec() is None
    BaseWAMArchitecture.set_training_runtime(arch, gradient_checkpointing_layers=[0, 1])
    assert spec() == [0, 1]
    BaseWAMArchitecture.set_training_runtime(
        arch, gradient_checkpointing_layers=[0, 1], gradient_checkpointing_layers_video_only=2
    )
    assert spec() == 2
