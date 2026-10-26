"""training.gradient_checkpointing_layers -> per-layer full recompute in the MoT driver."""

import pytest

from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver, resolve_checkpoint_layers


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
