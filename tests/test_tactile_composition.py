import json

import numpy as np
import pytest

from openwam.tactile.composition import SensorTile, compose_mosaic
from openwam.tactile.layout import LAYOUT_VERSION, SurfaceLayout, SurfaceSpec
from openwam.tactile.registry import REGISTRY_VERSION


def test_composition_preserves_values_rotation_gaps_and_missingness(tmp_path):
    # Two pads on ONE jaw A, separated by an unmeasured gap; B stays separate.
    a = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    b = np.array([[-2, -1], [6, 7], [8, 9]], dtype=np.float32)
    tiles = (SensorTile("a", (2, 3), (0, 0)), SensorTile("b", (3, 2), (0, 4), 1))
    value, valid, roi = compose_mosaic((2, 7), tiles, {"a": a, "b": b})
    assert valid and roi.sum() == 12 and not roi[:, 3].any()
    np.testing.assert_array_equal(value[:, :3], a)
    np.testing.assert_array_equal(value[:, 4:], np.rot90(b))
    assert value[0, 0] == 0 and roi[0, 0]  # A real measured zero remains valid.
    for packets, observed in [
        ({"a": a}, None),
        ({"a": a, "b": b}, {"a": True, "b": False}),
        ({"a": a, "b": np.full_like(b, np.nan)}, None),
    ]:
        value, valid, same_roi = compose_mosaic((2, 7), tiles, packets, observed=observed)
        assert not valid and not value.any()
        np.testing.assert_array_equal(roi, same_roi)


def test_layout_load_composes_to_unique_right_region(tmp_path):
    doc = dict(
        schema=LAYOUT_VERSION,
        registry=REGISTRY_VERSION,
        name="single_arm_two_pads_per_a",
        surfaces=[
            dict(
                native_key="composed/a",
                region="right.gripper.jaw_a.surface",
                sensor="fixture",
                representation="normal_force_fixture",
                units="N",
                native_shape=[2, 4],
                tiles=[
                    dict(native_key="pad1", native_shape=[2, 2], origin=[0, 0]),
                    dict(native_key="pad2", native_shape=[2, 2], origin=[0, 2]),
                ],
            )
        ],
    )
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(doc))
    layout = SurfaceLayout.load(path)
    result = layout.compose({"pad1": np.zeros((2, 2), np.float32), "pad2": np.ones((2, 2), np.float32)})
    assert list(result["maps"]) == ["right.gripper.jaw_a.surface"]
    assert layout.identity_fields()["tactile_region_names"] == ["right.gripper.jaw_a.surface"]
    assert all(result["valid"].values())
    assert result["maps"]["right.gripper.jaw_a.surface"].shape == (2, 4)


def test_bad_composition_never_sums_overlapping_or_reused_sensors():
    a = SensorTile("a", (2, 2), (0, 0))
    with pytest.raises(ValueError, match="overlap"):
        compose_mosaic((2, 2), (a, SensorTile("b", (2, 2), (0, 0))), {})
    with pytest.raises(ValueError, match="exceed"):
        compose_mosaic((1, 2), (a,), {})
    with pytest.raises(ValueError, match="dtype"):
        compose_mosaic((2, 2), (a,), {"a": np.ones((2, 2), np.uint8)})
    with pytest.raises(ValueError, match="shape"):
        compose_mosaic((2, 2), (a,), {"a": np.ones((3, 2), np.float32)})
    fields = dict(sensor="test", representation="deform", units="uint8", native_shape=(2, 2), tiles=(a,))
    with pytest.raises(ValueError, match="duplicated"):
        SurfaceLayout(
            "bad",
            (
                SurfaceSpec("out/a", "right.gripper.jaw_a.surface", **fields),
                SurfaceSpec("out/b", "right.gripper.jaw_b.surface", **fields),
            ),
        )


@pytest.mark.parametrize("bad", [-1, 4, 1.5, True])
def test_rotation_is_explicit_and_discrete(bad):
    with pytest.raises(ValueError, match="quarter_turns"):
        SensorTile("a", (2, 2), (0, 0), bad)
