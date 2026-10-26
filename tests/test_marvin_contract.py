import numpy as np
import pytest

from openwam.dataloader.marvin import (
    RAW_ACTION_DIM,
    ROT6D_RAW_DIMS,
    marvin_raw_vector,
    rot6d_rows_to_columns,
)
from openwam.dataloader.utils.unify_action import map_to_unify, parse_unify_spec
from scripts.convert_marvin_lerobot_v3 import _source_rot6d_provenance


def test_rows_to_columns_identity_and_z_rotation():
    identity_rows = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    np.testing.assert_allclose(rot6d_rows_to_columns(identity_rows), identity_rows)

    # Matrix rows for +90 degrees around z. OpenWAM expects columns.
    rows = np.array([0, -1, 0, 1, 0, 0], dtype=np.float32)
    expected_columns = np.array([0, 1, 0, -1, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(rot6d_rows_to_columns(rows), expected_columns, atol=1e-6)


def test_marvin_raw_vector_and_unified_mask_contract():
    source = np.arange(76, dtype=np.float32)
    raw = marvin_raw_vector(source)
    assert raw.shape == (RAW_ACTION_DIM,)
    np.testing.assert_array_equal(raw[:9], source[58:67])
    np.testing.assert_array_equal(raw[9:31], source[14:36])
    np.testing.assert_array_equal(raw[31:40], source[67:76])
    np.testing.assert_array_equal(raw[40:62], source[36:58])

    mapping = parse_unify_spec(["0-8", "10-31", "34-42", "44-65"])
    unified, mask = map_to_unify(raw, mapping)
    assert unified.shape == (80,)
    assert mask.sum() == 62
    assert not mask[9]
    assert not mask[32:34].any()
    assert not mask[43]
    assert not mask[66:80].any()
    assert set(ROT6D_RAW_DIMS) == set(range(3, 9)) | set(range(34, 40))


def test_rot6d_provenance_supports_each_existing_metadata_form(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "eef_action_backfill.json").write_text(
        '{"rotation_6d_layout": "[R00,R01,R02,R10,R11,R12]"}'
    )
    assert _source_rot6d_provenance(tmp_path) == [
        (meta / "eef_action_backfill.json", "[R00,R01,R02,R10,R11,R12]")
    ]


def test_converter_refuses_an_existing_output(tmp_path):
    from scripts.convert_marvin_lerobot_v3 import convert

    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert(tmp_path / "source", output)
