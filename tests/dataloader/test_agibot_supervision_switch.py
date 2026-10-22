"""The public AgiBot supervision switch must reach every bucket and modality."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from openwam.dataloader.agibotworld import AgiBotWorldDataset
from openwam.dataloader.registry import build_dataset
from tests.dataloader.test_agibotworld import _make_bucket, patch_decode as patch_decode
from tests.model.test_video_only_routing import Probe, inputs


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("dex", [False, True])
@pytest.mark.parametrize("construction", ["direct", "single_config", "root_config"])
def test_disabled_preserves_values_and_disables_both_masks(tmp_path, patch_decode, unified, dex, construction):
    root = tmp_path / "data"
    bucket = _make_bucket(
        root / ("475" if dex else "327"), [{"length": 40, "flag": 0, "delta": 0}],
        action_velocity=[1.25, 0, -0.5], state_velocity=[0.25, 0, 0.4],
        action_gripper=[0, 1], state_gripper=[0.035, 0.125],
    )
    if dex:
        path = bucket / "data/chunk-000/file-000.parquet"
        table = pq.read_table(path)
        fingers = np.tile(np.arange(12, dtype=np.float32), (40, 1))
        for name in ["action.dex", "observation.state.dex"]:
            table = table.append_column(name, pa.array(fingers.tolist(), type=pa.list_(pa.float32(), 12)))
        pq.write_table(table, path)

    def build(flag):
        kwargs = dict(num_frames=33, video_stride=4, height=384, width=320,
                      multiview=True, unify_action=unified, normalize_mode=None)
        if flag is not None:
            kwargs["enable_action_supervision"] = flag
        if construction == "direct":
            return AgiBotWorldDataset(str(bucket), **kwargs)
        return build_dataset({"type": "agibotworld", "dataset_dir": str(root if construction == "root_config" else bucket),
                              **kwargs}, split="train")

    default, enabled, disabled = build(None), build(True), build(False)
    assert len(default) == len(enabled) == len(disabled)
    for idx in [0, len(disabled) - 1]:
        original, on, off = default[idx], enabled[idx], disabled[idx]
        assert off["training_modality"] == "video_only"
        assert on.get("training_modality", "video_action") == "video_action"
        for key in ["action", "proprio", "video_mask"]:
            torch.testing.assert_close(off[key], on[key], rtol=0, atol=0)
            torch.testing.assert_close(on[key], original[key], rtol=0, atol=0)
        for key in ["action_mask", "proprio_mask"]:
            assert on[key].any()
            assert not off[key].any()
            torch.testing.assert_close(on[key], original[key], rtol=0, atol=0)
        assert off["prompt"] == on["prompt"]
        for before, after in zip(on["video"], off["video"]):
            np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_disabled_sample_uses_video_only_loss_path(tmp_path, patch_decode):
    bucket = _make_bucket(tmp_path / "327", [{"length": 40, "flag": 0, "delta": 0}])
    ds = build_dataset({"type": "agibotworld", "dataset_dir": str(bucket), "unify_action": True,
                        "enable_action_supervision": False, "normalize_mode": None})
    sample = ds[0]
    data = inputs([sample.get("training_modality") == "video_only"])
    data["actions"] = sample["action"].unsqueeze(0)
    data["proprio"] = sample["proprio"]
    data["action_is_pad"] = ~sample["action_mask"].unsqueeze(0)
    data["proprio_mask"] = sample["proprio_mask"].unsqueeze(0)
    model = Probe()
    loss = model.compute_loss(**data)
    loss["loss"].backward()
    assert loss["loss_action"].item() == 0
    assert model.calls == [(1, True)]
    assert model.action.grad is None and model.proprio.grad is None
    assert model.video.grad is not None
