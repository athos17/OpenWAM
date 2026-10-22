"""Deploy must use state statistics for inputs and action statistics for outputs."""
import numpy as np
import pytest

from openwam.dataloader.transforms.normalize import Normalizer, load_mode_stats
from openwam.dataloader.utils.normalization import apply_normalization


def stats(lo, hi):
    lo, hi = np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)
    return dict(min=lo, max=hi, q01=lo, q99=hi, mean=(lo + hi) / 2, std=(hi - lo) / 2)


@pytest.mark.parametrize("train_mode,deploy_mode", [("quantile", "q99"), ("min-max", "min_max"), ("z-score", "mean_std")])
def test_state_input_and_action_output_use_separate_statistics(tmp_path, train_mode, deploy_mode):
    action = stats([-2, 0], [2, 1])
    state = stats([-10, 35], [10, 125])
    path = tmp_path / "normalization_stats.npy"
    np.save(path, {"eef": action, "eef_state": state})
    normalizer = Normalizer(mode=deploy_mode, stats=load_mode_stats(str(path), "eef"))
    raw_state = np.array([[2, 80], [8, 100]], dtype=np.float32)
    np.testing.assert_allclose(normalizer.normalize(raw_state), apply_normalization(raw_state, state, train_mode), atol=1e-6)
    raw_action = np.array([[1, .25], [-1, .75]], dtype=np.float32)
    np.testing.assert_allclose(normalizer.unnormalize(apply_normalization(raw_action, action, train_mode)), raw_action, atol=1e-6)


def test_invalid_state_block_rejected(tmp_path):
    path = tmp_path / "stats.npy"
    np.save(path, {"eef": stats([-1], [1]), "eef_state": [1, 2]})
    with pytest.raises(TypeError, match="must be a stats dict"):
        load_mode_stats(str(path), "eef")
