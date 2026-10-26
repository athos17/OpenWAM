from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.deploy.obs_preprocess import ObsValidationError
from openwam.deploy.server import MAX_ACTION_CHUNK, PolicyServer


class _Preprocessor:
    def __init__(self):
        self.calls = []

    def preprocess(self, obs):
        processed = {"processed": obs}
        self.calls.append(processed)
        return processed


class _Policy:
    def __init__(self):
        self.observations = []

    def predict_action(self, obs):
        self.observations.append(obs)
        return np.full(3, len(self.observations), dtype=np.float32)


def _server():
    server = PolicyServer(
        engine=SimpleNamespace(architecture=None),
        cfg=OmegaConf.create({}),
    )
    server._policy = _Policy()
    server._obs_preprocessor = _Preprocessor()
    server._execution_config = SimpleNamespace(mode="sync", inference_horizon=32)
    return server


def test_predict_chunk_preprocesses_once_and_returns_consecutive_actions():
    server = _server()

    result = server.predict_chunk({"images": {}}, 3)

    assert result["count"] == 3
    assert result["step"] == 3
    assert result["actions"] == [[1.0] * 3, [2.0] * 3, [3.0] * 3]
    assert len(server._obs_preprocessor.calls) == 1
    assert server._policy.observations == [server._obs_preprocessor.calls[0]] * 3


@pytest.mark.parametrize("count", [True, 0, -1, MAX_ACTION_CHUNK + 1, 1.5])
def test_predict_chunk_rejects_invalid_count(count):
    server = _server()

    with pytest.raises(ObsValidationError, match="count"):
        server.predict_chunk({"images": {}}, count)


def test_protocol_capabilities_advertise_chunk_and_executor():
    server = _server()

    capabilities = server._protocol_capabilities()

    assert "obs_chunk" in capabilities["supported_requests"]
    assert capabilities["max_action_chunk"] == 32
    assert capabilities["inference_mode"] == "sync"
    assert capabilities["inference_horizon"] == 32


def test_predict_chunk_requires_sync_executor():
    server = _server()
    server._execution_config = SimpleNamespace(mode="async", inference_horizon=32)

    with pytest.raises(ObsValidationError, match="inference_mode='sync'"):
        server.predict_chunk({"images": {}}, 1)


def test_predict_chunk_does_not_cross_inference_horizon():
    server = _server()
    server._execution_config = SimpleNamespace(mode="sync", inference_horizon=2)

    with pytest.raises(ObsValidationError, match=r"\[1, 2\]"):
        server.predict_chunk({"images": {}}, 3)
