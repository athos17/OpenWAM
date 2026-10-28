"""Validation tools must audit actual bytes and keep their working set bounded."""

import hashlib

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.validation.touchwam_alpha_init import assign_and_hash, read_header
from scripts.validation.touchwam_multisource_gpu import make_inputs, tiny_model


def test_streamed_assignment_preserves_pretrained_bits_and_releases_parameter(tmp_path):
    value = torch.randn(64, 128, dtype=torch.bfloat16)
    path = tmp_path / 'weights.safetensors'
    save_file({'weight': value}, str(path))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    model = torch.nn.Linear(128, 64, bias=False, device='meta')
    header, raw_header = read_header(path)
    begin, end = header['weight']['data_offsets']
    with safe_open(str(path), framework='pt', device='cpu') as handle, open(path, 'rb') as raw:
        digest = assign_and_hash(model, 'weight', handle.get_tensor('weight'), raw,
                                 len(raw_header) + begin, end - begin, chunk_bytes=4096)
    assert model.weight.is_meta
    assert digest == hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_validation_builder_uses_real_wan_and_declared_missing_modalities():
    torch.set_num_threads(1)
    model = tiny_model(torch.device('cpu'), torch.float32)
    assert model.video_backbone.dit.__class__.__name__ == 'WanModel'
    assert model.action_dim == 80
    sample = make_inputs('video_tactile', torch.device('cpu'), torch.float32)
    assert 'actions' not in sample and 'proprio' not in sample
    assert sample['future_tactile_maps'].shape == (1, 8, 2, 1, 32, 32)
    losses = model.compute_loss(**sample)
    losses['loss'].backward()
    assert losses['loss_action'] == 0 and torch.isfinite(losses['loss'])
