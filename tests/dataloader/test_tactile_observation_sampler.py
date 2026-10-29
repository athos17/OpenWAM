"""Global batches cannot stack image histories with native pressure matrices."""
import math
from itertools import zip_longest

import numpy as np
import pytest
import torch
from accelerate.data_loader import prepare_data_loader

from openwam.dataloader.mixture import MixtureDataset
from openwam.dataloader.modality_sampler import GlobalModalitySampler, batch_tactile_observation_shape
from openwam.dataloader.touchwam_pretrain import HomogeneousPublishedBuckets


class Source(torch.utils.data.Dataset):
    action_dim = 80
    def __init__(self, count, shape=None, mode="video_action_tactile", codec="tactile_intensity_v1", encoder="sharpa"):
        self.count, self.tactile_observation_shape = count, shape
        self.training_modality, self.tactile_codec, self.encoder = mode, codec, encoder
    def __len__(self):
        return self.count
    def __getitem__(self, index):
        sample = dict(training_modality=self.training_modality, tactile_codec=self.tactile_codec,
                      tactile_region_names=["right.thumb.tip"], value=index)
        if self.tactile_observation_shape is not None:
            c = 3 if self.tactile_codec == "tactile_rgb_v1" else 1
            sample.update(tactile_observation_maps=torch.zeros(1, 1, c, *self.tactile_observation_shape),
                          tactile_encoder_names=[self.encoder], tactile_observation_shape=self.tactile_observation_shape)
        return sample


@pytest.mark.parametrize("batch,world", [(2, 2), (1, 3), (3, 1)])
def test_same_codec_different_observation_shapes_have_separate_global_batches(batch, world):
    sources = [Source(7, (224, 224)), Source(11, (4, 4), encoder="pressure"),
               Source(13, (224, 224), encoder="other_image"), Source(5, mode="video_only", codec=None)]
    dataset = MixtureDataset(sources, names=["sharpa", "matrix", "other", "ego"])
    sampler = GlobalModalitySampler(dataset, batch, world)
    assert sampler.observation_shape_batching
    order = list(sampler)
    assert set(order) == set(range(len(dataset)))
    assert len(order) == len(sampler)
    assert len(order) - len(dataset) < 3 * batch * world
    # Different image frontends share the same shape bucket, avoiding an
    # unnecessary per-sensor batch and preserving distributed route unions.
    assert sampler._source_codes[0] == sampler._source_codes[2]
    loaders = [prepare_data_loader(
        torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=sampler, collate_fn=list),
        device=torch.device("cpu"), num_processes=world, process_index=rank,
        split_batches=False, put_on_device=False) for rank in range(world)]
    for global_batches in zip_longest(*loaders):
        assert all(b is not None and len(b) == batch for b in global_batches)
        assert len({batch_tactile_observation_shape(b) for b in global_batches}) == 1
    assert dataset.training_data_contract["sources"]["matrix"]["tactile_observation_shape"] == [4, 4]
    assert "tactile_observation_shape" not in dataset.training_data_contract["sources"]["ego"]
    sampler.set_epoch(3)
    current = list(sampler)
    assert current != order
    sampler.set_epoch(3)
    assert list(sampler) == current


def old_sampler_order(dataset, block):
    codes = {"video_action": 0, "video_only": 1, "video_tactile": 2, "video_action_tactile": 3}
    codecs = {None: 0, "tactile_intensity_v1": 1, "sharpa_deform_gray_v1": 2, "tactile_rgb_v1": 3}
    flags = np.asarray([4 * codes[m] + codecs[c] for m, c in zip(dataset.source_modalities, dataset.source_tactile_codecs)])
    active = flags[dataset._index_map[:, 0]]
    positions = [np.flatnonzero(active == flag) for flag in sorted(set(flags.tolist()))]
    schedule = np.concatenate([np.full(math.ceil(len(p) / block), i, dtype=np.uint8) for i, p in enumerate(positions)])
    np.random.RandomState(dataset._seed ^ 0x4D4F44).shuffle(schedule)
    cursors = [0] * len(positions)
    result = []
    for flag in schedule:
        pos, start = positions[int(flag)], cursors[int(flag)]
        chunk = pos[start:start + block]
        cursors[int(flag)] += len(chunk)
        if len(chunk) < block:
            chunk = np.concatenate((chunk, np.resize(pos, block - len(chunk))))
        result.extend(map(int, chunk))
    return result


def test_legacy_sampler_order_and_contract_remain_exact():
    dataset = MixtureDataset([Source(7), Source(11, codec="tactile_rgb_v1"), Source(4, mode="video_only", codec=None)])
    sampler = GlobalModalitySampler(dataset, 2, 3)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        assert list(sampler) == old_sampler_order(dataset, 6)
    assert not sampler.observation_shape_batching
    assert all("tactile_observation_shape" not in s for s in dataset.training_data_contract["sources"].values())


def test_sample_shape_drift_and_unregistered_observations_rejected():
    dataset = MixtureDataset([Source(2, (4, 4))])
    dataset.datasets[0].tactile_observation_shape = (8, 8)
    with pytest.raises(ValueError, match="changed after sampler"):
        dataset[0]
    with pytest.raises(ValueError, match="one tactile observation shape"):
        HomogeneousPublishedBuckets([Source(1, (4, 4)), Source(1, (224, 224))])
    with pytest.raises(ValueError, match="without tactile"):
        MixtureDataset([Source(1, (4, 4), mode="video_only", codec=None)])


def test_runtime_shape_checks_fail_before_model_collectives():
    image, pressure, legacy = Source(1, (224, 224))[0], Source(1, (4, 4))[0], Source(1)[0]
    for samples in ([image, pressure], [image, legacy]):
        with pytest.raises(ValueError, match="mixed tactile observation shapes"):
            batch_tactile_observation_shape(samples)
    with pytest.raises(ValueError, match="differs from its maps"):
        batch_tactile_observation_shape([dict(image, tactile_observation_shape=(8, 8))])
    with pytest.raises(ValueError, match="consistent with their codec"):
        batch_tactile_observation_shape([dict(image, tactile_codec="tactile_rgb_v1")])
    assert batch_tactile_observation_shape([legacy]) is None
    assert batch_tactile_observation_shape([dict(training_modality="video_only",
        tactile_observation_maps=torch.zeros(1, 0, 1, 224, 224))]) is None


def test_trainer_uses_shape_buckets_even_when_modality_and_codec_match():
    from tests.dataloader.test_mixture_trainer_dataloader import _bare_trainer

    dataset = MixtureDataset([Source(7, (224, 224)), Source(9, (4, 4), encoder="pressure")])
    loader = _bare_trainer(dataset).build_dataloader(batch_size=2)
    assert isinstance(loader.sampler, GlobalModalitySampler)
    assert dataset.training_layout["batching"] == "global_modality_codec_observation_v4"


def test_global_validator_rejects_remote_shape_mismatch(monkeypatch):
    from openwam.train.utils import modality_optimizer as module

    sample = Source(1, (224, 224))[0]
    assert module.validate_global_modality_layout([sample], "cpu") == "video_action_tactile"
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    def gather(states, local):
        states[0].copy_(local)
        states[1].copy_(local)
        states[1][-2:] = 4
    monkeypatch.setattr(module.dist, "all_gather", gather)
    with pytest.raises(ValueError, match="observation shape"):
        module.validate_global_modality_layout([sample], "cpu")
