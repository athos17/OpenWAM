"""Optical color and missing-data semantics at the shared region boundary."""

import copy

import pytest
import torch

from openwam.tactile.regions import CODEC, INTENSITY_CODEC, RGB_CODEC, collate_regions


def optical_sample():
    maps = torch.zeros(2, 1, 3, 8, 8, dtype=torch.uint8)
    maps[1, 0, 0] = 255
    maps[1, 0, 1] = 128
    return dict(
        tactile_codec=RGB_CODEC,
        tactile_region_names=["right.gripper.jaw_a.surface"],
        tactile_sensor_names=["optical_sensor"],
        tactile_layout_present=torch.tensor([True]),
        tactile_maps=maps,
        tactile_valid=torch.tensor([[True], [True]]),
    )


def collate(samples, codec=RGB_CODEC):
    return collate_regions(samples, map_size=8, history=2, future_frames=3,
                           sensor_names=["optical_sensor"], codec=codec)


def test_rgb_keeps_color_and_observed_black_separate_from_missing_future():
    batch = collate([optical_sample(), {}])
    assert batch["tactile_maps"].shape == (2, 2, 1, 3, 8, 8)
    assert batch["tactile_spatial_valid"].shape == (2, 1, 1, 8, 8)
    torch.testing.assert_close(batch["tactile_maps"][0, 1, 0, :, 0, 0],
                               torch.tensor([1.0, 128 / 255, 0.0]))
    assert batch["tactile_valid"][0, 0, 0]  # A real black frame is observed.
    assert not batch["tactile_valid"][1].any()  # An absent layout is not.
    assert batch["tactile_query_mask"][0].all()
    assert not batch["future_tactile_valid"].any()  # No fabricated future labels.


def test_rgb_is_not_accepted_as_normalized_intensity_or_mixed_with_it():
    rgb = optical_sample()
    with pytest.raises(ValueError, match="tactile_codec"):
        collate([rgb], codec=INTENSITY_CODEC)
    intensity = copy.deepcopy(rgb)
    intensity["tactile_codec"] = INTENSITY_CODEC
    intensity["tactile_maps"] = intensity["tactile_maps"][:, :, :1]
    with pytest.raises(ValueError, match="tactile_codec"):
        collate([rgb, intensity])
    # Existing SDK grayscale compatibility remains intentional and one-channel.
    intensity["tactile_codec"] = CODEC
    assert collate([intensity], codec=INTENSITY_CODEC)["tactile_maps"].shape[3] == 1


def test_rgb_requires_three_color_channels_and_explicit_sensor_identity():
    sample = optical_sample()
    sample["tactile_maps"] = sample["tactile_maps"][:, :, :1]
    with pytest.raises(ValueError, match="tactile_maps: expected"):
        collate([sample])
    sample = optical_sample()
    sample.pop("tactile_sensor_names")
    with pytest.raises(ValueError, match="sensor/source identity"):
        collate([sample])


def test_rgb_roi_masks_pixels_without_changing_channel_count():
    sample = optical_sample()
    sample["tactile_spatial_valid"] = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    sample["tactile_spatial_valid"][..., :2, :] = False
    sample["future_tactile_maps"] = torch.zeros(3, 1, 3, 8, 8)
    sample["future_tactile_valid"] = torch.tensor([[True], [False], [True]])
    sample["tactile_query_mask"] = torch.tensor([[True], [True], [False]])
    result = collate([sample])
    assert not result["tactile_spatial_valid"][..., :2, :].any()
    assert result["future_tactile_maps"].shape[3] == 3
    assert not torch.equal(result["future_tactile_valid"], result["tactile_query_mask"])
