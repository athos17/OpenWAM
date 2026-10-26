"""Stateless pixel/latent preprocessing for the Wan backbone.

Pure functions lifted verbatim from the former ``BasePipeline`` helpers so the
backbone owns its deploy preprocessing without a pipeline object. ``dtype`` /
``device`` (and the division factors for :func:`check_resize_height_width`) are
passed explicitly instead of read off ``self`` — same numerics, no hidden state.
"""

from __future__ import annotations

import numpy as np
import torch
from einops import reduce, repeat
from PIL import Image


def preprocess_image(image, *, dtype, device, pattern="B C H W", min_value=-1, max_value=1):
    """PIL.Image -> tensor in ``[min_value, max_value]`` on ``(dtype, device)``."""
    image = torch.Tensor(np.array(image, dtype=np.float32))
    image = image.to(dtype=dtype, device=device)
    image = image * ((max_value - min_value) / 255) + min_value
    image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
    return image


def preprocess_video(video, *, dtype, device, pattern="B C T H W", min_value=-1, max_value=1):
    """List[PIL.Image] -> stacked tensor along the ``T`` axis of ``pattern``."""
    video = [
        preprocess_image(image, dtype=dtype, device=device, min_value=min_value, max_value=max_value) for image in video
    ]
    video = torch.stack(video, dim=pattern.index("T") // 2)
    return video


def stack_video_uint8(clips) -> torch.Tensor | None:
    """List[List[PIL.Image]] -> uint8 ``[B, T, H, W, C]`` CPU tensor, or None.

    None when the clips are not one uniform RGB uint8 shape (the caller then keeps
    the per-frame path). Pinned when CUDA is available so the upload can be async.
    """
    if not clips or not all(clips):
        return None
    arrays = [[np.asarray(image) for image in clip] for clip in clips]
    first = arrays[0][0]
    if first.dtype != np.uint8 or first.ndim != 3:
        return None
    if any(len(clip) != len(arrays[0]) for clip in arrays) or any(
        a.shape != first.shape or a.dtype != np.uint8 for clip in arrays for a in clip
    ):
        return None
    video = torch.from_numpy(np.stack([np.stack(clip) for clip in arrays]))
    return video.pin_memory() if torch.cuda.is_available() else video


def preprocess_video_batch(clips, *, dtype, device, min_value=-1, max_value=1) -> torch.Tensor:
    """List of clips (List[PIL.Image]) -> ``[B, C, T, H, W]`` in ``[min_value, max_value]``.

    One uint8 upload per batch instead of one float32 upload per frame. Bitwise equal
    to ``torch.cat([preprocess_video(c) for c in clips])``: uint8 -> ``dtype`` is exact
    for 0..255 and the scale/offset run as the same two elementwise ops on ``device``.
    The pinned, non-blocking upload also keeps the host from waiting on queued GPU
    work (e.g. the text encoder), which a pageable copy would do.
    """
    video = stack_video_uint8(clips)
    if video is None:
        return torch.cat([preprocess_video(c, dtype=dtype, device=device, min_value=min_value, max_value=max_value) for c in clips])
    video = video.to(device=device, non_blocking=True).to(dtype=dtype)
    video = video * ((max_value - min_value) / 255) + min_value
    return video.permute(0, 4, 1, 2, 3).contiguous()


def _vae_output_to_image(vae_output, pattern="B C H W", min_value=-1, max_value=1):
    if pattern != "H W C":
        vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
    image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
    image = image.to(device="cpu", dtype=torch.uint8)
    image = Image.fromarray(image.numpy())
    return image


def vae_output_to_video(vae_output, pattern="B C T H W", min_value=-1, max_value=1):
    """VAE pixel tensor -> list[PIL.Image]."""
    if pattern != "T H W C":
        vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
    return [
        _vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output
    ]


def check_resize_height_width(
    height,
    width,
    num_frames=None,
    *,
    height_division_factor,
    width_division_factor,
    time_division_factor,
    time_division_remainder,
):
    """Round ``height``/``width``/``num_frames`` up to the backbone's model grid.

    ``time_division_factor == 1`` (per-frame encoders) skips the temporal check,
    which would otherwise bump ``num_frames`` by 1 every call (any ``N % 1 == 0``).
    """
    if height % height_division_factor != 0:
        height = (height + height_division_factor - 1) // height_division_factor * height_division_factor
        print(f"height % {height_division_factor} != 0. We round it up to {height}.")
    if width % width_division_factor != 0:
        width = (width + width_division_factor - 1) // width_division_factor * width_division_factor
        print(f"width % {width_division_factor} != 0. We round it up to {width}.")
    if num_frames is None:
        return height, width
    if time_division_factor > 1 and (num_frames % time_division_factor != time_division_remainder):
        num_frames = (
            num_frames + time_division_factor - 1
        ) // time_division_factor * time_division_factor + time_division_remainder
        print(f"num_frames % {time_division_factor} != {time_division_remainder}. We round it up to {num_frames}.")
    return height, width, num_frames


def generate_noise(shape, *, dtype, device, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32):
    """Seeded Gaussian noise built on ``rand_device`` then moved to ``(dtype, device)``."""
    generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
    return noise.to(dtype=dtype, device=device)
