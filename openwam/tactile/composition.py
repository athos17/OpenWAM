"""Deterministic 2D sensor mosaics before source-specific normalization/codecs.

Tiles must already share a physical representation, units and calibration.
The explicit placement is a layout convention, not automatic 3D registration.
Missing tiles invalidate the entire region frame (conservative v2 contract).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorTile:
    native_key: str
    native_shape: tuple[int, int]
    origin: tuple[int, int]
    quarter_turns: int = 0

    def __post_init__(self):
        object.__setattr__(self, "native_shape", tuple(self.native_shape))
        object.__setattr__(self, "origin", tuple(self.origin))
        if not isinstance(self.native_key, str) or not self.native_key.strip():
            raise ValueError("Tile native_key must be nonempty")
        if len(self.native_shape) != 2 or any(type(x) is not int or x < 1 for x in self.native_shape):
            raise ValueError("Tile shape must contain two positive integers")
        if len(self.origin) != 2 or any(type(x) is not int or x < 0 for x in self.origin):
            raise ValueError("Tile origin must contain nonnegative row, column")
        if type(self.quarter_turns) is not int or self.quarter_turns not in range(4):
            raise ValueError("quarter_turns must be an integer in [0,3]")

    @property
    def output_shape(self):
        return self.native_shape[::-1] if self.quarter_turns % 2 else self.native_shape


def mosaic_roi(shape, tiles):
    """Validate fixed, disjoint placement; holes remain outside the spatial ROI."""
    roi = np.zeros(shape, dtype=bool)
    if not tiles or len({t.native_key for t in tiles}) != len(tiles):
        raise ValueError("Mosaic requires unique native tile keys")
    for tile in tiles:
        y, x = tile.origin
        h, w = tile.output_shape
        if y + h > shape[0] or x + w > shape[1] or roi[y : y + h, x : x + w].any():
            raise ValueError("Mosaic tiles overlap or exceed the output grid")
        roi[y : y + h, x : x + w] = True
    return roi


def compose_mosaic(shape, tiles, packets, *, observed=None, dtype=np.float32):
    """Return (map, observed, static ROI), keeping placement stable on dropouts.

    packets maps native_key to a numeric 2D array or None. observed may mark
    stale/invalid packets explicitly. If any source is absent, discard the
    entire frame rather than treating the missing part as a measured zero.
    """
    roi = mosaic_roi(shape, tiles)
    dtype = np.dtype(dtype)
    if dtype.kind not in "uif":
        raise ValueError("Mosaic requires numeric dtype")
    result = np.zeros(shape, dtype=dtype)
    complete = True
    for tile in tiles:
        value = packets.get(tile.native_key)
        if value is None or (observed is not None and not observed.get(tile.native_key, False)):
            complete = False
            continue
        value = np.asarray(value)
        if value.shape != tile.native_shape or value.dtype != dtype:
            raise ValueError("Tile shape/dtype differs from the declared layout; convert units upstream")
        if not np.isfinite(value).all():
            complete = False
            continue
        y, x = tile.origin
        h, w = tile.output_shape
        result[y : y + h, x : x + w] = np.rot90(value, tile.quarter_turns)
    if not complete:
        result.fill(0)
    return result, complete, roi
