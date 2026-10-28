"""Declarative native-sensor to canonical-region mappings (no value conversion)."""

import json
from dataclasses import dataclass
from pathlib import Path

from .composition import SensorTile, compose_mosaic, mosaic_roi
from .registry import REGISTRY_VERSION, region_spec

LAYOUT_VERSION = "touchwam_region_layout_v2"


@dataclass(frozen=True)
class SurfaceSpec:
    native_key: str
    region: str
    sensor: str
    representation: str
    units: str
    native_shape: tuple[int, int]
    tiles: tuple[SensorTile, ...] = ()
    native_name: str = ""
    mapping_note: str = ""
    geometry_ref: str = ""

    def __post_init__(self):
        spec = region_spec(self.region)
        object.__setattr__(self, "native_shape", tuple(self.native_shape))
        object.__setattr__(self, "tiles", tuple(self.tiles))
        for field in ("native_key", "sensor", "representation", "units"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"Surface {field} must be a nonempty string")
        for field in ("native_name", "mapping_note", "geometry_ref"):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f"Surface {field} must be a string")
        if len(self.native_shape) != 2 or any(type(n) is not int or n < 1 for n in self.native_shape):
            raise ValueError("native_shape must contain positive integer rows, columns")
        if "generic" in spec.groups and not self.mapping_note.strip():
            raise ValueError("Generic mappings require an explicit mapping_note; never silently guess anatomy")
        if self.tiles:
            mosaic_roi(self.native_shape, self.tiles)

    @property
    def key(self):
        return region_spec(self.region).id

    @property
    def source_tiles(self):
        return self.tiles or (SensorTile(self.native_key, self.native_shape, (0, 0)),)


@dataclass(frozen=True)
class SurfaceLayout:
    name: str
    surfaces: tuple[SurfaceSpec, ...]

    def __post_init__(self):
        object.__setattr__(self, "surfaces", tuple(self.surfaces))
        if not isinstance(self.name, str) or not self.name.strip() or not self.surfaces:
            raise ValueError("Layout requires a name and at least one surface")
        if len({s.native_key for s in self.surfaces}) != len(self.surfaces):
            raise ValueError("Duplicate native_key in layout")
        if len({s.key for s in self.surfaces}) != len(self.surfaces):
            raise ValueError("Duplicate region; compose native sensors into one regional map first")
        source_keys = [t.native_key for s in self.surfaces for t in s.source_tiles]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("A native sensor cannot be duplicated across regional supervision")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if set(data) != {"schema", "registry", "name", "surfaces"}:
            raise ValueError("Unexpected layout fields")
        if data["schema"] != LAYOUT_VERSION or data["registry"] != REGISTRY_VERSION:
            raise ValueError("Unsupported layout or registry version")
        surfaces = []
        for source in data["surfaces"]:
            source = dict(source)
            source["tiles"] = tuple(SensorTile(**t) for t in source.get("tiles", []))
            surfaces.append(SurfaceSpec(**source))
        return cls(data["name"], tuple(surfaces))

    def select(self, mode="all"):
        if mode not in ("all", "fingertips", "without_palm"):
            raise ValueError("Unknown layout selection")
        surfaces = tuple(
            s
            for s in self.surfaces
            if mode == "all"
            or (mode == "fingertips" and "fingertip" in region_spec(s.region).groups)
            or (mode == "without_palm" and "palm" not in region_spec(s.region).groups)
        )
        return SurfaceLayout(self.name, surfaces)

    def identity_fields(self):
        """Reader metadata only; does not assert codec/action/timing support."""
        return dict(
            tactile_region_names=[s.region for s in self.surfaces],
            tactile_sensor_names=[s.sensor for s in self.surfaces],
        )

    def compose(self, packets, *, observed=None, dtype="float32"):
        """Native maps -> one map/validity/ROI per region, before codec/resize.

        Per-source readers align timestamps and physical units first. All
        constituent sensors of one region must be observed for a valid frame.
        """
        values = {
            s.region: compose_mosaic(s.native_shape, s.source_tiles, packets, observed=observed, dtype=dtype)
            for s in self.surfaces
        }
        return dict(
            maps={k: v[0] for k, v in values.items()},
            valid={k: v[1] for k, v in values.items()},
            spatial_valid={k: v[2] for k, v in values.items()},
        )
