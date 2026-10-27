"""Declarative native-sensor to canonical-region mappings (no value conversion)."""

import json
from dataclasses import dataclass
from pathlib import Path

from .registry import REGISTRY_VERSION, region_spec

LAYOUT_VERSION = "touchwam_surface_layout_v1"


@dataclass(frozen=True)
class SurfaceSpec:
    native_key: str
    region: str
    sensor: str
    representation: str
    units: str
    native_shape: tuple[int, int]
    surface_id: int = 0
    instance: str = "default"
    native_name: str = ""
    mapping_note: str = ""
    geometry_ref: str = ""

    def __post_init__(self):
        spec = region_spec(self.region)
        object.__setattr__(self, "native_shape", tuple(self.native_shape))
        for field in ("native_key", "sensor", "representation", "units", "instance"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"Surface {field} must be a nonempty string")
        for field in ("native_name", "mapping_note", "geometry_ref"):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f"Surface {field} must be a string")
        if type(self.surface_id) is not int or not 0 <= self.surface_id <= 65535:
            raise ValueError("surface_id must be an integer in [0,65535]")
        if len(self.native_shape) != 2 or any(type(n) is not int or n < 1 for n in self.native_shape):
            raise ValueError("native_shape must contain positive integer rows, columns")
        if "generic" in spec.groups and not self.mapping_note.strip():
            raise ValueError("Generic mappings require an explicit mapping_note; never silently guess anatomy")

    @property
    def key(self):
        return region_spec(self.region).id, self.surface_id


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
            raise ValueError("Duplicate region/surface_id; assign stable surface IDs to repeated regions")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if set(data) != {"schema", "registry", "name", "surfaces"}:
            raise ValueError("Unexpected layout fields")
        if data["schema"] != LAYOUT_VERSION or data["registry"] != REGISTRY_VERSION:
            raise ValueError("Unsupported layout or registry version")
        return cls(
            data["name"], tuple(SurfaceSpec(**dict(s, native_shape=tuple(s["native_shape"]))) for s in data["surfaces"])
        )

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
            tactile_surface_ids=[s.surface_id for s in self.surfaces],
            tactile_sensor_names=[s.sensor for s in self.surfaces],
        )
