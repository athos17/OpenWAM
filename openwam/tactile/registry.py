"""Frozen, sensor-independent tactile region vocabulary.

IDs identify coarse physical regions, not taxels, sensor models or token offsets.
Each sample uses at most one map per region. Native sensors sharing a region
must be composed by the dataset adapter. Single-arm grippers use the right side.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

REGISTRY_VERSION = "touchwam_regions_67_v2"
PREVIOUS_REGISTRY_VERSION = "touchwam_regions_64_v1"
PREVIOUS_REGISTRY_SHA256 = "15ec04d9d4bc5262f2aa9af960e58595ec3e4329d4f40f3b33e3c52d7c2464a4"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIP_NAMES = tuple(f"{side}.{finger}.tip" for side in ("right", "left") for finger in FINGERS)
LEGACY_REGION_NAMES = (*FINGERTIP_NAMES, "right.palm", "left.palm")


@dataclass(frozen=True)
class RegionSpec:
    id: int
    name: str
    family: str
    side: str
    groups: tuple[str, ...] = ()


def _build_registry():
    rows = []

    def add(name, family, side, *groups):
        rows.append(RegionSpec(len(rows), name, family, side, tuple(groups)))

    # The first twelve IDs are an immutable compatibility prefix.
    for name in FINGERTIP_NAMES:
        add(name, "hand", name.split(".")[0], "finger", "fingertip")
    for side in ("right", "left"):
        add(f"{side}.palm", "hand", side, "palm")
    for side in ("right", "left"):
        add(f"{side}.thumb.proximal", "hand", side, "finger")
        # A composite coverage class; do not label a base/thenar pad pure palm.
        add(f"{side}.thumb.base_thenar", "hand", side, "finger", "palm", "composite")
        for finger in FINGERS[1:]:
            for segment in ("middle", "proximal", "base"):
                add(f"{side}.{finger}.{segment}", "hand", side, "finger")
        for part in ("thenar", "hypothenar", "central", "distal", "proximal"):
            add(f"{side}.palm.{part}", "hand", side, "palm")
        add(f"{side}.hand.dorsal", "hand", side, "dorsal")
        add(f"{side}.hand.other", "hand", side, "generic")
        add(f"{side}.wrist", "arm", side)
        add(f"{side}.forearm", "arm", side)
    # v2 explicitly assigns the old unsided gripper slots to the canonical
    # right/single arm. This semantic change requires registry warm-start.
    for part in ("jaw_a.surface", "jaw_b.surface", "base"):
        add(f"right.gripper.{part}", "gripper", "right", "gripper")
    for name in ("generic.surface", "generic.tool", "generic.body"):
        add(name, "generic", "unspecified", "generic")
    # Keep hand IDs and generic IDs unchanged.
    for part in ("jaw_a.surface", "jaw_b.surface", "base"):
        add(f"left.gripper.{part}", "gripper", "left", "gripper")
    return tuple(rows)


REGIONS = _build_registry()
REGION_NAMES = tuple(region.name for region in REGIONS)
PREVIOUS_REGION_NAMES = (
    *REGION_NAMES[:58],
    "gripper.jaw_a.surface",
    "gripper.jaw_b.surface",
    "gripper.base",
    *REGION_NAMES[61:64],
)
REGISTRY_SHA256 = hashlib.sha256(
    json.dumps([asdict(r) for r in REGIONS], sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def region_spec(name):
    try:
        return REGIONS[REGION_NAMES.index(name)]
    except ValueError:
        raise ValueError(f"Unregistered tactile region {name!r}; declare an explicit canonical mapping") from None


def validate_registry(names):
    """Legacy twelve-region models remain loadable with their original config."""
    if tuple(names) not in (REGION_NAMES, LEGACY_REGION_NAMES):
        raise ValueError(
            "region_names must be the frozen 67-region registry or the original 12-region registry; migrate v1 explicitly"
        )


def registry_document():
    return dict(version=REGISTRY_VERSION, sha256=REGISTRY_SHA256, regions=[asdict(r) for r in REGIONS])


if __name__ == "__main__":
    print(json.dumps(registry_document(), ensure_ascii=False, indent=2))
