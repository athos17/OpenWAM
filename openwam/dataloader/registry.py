"""Dataset registry for config-driven dataset construction.

Replaces the if-else dispatch chains in ``training/runtime.py`` with a
decorator-based registry pattern (consistent with the architecture registry).

Usage:
    # Register a dataset class
    @register_dataset("droid")
    class DROIDDataset(LeRobotBaseDataset):
        ...

    # Build from config
    dataset = build_dataset(config, split="train")
"""

from typing import Dict, Type

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.utils import get_cfg

DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class by name.

    Args:
        name: Config-level type name (e.g., "droid", "bridge_v2", "oxe").
    """

    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls

    return wrapper


def build_dataset(config, split: str = "train") -> BaseDataset:
    """Build a dataset from a config dict or DictConfig.

    Looks up the dataset class from ``DATASET_REGISTRY`` using ``config.type``
    and dispatches to ``cls.from_config(config, split)``. Every registered
    class is required to implement ``from_config`` — there is no fallback
    generic-kwargs path.

    Args:
        config: Dict-like config with at least a ``type`` field.
        split: "train" or "val".

    Returns:
        Instantiated dataset.
    """
    dtype = get_cfg(config, "type")
    if dtype not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type '{dtype}'. Available: {list(DATASET_REGISTRY.keys())}")

    cls = DATASET_REGISTRY[dtype]
    if not hasattr(cls, "from_config"):
        raise TypeError(
            f"Registered dataset class {cls.__name__} (type={dtype!r}) lacks a "
            f"from_config classmethod. All registered datasets must implement it."
        )
    publication = None
    if get_cfg(config, "require_publication", False):
        from openwam.dataloader.publication import source_publication_contract
        publication = source_publication_contract(config)
    dataset = cls.from_config(config, split=split)
    source_kind = get_cfg(config, "source_kind")
    if get_cfg(config, "source_family") == "ftp1":
        if source_kind not in (None, "robot"):
            raise ValueError("FTP sources must have robot actions; ego routing is forbidden")
        source_kind = "robot"
    allowed = {"ego": {"video_only", "video_tactile"},
               "robot": {"video_action", "video_action_tactile"}}
    if source_kind is not None:
        if source_kind not in allowed:
            raise ValueError(f"Unknown source_kind {source_kind!r}")
        groups = getattr(dataset, "modality_groups", None)
        modes = set(groups) if groups else {getattr(dataset, "training_modality", "video_action")}
        expected = set(get_cfg(config, "expected_modalities", list(allowed[source_kind])))
        if not expected or not expected <= allowed[source_kind] or not modes <= expected:
            raise ValueError(f"{source_kind} source declares {expected} but emits {modes}; modality fallback is forbidden")
    if publication is not None:
        dataset.publication_contract = publication
    return dataset


def list_registered_datasets():
    """Return list of registered dataset type names."""
    return sorted(DATASET_REGISTRY.keys())


# ---- Auto-registration of built-in datasets ----
# This runs when the module is first imported.


def _register_builtins():
    """Register all built-in dataset classes."""
    from openwam.dataloader.agibotworld import MultiAgiBotWorldDataset
    from openwam.dataloader.ebench import EBenchDataset
    from openwam.dataloader.interndata_a1 import InternDataA1Dataset
    from openwam.dataloader.libero import LiberoDataset
    from openwam.dataloader.mixture import MixtureDataset
    from openwam.dataloader.muka_franka import MukaFrankaDataset
    from openwam.dataloader.oxe_droid import OxeDroidDataset
    from openwam.dataloader.robocasa365 import MultiTaskRoboCasa365Dataset
    from openwam.dataloader.robocasa_gr1 import MultiRoboCasaGR1Dataset
    from openwam.dataloader.robocoin import MultiRobotCOINDataset
    from openwam.dataloader.robodojo import MultiTaskRoboDojoDataset
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset
    from openwam.dataloader.sharpadex_tianji import SharpaDexTianjiDataset
    from openwam.dataloader.vlabench import VLABenchDataset
    register_dataset("sharpadex_tianji")(SharpaDexTianjiDataset)
    from openwam.dataloader.sharpadex_tianji_global import SharpaDexTianjiGlobalDataset
    register_dataset("sharpadex_tianji_global")(SharpaDexTianjiGlobalDataset)
    from openwam.dataloader.sharpadex_native_tcp import SharpaDexNativeTCPDataset
    from openwam.dataloader.touchwam_pretrain import LeRobotVideoIntensityDataset, SharpaNativeIntensityDataset, EgoDexReuseVideoDataset
    register_dataset("sharpadex_native_tcp")(SharpaDexNativeTCPDataset)
    register_dataset("sharpadex_native_intensity")(SharpaNativeIntensityDataset)
    register_dataset("lerobot_video_intensity")(LeRobotVideoIntensityDataset)
    from openwam.dataloader.robot_intensity import LeRobotRobotIntensityDataset
    register_dataset("lerobot_robot_intensity")(LeRobotRobotIntensityDataset)
    register_dataset("egodex_reuse_video")(EgoDexReuseVideoDataset)

    from openwam.dataloader.egodex_video import EgoDexVideoDataset
    register_dataset("egodex_video")(EgoDexVideoDataset)

    from openwam.dataloader.bench2dex_tactile import Bench2DexTactileMixture

    register_dataset("bench2dex_tactile")(Bench2DexTactileMixture)
    from openwam.dataloader.bench2dex_deform import Bench2DexDeformMixture
    from openwam.dataloader.marvin_deform import MarvinDeformDataset

    register_dataset("bench2dex_deform")(Bench2DexDeformMixture)
    register_dataset("marvin_deform")(MarvinDeformDataset)
    register_dataset("robotwin")(MultiTaskRoboTwinDataset)
    register_dataset("robodojo")(MultiTaskRoboDojoDataset)
    register_dataset("agibotworld")(MultiAgiBotWorldDataset)
    register_dataset("mixture")(MixtureDataset)
    register_dataset("robocasa_gr1")(MultiRoboCasaGR1Dataset)
    register_dataset("robocoin")(MultiRobotCOINDataset)
    register_dataset("ebench")(EBenchDataset)
    register_dataset("libero")(LiberoDataset)
    register_dataset("muka_franka")(MukaFrankaDataset)
    register_dataset("oxe_droid")(OxeDroidDataset)
    register_dataset("robocasa365")(MultiTaskRoboCasa365Dataset)
    # Registered on the SINGLE-bucket class: its from_config returns either one
    # bucket or a MultiInternDataA1Dataset depending on whether dataset_dir
    # points at a bucket or at the dataset root.
    register_dataset("interndata_a1")(InternDataA1Dataset)
    register_dataset("vlabench")(VLABenchDataset)


_register_builtins()
