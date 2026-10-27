"""TouchWAM tactile generation, compatible with existing V2 checkpoints."""

from .architecture import TouchWAMArchitecture
from .regional import RegionalTouchWAMArchitecture

__all__ = ["TouchWAMArchitecture", "RegionalTouchWAMArchitecture"]
