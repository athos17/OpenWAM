"""Training-time action prefix conditioning (arXiv:2512.05964).

OpenWAM uses sigma=0 for clean data, opposite to the paper's time axis.
The maximum delay is inclusive. No parameters are added to the model.
"""

from dataclasses import dataclass
from numbers import Integral

import torch


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = False
    max_delay_steps: int = 16
    delay_distribution: str = "uniform"


def rtc_config(config=None) -> RTCConfig:
    if isinstance(config, RTCConfig):
        return config
    config = config or {}
    enabled = config.get("enabled", False)
    maximum = config.get("max_delay_steps", 16)
    distribution = config.get("delay_distribution", "uniform")
    if not isinstance(enabled, bool):
        raise ValueError("training.rtc.enabled must be boolean")
    if isinstance(maximum, bool) or not isinstance(maximum, Integral) or maximum < 0:
        raise ValueError("training.rtc.max_delay_steps must be a non-negative integer (inclusive)")
    if distribution != "uniform":
        raise ValueError("training.rtc.delay_distribution must be 'uniform'")
    return RTCConfig(enabled, int(maximum), distribution)


def condition_training_actions(actions, noisy_actions, timesteps, action_is_pad, config, delay=None):
    """Sample one delay per example; retain at least one real postfix target.

    Only an initial contiguous span of valid actions can become a prefix.
    Samples without action supervision get d=0 and retain their original mask.
    ``delay`` ([B] ints, optional) supplies per-example delays drawn by the data
    pipeline (e.g. subtask prompts that depend on the first generated step); they
    are clamped to the same bounds as sampled delays.
    """
    batch, horizon, _ = actions.shape
    if config.max_delay_steps >= horizon:
        raise ValueError("training.rtc.max_delay_steps must be < action horizon")
    if action_is_pad is None:
        pad = torch.zeros_like(actions, dtype=torch.bool)
    else:
        pad = action_is_pad.to(device=actions.device, dtype=torch.bool)
        if pad.ndim == 2:
            pad = pad.unsqueeze(-1).expand_as(actions)
        if pad.shape != actions.shape:
            raise ValueError("RTC requires action padding mask [B,H] or [B,H,D]")
    valid_steps = (~pad).any(dim=-1)
    contiguous = valid_steps.long().cumprod(dim=1).sum(dim=1)
    maximum = (contiguous - 1).clamp(min=0, max=config.max_delay_steps)
    if delay is not None:
        delay = torch.as_tensor(delay, device=actions.device).long().reshape(batch)
        if (delay < 0).any():
            raise ValueError("RTC delays must be non-negative")
        delay = torch.minimum(delay, maximum)
    elif config.max_delay_steps:
        delay = (torch.rand(batch, device=actions.device) * (maximum + 1)).long()
    else:
        delay = torch.zeros_like(maximum)
    prefix = torch.arange(horizon, device=actions.device)[None, :] < delay[:, None]
    noisy = torch.where(prefix[..., None], actions, noisy_actions)
    times = torch.where(prefix, torch.zeros_like(timesteps[:, None]), timesteps[:, None])
    return noisy, times, pad | prefix[..., None]
