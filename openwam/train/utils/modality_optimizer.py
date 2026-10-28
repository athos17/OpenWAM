"""Preserve inactive stream weights and Adam state under ZeRO flattening.

ZeRO-2 fills unused gradients with zeros. Plain AdamW then applies momentum and
weight decay to inactive parameters in its flat partitions. Separate V/A/T
groups let us restore the usual ``grad is None`` semantics for a whole stream.
Activation is OR-ed across microbatches until the optimizer boundary.
"""

import torch
import torch.distributed as dist

from openwam.dataloader.modality_sampler import (
    MODALITY_CODES, TACTILE_CODEC_CODES, batch_modality_code, batch_tactile_codec,
)


def build_modality_groups(architecture, *, action_lr=None, video_lr=None):
    groups = {name: [] for name in ("video", "action", "proprio", "tactile")}
    separate_representations = bool(getattr(architecture, "tactile_spec", {}).get("rgb_enabled", False))
    if separate_representations:
        groups.update(tactile_intensity=[], tactile_rgb=[])
    owners = {"video_backbone": "video", "action_backbone": "action", "proprio_encoder": "proprio",
              "touch_v2": "tactile"}
    for name, module in architecture.get_trainable_modules().items():
        if name not in owners:
            raise ValueError(f"Unclassified TouchWAM trainable module: {name}")
        for parameter_name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            owner = owners[name]
            if name == "touch_v2" and separate_representations:
                head = parameter_name.split(".")[0]
                if head in ("input_projection", "spatial_projection", "output"):
                    owner = "tactile_intensity"
                elif head in ("rgb_input_projection", "rgb_spatial_projection", "rgb_output"):
                    owner = "tactile_rgb"
            groups[owner].append(parameter)
    actual = [p for values in groups.values() for p in values]
    if len({id(p) for p in actual}) != len(actual) or {id(p) for p in actual} != {
        id(p) for p in architecture.parameters() if p.requires_grad
    }:
        raise ValueError("TouchWAM optimizer groups must cover every trainable parameter exactly once")
    result = []
    for stream, params in groups.items():
        if not params:
            continue
        group = {"params": params, "touchwam_stream": stream, "touchwam_active_in_window": False}
        lr = video_lr if stream == "video" else action_lr if stream == "action" else None
        if lr is not None:
            group["lr"] = lr
        result.append(group)
    return result


class ModalityAwareAdamW(torch.optim.AdamW):
    """AdamW with serializable activation flags and unchanged inactive moments.

Call ``activate`` before every backward. Under ZeRO-2 the engine invokes this
optimizer once per partition group, so only the groups passed to ``step`` are
reset there. The trainer also resets all groups at each accumulation boundary
to cover an overflow-skipped engine step. CPU optimizer replacement is not
supported: Accelerate must retain this optimizer instance.
"""

    def activate(self, modality, tactile_codec=None):
        if modality not in MODALITY_CODES:
            raise ValueError(f"Unknown training modality: {modality}")
        active = {"video"}
        if "action" in modality:
            active.update(("action", "proprio"))
        if "tactile" in modality:
            active.add("tactile")
            codec = tactile_codec or "tactile_intensity_v1"
            if codec not in TACTILE_CODEC_CODES:
                raise ValueError(f"Unsupported tactile codec: {codec}")
            active.add("tactile_rgb" if codec == "tactile_rgb_v1" else "tactile_intensity")
        for group in self.param_groups:
            group["touchwam_active_in_window"] |= group["touchwam_stream"] in active

    def reset_activation(self):
        for group in self.param_groups:
            group["touchwam_active_in_window"] = False

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("TouchWAM modality-aware optimizer requires explicit forward/backward, no closure")
        held = []
        for group in self.param_groups:
            if not group["touchwam_active_in_window"]:
                for parameter in group["params"]:
                    held.append((parameter, parameter.grad))
                    parameter.grad = None
        try:
            return super().step()
        finally:
            for parameter, gradient in held:
                parameter.grad = gradient
            self.reset_activation()


def validate_global_modality_layout(samples, device):
    """All ranks reject invalid/mixed graphs before entering model collectives.

Region counts may differ. A T microbatch requires a nonempty declared layout
on every rank; missing observations/targets retain the layout and use masks.
True N=0 belongs to V/VA microbatches. Do not manufacture a tactile layout.
"""
    code = batch_modality_code(samples)
    valid = code >= 0 and all(
        bool(s.get("tactile_region_names")) == ("tactile" in s.get("training_modality", "video_action"))
        for s in samples
    )
    try:
        codec = TACTILE_CODEC_CODES[batch_tactile_codec(samples)]
    except ValueError:
        codec, valid = -1, False
    state = torch.tensor([code, int(valid), codec], device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        states = [torch.empty_like(state) for _ in range(dist.get_world_size())]
        dist.all_gather(states, state)
    else:
        states = [state]
    values = [s.tolist() for s in states]
    if not all(v[1] for v in values) or len({(v[0], v[2]) for v in values}) != 1:
        raise ValueError(
            f"TouchWAM distributed microbatches require one global modality, tactile codec and matching layout presence; "
            f"got [modality_code, valid_layout, tactile_codec] per rank: {values}. Use GlobalModalitySampler; "
            "T samples keep their real regional layout and mark missing data with masks; V/VA use N=0."
        )
    return next(name for name, value in MODALITY_CODES.items() if value == code)
