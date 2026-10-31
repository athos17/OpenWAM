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
    observation = getattr(architecture, "tactile_spec", {}).get("observation_encoder") or {}
    encoder_specs = list(observation.get("sensors", ())) if observation.get("kind") == "ftp1_heterogeneous_v1" else []
    encoder_groups = {}
    image_ids = [i for i, item in enumerate(encoder_specs) if item["kind"] == "image"]
    if encoder_specs:
        groups["tactile_observation_image_shared"] = []
        encoder_groups["tactile_observation_image_shared"] = image_ids
        for i, item in enumerate(encoder_specs):
            group = "tactile_observation_" + item["kind"] + ":" + item["name"]
            groups[group] = []
            encoder_groups[group] = [i]
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
            if name == "touch_v2" and parameter_name.startswith("observation_encoder."):
                parts = parameter_name.split(".")
                if not encoder_specs:
                    raise ValueError("Observation encoder parameters require an explicit heterogeneous specification")
                if parts[1] == "image_frontends":
                    owner = "tactile_observation_image:" + parts[2]
                elif parts[1] == "matrix_encoders":
                    owner = "tactile_observation_matrix:" + parts[2]
                elif parts[1] in ("image_trunk", "image_projection"):
                    owner = "tactile_observation_image_shared"
                else:
                    raise ValueError(f"Unclassified observation encoder parameter: {parameter_name}")
                if owner not in groups:
                    raise ValueError(f"Observation encoder absent from specification: {owner}")
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
        if stream in encoder_groups:
            group["touchwam_observation_encoder_ids"] = encoder_groups[stream]
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

    def activate(self, modality, tactile_codec=None, *, observation_encoder_ids=()):
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
        observed = set(observation_encoder_ids) if "tactile" in modality else set()
        for group in self.param_groups:
            enabled = (bool(observed.intersection(group["touchwam_observation_encoder_ids"]))
                       if "touchwam_observation_encoder_ids" in group else group["touchwam_stream"] in active)
            group["touchwam_active_in_window"] |= enabled

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
            else:
                # ZeRO-2 hands active groups zero gradients for parameters unused in this
                # window (e.g. a tactile block's action read on VT batches), so AdamW still
                # applies their moments and weight decay. Backends that leave those gradients
                # None (FSDP2) get the same zeros; under ZeRO-2 this finds nothing to fill.
                for parameter in group["params"]:
                    if parameter.grad is None and parameter.requires_grad:
                        parameter.grad = torch.zeros_like(parameter)
        try:
            return super().step()
        finally:
            for parameter, gradient in held:
                parameter.grad = gradient
            self.reset_activation()


def _local_layout_state(samples):
    code = batch_modality_code(samples)
    valid = code >= 0 and all(
        bool(s.get("tactile_region_names")) == ("tactile" in s.get("training_modality", "video_action"))
        for s in samples
    )
    try:
        codec = TACTILE_CODEC_CODES[batch_tactile_codec(samples)]
    except ValueError:
        codec, valid = -1, False
    from openwam.dataloader.modality_sampler import batch_tactile_observation_shape
    try:
        shape = batch_tactile_observation_shape(samples) or (0, 0)
    except ValueError:
        shape, valid = (-1, -1), False
    return [code, int(valid), codec, *shape]


def _check_layout_states(values):
    if not all(v[1] for v in values) or len({(v[0], *v[2:]) for v in values}) != 1:
        raise ValueError(
            f"TouchWAM distributed microbatches require one global modality, tactile codec, observation shape and matching layout presence; "
            f"got [modality_code, valid_layout, tactile_codec, observation_h, observation_w] per rank: {values}. Use GlobalModalitySampler; "
            "T samples keep their real regional layout and mark missing data with masks; V/VA use N=0."
        )
    return next(name for name, value in MODALITY_CODES.items() if value == values[0][0])


def validate_global_modality_layout(samples, device):
    """All ranks reject invalid/mixed graphs before entering model collectives.

Region counts may differ. A T microbatch requires a nonempty declared layout
on every rank; missing observations/targets retain the layout and use masks.
True N=0 belongs to V/VA microbatches. Do not manufacture a tactile layout.
"""
    state = torch.tensor(_local_layout_state(samples), device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        states = [torch.empty_like(state) for _ in range(dist.get_world_size())]
        dist.all_gather(states, state)
    else:
        states = [state]
    return _check_layout_states([s.tolist() for s in states])


class DeferredModalitySync:
    """``training.touchwam_p1.deferred_modality_sync``: same checks and optimizer activation,
    without stalling the host mid-step.

    The blocking path all-gathers the layout before ``prepare_inputs`` and all-reduces the
    observed encoder IDs right after the forward; each ``.tolist()`` drains the CUDA stream
    and waits for the slowest rank. Here the layout (CPU batch metadata) goes through a
    gloo all-gather in the background, and the per-encoder activity bitmask is all-reduced on
    the stream and copied to pinned memory. Both are resolved after ``engine.backward`` has
    queued the whole backward and before ``engine.step`` runs the optimizer, which is the
    only consumer. The first ``blocking_steps`` steps keep the blocking check, so a broken
    sampler still fails before any ZeRO collective at startup.
    """

    def __init__(self, device, encoder_count, blocking_steps=20):
        self.device, self.encoder_count, self.blocking_steps = device, encoder_count, blocking_steps
        self.distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        # Collective call: every rank constructs this object at the same point of startup.
        self.group = dist.new_group(backend="gloo") if self.distributed else None
        self._mask_host = torch.zeros(encoder_count, dtype=torch.int64, pin_memory=torch.cuda.is_available())
        self.reset()

    def reset(self):
        self._modality = self._layout_work = self._layout_out = None
        self._codec = None
        self._mask_event = None

    def start_layout(self, samples, step):
        from openwam.dataloader.modality_sampler import batch_tactile_codec

        if self._modality is not None or self._layout_work is not None:
            raise RuntimeError("Previous microbatch modality was never resolved")
        if step < self.blocking_steps or not self.distributed:
            self._modality = validate_global_modality_layout(samples, self.device)
        else:
            state = torch.tensor(_local_layout_state(samples), dtype=torch.int64)
            self._layout_out = [torch.empty_like(state) for _ in range(dist.get_world_size())]
            self._layout_work = dist.all_gather(self._layout_out, state, group=self.group, async_op=True)
        try:
            self._codec = batch_tactile_codec(samples)
        except ValueError:
            self._codec = None  # the layout check reports the mixed codec on every rank

    def start_encoders(self, mask):
        """``mask``: int64 [encoder_count] on device, 1 where an encoder observed real input."""
        if mask.shape != (self.encoder_count,) or mask.dtype != torch.int64:
            raise ValueError("Observation encoder activity must be int64 [encoder_count]")
        if self.distributed:
            dist.all_reduce(mask, op=dist.ReduceOp.MAX, async_op=True).wait()  # stream wait only
        self._mask_host.copy_(mask, non_blocking=True)
        self._mask_event = torch.cuda.Event()
        self._mask_event.record()

    def resolve(self):
        """(modality, observed encoder IDs) for ``ModalityAwareAdamW.activate``."""
        if self._layout_work is not None:
            self._layout_work.wait()
            self._modality = _check_layout_states([s.tolist() for s in self._layout_out])
        if self._modality is None:
            raise RuntimeError("resolve() before start_layout()")
        ids = []
        if self._mask_event is not None:
            self._mask_event.synchronize()
            ids = self._mask_host.nonzero(as_tuple=False).flatten().tolist()
        modality, codec = self._modality, self._codec
        self.reset()
        return modality, codec, ids


def global_observation_encoder_ids(inputs, encoder_count, device, *, actual_ids=None):
    """Union actual observed frontends across ranks before a ZeRO optimizer step.

    Region availability and future labels alone never activate observation
    parameters. Missing history leaves moments/decay untouched, including when
    a different rank is the only one with an observed sensor.
    """
    active = torch.zeros(encoder_count, dtype=torch.int64, device=device)
    ids = inputs.get("tactile_encoder_ids")
    valid = inputs.get("tactile_valid")
    if actual_ids is not None:
        selected = torch.as_tensor(actual_ids, device=device, dtype=torch.long)
        if ((selected < 0) | (selected >= encoder_count)).any():
            raise ValueError("Observed encoder ID is outside the configured registry")
        active[selected] = 1
    elif ids is not None and valid is not None:
        ids = torch.as_tensor(ids, device=device)
        valid = torch.as_tensor(valid, device=device)
        if valid.dtype != torch.bool or ids.dtype != torch.long or valid.ndim != 3 or ids.shape != valid.shape[::2]:
            raise ValueError("Observation activation requires [B,N] encoder IDs and [B,T,N] validity")
        observed = valid.any(1)
        present = inputs.get("tactile_layout_present")
        if present is not None:
            observed &= torch.as_tensor(present, device=device)
        spatial = inputs.get("tactile_observation_spatial_valid")
        if spatial is not None:
            observed &= torch.as_tensor(spatial, device=device).flatten(2).any(-1)
        selected = ids[observed]
        if ((selected < 0) | (selected >= encoder_count)).any():
            raise ValueError("Observed encoder ID is outside the configured registry")
        active[selected] = 1
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(active, op=dist.ReduceOp.MAX)
    return active.nonzero(as_tuple=False).flatten().tolist()
