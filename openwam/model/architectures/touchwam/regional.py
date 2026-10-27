"""Optional-palm deformation generation using the existing V/A/T bridges."""

import json

import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.tactile.regions import CODEC, REGION_NAMES, SCHEMA, collate_regions

from .architecture import TouchWAMArchitecture
from .regional_modules import CODECS, RegionalTactileFlowStream, regional_flow_loss


@register_architecture(
    "touchwam_regions",
    framework="dual_system",
    variant="touchwam_regions",
    note="Named deformation regions with optional palms and independent observation/query masks.",
)
class RegionalTouchWAMArchitecture(TouchWAMArchitecture):
    model_version = "touchwam_regions_v1"

    def _init_tactile(self, cfg):
        defaults = dict(
            history=4,
            map_size=32,
            patch_size=16,
            width=256,
            heads=8,
            layers=6,
            history_layers=2,
            bridge_layers=None,
            future_frames=8,
            future_stride=3,
            fps=30,
            loss_weight=1.0,
            foreground_weight=1.0,
            codec=CODEC,
            region_names=list(REGION_NAMES),
            sensor_names=["sharpa_real", "bench2dex_sim"],
            palm_mode="auto",
            palm_dropout=0.0,
            history_dropout=0.0,
            allow_missing_history=False,
        )
        provided = dict(cfg.get("tactile", {}))
        if set(provided) - set(defaults):
            raise ValueError(f"Unknown regional tactile settings: {sorted(set(provided) - set(defaults))}")
        defaults.update(provided)
        s = self.tactile_spec = defaults
        for key in (
            "history",
            "map_size",
            "patch_size",
            "width",
            "heads",
            "layers",
            "history_layers",
            "future_frames",
            "future_stride",
            "fps",
        ):
            if type(s[key]) is not int or s[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if s["width"] % s["heads"] or s["width"] % 2:
            raise ValueError("Invalid width/heads")
        for key in ("loss_weight", "foreground_weight", "palm_dropout", "history_dropout"):
            value = s[key]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not torch.isfinite(torch.tensor(value))
            ):
                raise ValueError(f"Invalid {key}")
        if s["loss_weight"] < 0 or s["foreground_weight"] < 1:
            raise ValueError("Invalid loss weights")
        if any(not 0 <= s[k] <= 1 for k in ("palm_dropout", "history_dropout")):
            raise ValueError("Dropout probabilities must be in [0,1]")
        if s["palm_mode"] not in ("auto", "off") or type(s["allow_missing_history"]) is not bool:
            raise ValueError("Invalid palm mode / missing-history policy")
        if s["allow_missing_history"] and s["history_dropout"] == 0:
            raise ValueError("Missing-history generation requires a history_dropout training configuration")
        for key in ("region_names", "sensor_names"):
            s[key] = list(s[key])
            if not s[key] or any(not isinstance(x, str) or not x for x in s[key]) or len(set(s[key])) != len(s[key]):
                raise ValueError(f"{key} must contain unique names")
        if s["codec"] not in CODECS:
            raise ValueError(f"Unsupported tactile codec: {s['codec']}; pressure is not deform")
        depth = self.action_backbone.num_layers
        layers = s["bridge_layers"]
        layers = [(i + 1) * depth // s["layers"] - 1 for i in range(s["layers"])] if layers is None else list(layers)
        if (
            len(layers) != s["layers"]
            or layers != sorted(set(layers))
            or any(type(i) is not int or not 0 <= i < depth for i in layers)
        ):
            raise ValueError("bridge_layers must be ordered, unique and inside the V/A depth")
        s["bridge_layers"] = layers
        self.touch_v2 = RegionalTactileFlowStream(s, self.video_backbone.dim, self.action_backbone.dim)

    @property
    def checkpoint_metadata(self):
        return dict(
            model_version=self.model_version,
            tactile_schema=SCHEMA,
            tactile_codec=self.tactile_spec["codec"],
            tactile_spec=json.dumps(self.tactile_spec, sort_keys=True),
        )

    @property
    def tactile_input_contract(self):
        s = self.tactile_spec
        return dict(
            model_version=self.model_version,
            schema=SCHEMA,
            codec=s["codec"],
            region_names=s["region_names"],
            sensor_names=s["sensor_names"],
            representation="deform_gray",
            channels=["deform_intensity"],
            float_range=[0, 1],
            history=s["history"],
            map_size=s["map_size"],
            fps=s["fps"],
            future_frames=s["future_frames"],
            future_stride=s["future_stride"],
            layout="tactile_region_ids + tactile_layout_present",
            observation_mask="tactile_valid",
            query_mask="tactile_query_mask",
            spatial_mask="tactile_spatial_valid",
            palm_mode=s["palm_mode"],
            allow_missing_history=s["allow_missing_history"],
            output_validity="generation request, not measured contact or sensor validity",
        )

    def _condition(
        self,
        maps,
        valid,
        batch_size,
        *,
        region_ids=None,
        layout_present=None,
        spatial_valid=None,
        sensor_ids=None,
        palm_mode=None,
        augment=False,
    ):
        s, device = self.tactile_spec, self.device

        def tensor(value, ndim, dtype, name):
            value = torch.as_tensor(value, device=device)
            if value.ndim == ndim - 1 and batch_size == 1:
                value = value.unsqueeze(0)
            if value.ndim != ndim or value.shape[0] != batch_size or value.dtype != dtype:
                raise ValueError(f"Invalid {name}: expected {dtype}, rank {ndim}, batch {batch_size}")
            return value

        if region_ids is None:
            if maps is not None or valid is not None or layout_present is not None:
                raise ValueError("Explicit tactile_region_ids and tactile_layout_present are required")
            region_ids = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
            layout_present = torch.zeros_like(region_ids, dtype=torch.bool)
        if layout_present is None:
            raise ValueError("Explicit tactile_layout_present is required")
        ids = tensor(region_ids, 2, torch.long, "region IDs")
        present = tensor(layout_present, 2, torch.bool, "layout presence")
        if ids.shape != present.shape or ids.shape[1] < 1 or ((ids < -1) | (ids >= len(s["region_names"]))).any():
            raise ValueError("Invalid region layout")
        if (present & (ids < 0)).any():
            raise ValueError("Padding IDs cannot be present")
        for row in ids:
            known = row[row >= 0]
            if known.unique().numel() != known.numel():
                raise ValueError("Duplicate region identities in one sample")
        palm_ids = [i for i, name in enumerate(s["region_names"]) if name.endswith(".palm") or ".palm." in name]
        palms = torch.zeros_like(present)
        for i in palm_ids:
            palms |= ids == i
        mode = s["palm_mode"] if palm_mode is None else palm_mode
        if mode not in ("auto", "off"):
            raise ValueError("palm_mode must be auto or off")
        active = present.clone()
        if mode == "off":
            active &= ~palms
        if augment and s["palm_dropout"] and palms.any():
            drop = torch.rand(batch_size, 1, device=device) < s["palm_dropout"]
            active &= ~(drop & palms)
        n = ids.shape[1]
        shape = (batch_size, n, 1, s["map_size"], s["map_size"])
        spatial = (
            torch.ones(shape, dtype=torch.bool, device=device)
            if spatial_valid is None
            else tensor(spatial_valid, 5, torch.bool, "spatial mask")
        )
        if spatial.shape != shape:
            raise ValueError("Spatial mask shape differs from layout")
        active &= spatial.flatten(2).any(-1)
        spatial = spatial & active[..., None, None, None]
        sensors = torch.zeros_like(ids) if sensor_ids is None else tensor(sensor_ids, 2, torch.long, "sensor IDs")
        if sensors.shape != ids.shape or ((sensors < 0) | (sensors >= len(s["sensor_names"]))).any():
            raise ValueError("Unknown sensor identity")
        if maps is None:
            if valid is not None:
                raise ValueError("Observation validity requires maps")
            maps = torch.zeros(
                batch_size, s["history"], n, 1, s["map_size"], s["map_size"], device=device, dtype=self.dtype
            )
            valid = torch.zeros(batch_size, s["history"], n, device=device, dtype=torch.bool)
        elif valid is None:
            raise ValueError("Maps require explicit observation validity")
        maps = torch.as_tensor(maps, device=device)
        if maps.dtype == torch.uint8:
            maps = maps.to(self.dtype) / 255
        elif not maps.is_floating_point():
            raise ValueError("Deform maps must be uint8 or float [0,1]")
        else:
            maps = maps.to(self.dtype)
        if maps.ndim == 5 and batch_size == 1:
            maps = maps.unsqueeze(0)
        if maps.shape != (batch_size, s["history"], n, 1, s["map_size"], s["map_size"]):
            raise ValueError("History map shape differs from layout/configuration")
        valid = tensor(valid, 3, torch.bool, "observation validity")
        if valid.shape != maps.shape[:3]:
            raise ValueError("Observation mask shape differs from history")
        valid = valid & active[:, None]
        if augment and s["history_dropout"]:
            # Drop entire region histories, so completely stale sensors occur in training.
            valid &= ~(torch.rand(batch_size, 1, n, device=device) < s["history_dropout"])
        history, mask, available = self.touch_v2.encode_history(maps, valid, ids, sensors, spatial)
        return history, mask, ids, active, spatial, sensors, available

    def _query(self, condition, requested=None):
        active, spatial = condition[3], condition[4]
        shape = (active.shape[0], self.tactile_spec["future_frames"], active.shape[1])
        if requested is None:
            requested = active[:, None].expand(shape)
        else:
            requested = torch.as_tensor(requested, device=self.device)
            if requested.ndim == 2 and shape[0] == 1:
                requested = requested.unsqueeze(0)
            if requested.dtype != torch.bool or requested.shape != shape:
                raise ValueError("Query mask must be bool [B,F,N]")
        requested = requested & active[:, None]
        return requested, self.touch_v2.codec.pixel_mask(requested, spatial)

    def forward(
        self,
        noisy_actions,
        action_timestep,
        *,
        tactile_maps=None,
        tactile_valid=None,
        tactile_region_ids=None,
        tactile_layout_present=None,
        tactile_spatial_valid=None,
        tactile_sensor_ids=None,
        tactile_palm_mode=None,
        tactile_query_mask=None,
        touch_condition=None,
        **kwargs,
    ):
        if "future_tactile_maps" in kwargs or "future_tactile_valid" in kwargs:
            raise ValueError("Clean future tactile targets must not enter forward; use compute_loss")
        if touch_condition is None:
            touch_condition = self._condition(
                tactile_maps,
                tactile_valid,
                noisy_actions.shape[0],
                region_ids=tactile_region_ids,
                layout_present=tactile_layout_present,
                spatial_valid=tactile_spatial_valid,
                sensor_ids=tactile_sensor_ids,
                palm_mode=tactile_palm_mode,
            )
        elif any(
            x is not None
            for x in (
                tactile_maps,
                tactile_valid,
                tactile_region_ids,
                tactile_layout_present,
                tactile_spatial_valid,
                tactile_sensor_ids,
                tactile_palm_mode,
            )
        ):
            raise ValueError("Pass raw history/layout OR touch_condition")
        if tactile_query_mask is not None:
            if kwargs.get("tactile_future_valid") is not None:
                raise ValueError("Pass query mask OR token validity")
            kwargs["tactile_future_valid"] = self._query(touch_condition, tactile_query_mask)[1].any(-1)
        return super().forward(noisy_actions, action_timestep, touch_condition=touch_condition, **kwargs)

    def _prepare_tactile(
        self,
        noisy_actions,
        noisy_tactile,
        tactile_timestep,
        tactile_future_valid,
        tactile_maps,
        tactile_valid,
        touch_condition,
    ):
        history, history_valid, ids, _, spatial, sensors, _ = touch_condition
        _, possible = self._query(touch_condition)
        shape = possible.shape
        if noisy_tactile is None:
            noisy_tactile = noisy_actions.new_zeros(shape)
            tactile_future_valid = torch.zeros(shape[:2], dtype=torch.bool, device=noisy_actions.device)
            tactile_timestep = torch.zeros(shape[0], device=noisy_actions.device)
        elif tactile_timestep is None or tactile_future_valid is None:
            raise ValueError("Noisy future touch requires timestep and query validity")
        if (
            noisy_tactile.shape != shape
            or tactile_future_valid.dtype != torch.bool
            or tactile_future_valid.shape != shape[:2]
        ):
            raise ValueError("Future patch/query shape differs from regional layout")
        possible = possible & tactile_future_valid[..., None]
        tactile_future_valid = possible.any(-1)
        noisy_tactile = torch.where(possible, noisy_tactile, 0)
        if not torch.isfinite(noisy_tactile).all():
            raise ValueError("Nonfinite noise in requested tactile regions")
        future, time = self.touch_v2.prepare_future(noisy_tactile, tactile_timestep, ids, sensors, spatial)
        future = future.masked_fill(~tactile_future_valid[..., None], 0)
        return history, history_valid, future, time, tactile_future_valid

    def prepare_inputs(self, batch):
        samples = [batch] if isinstance(batch, dict) else batch
        if any(sample.get("training_modality") == "video_only" for sample in samples):
            raise ValueError("Regional TouchWAM currently requires video-action samples")
        s = self.tactile_spec
        expected = torch.arange(1, s["future_frames"] + 1) * s["future_stride"]
        for sample in samples:
            offsets = torch.as_tensor(sample.get("future_tactile_offsets", []))
            if not torch.equal(offsets.cpu(), expected) or sample.get("tactile_fps") != s["fps"]:
                raise ValueError("Dataset/model tactile offsets or fps differ")
        tactile = collate_regions(
            samples,
            map_size=s["map_size"],
            history=s["history"],
            future_frames=s["future_frames"],
            region_names=s["region_names"],
            sensor_names=s["sensor_names"],
        )
        result = DualSystemSelfAttnArchitecture.prepare_inputs(self, samples)
        result.update(tactile)
        return result

    def compute_loss(self, *, actions=None, lambda_video=1.0, lambda_action=1.0, **inputs):
        if actions is None or torch.as_tensor(inputs.pop("video_only_mask", False)).any():
            raise ValueError("Regional TouchWAM currently requires video-action targets")
        future_maps = torch.as_tensor(inputs.pop("future_tactile_maps"), device=self.device)
        future_valid = torch.as_tensor(inputs.pop("future_tactile_valid"), device=self.device)
        requested = inputs.pop("tactile_query_mask", None)
        condition = self._condition(
            inputs.pop("tactile_maps", None),
            inputs.pop("tactile_valid", None),
            actions.shape[0],
            region_ids=inputs.pop("tactile_region_ids", None),
            layout_present=inputs.pop("tactile_layout_present", None),
            spatial_valid=inputs.pop("tactile_spatial_valid", None),
            sensor_ids=inputs.pop("tactile_sensor_ids", None),
            palm_mode=inputs.pop("tactile_palm_mode", None),
            augment=self.training,
        )
        query, pixels = self._query(condition, requested)
        if (
            future_maps.shape[:3] != query.shape
            or future_valid.shape != query.shape
            or future_valid.dtype != torch.bool
        ):
            raise ValueError("Future target shape or validity differs from query layout")
        if future_maps.dtype == torch.uint8:
            future_maps = future_maps.to(self.dtype) / 255
        elif future_maps.is_floating_point():
            future_maps = future_maps.to(self.dtype)
        else:
            raise ValueError("Deform targets must be uint8 or float [0,1]")
        codec, s = self.touch_v2.codec, self.tactile_spec
        clean, labels = codec.encode(future_maps, future_valid & query, condition[4])
        if (query & ~labels).any():
            raise ValueError(
                "Requested future regions require complete real labels; filter incomplete windows, never use fake zeros"
            )
        if self.tactile_scheduler.timesteps is None:
            raise RuntimeError("Call init_training_schedulers before compute_loss")
        b = actions.shape[0]
        vs, acs, ts = self.video_scheduler, self.action_scheduler, self.tactile_scheduler
        lower = int(inputs.pop("min_timestep_boundary", 0) * len(vs.timesteps))
        upper = int(inputs.pop("max_timestep_boundary", 1) * len(vs.timesteps))
        vi, ai, ti = (
            torch.randint(lower, upper, (b,)),
            torch.randint(len(acs.timesteps), (b,)),
            torch.randint(len(ts.timesteps), (b,)),
        )

        def noise_sample(value, scheduler, index):
            value = value.to(device=self.device, dtype=self.dtype)
            sigma = scheduler.sigmas[index].to(value).reshape(b, *([1] * (value.ndim - 1)))
            noise = torch.randn_like(value)
            return (1 - sigma) * value + sigma * noise, noise - value, scheduler.timesteps[index].to(value)

        noisy_v, target_v, time_v = noise_sample(inputs["input_latents"], vs, vi)
        noisy_a, target_a, time_a = noise_sample(actions, acs, ai)
        noisy_t, target_t, time_t = noise_sample(clean, ts, ti)
        if inputs.get("first_frame_latents") is not None:
            ref = inputs["first_frame_latents"]
            noisy_v[:, :, : ref.shape[2]] = ref
        forward_inputs = dict(inputs)
        for key in ("action_is_pad", "video_is_pad", "input_latents"):
            forward_inputs.pop(key, None)
        proprio_mask = forward_inputs.pop("proprio_mask", None)
        if proprio_mask is not None:
            forward_inputs["_proprio_sample_mask"] = proprio_mask
        predictions = self(
            noisy_a,
            time_a,
            noisy_tactile=torch.where(pixels, noisy_t, 0),
            tactile_timestep=time_t,
            tactile_future_valid=pixels.any(-1),
            touch_condition=condition,
            latents=noisy_v,
            timestep=time_v,
            **forward_inputs,
        )
        loss_v = self._compute_video_loss(predictions[0], target_v, vi, inputs, self.device)
        loss_a = self._compute_action_loss(predictions[1], target_a, ai, acs, inputs, self.device)
        loss_t = regional_flow_loss(
            codec, predictions[2], target_t, clean, pixels, query.shape[2], s["foreground_weight"]
        )
        return dict(
            loss=lambda_video * loss_v + lambda_action * loss_a + s["loss_weight"] * loss_t,
            loss_video=(lambda_video * loss_v).detach(),
            loss_action=(lambda_action * loss_a).detach(),
            loss_tactile=(s["loss_weight"] * loss_t).detach(),
            loss_tactile_deform=loss_t.detach(),
        )

    def prepare_tactile_generation(self, maps, valid, **options):
        requested = options.pop("query_mask", None)
        condition = self._condition(maps, valid, 1, **options)
        query, pixels = self._query(condition, requested)
        if not self.tactile_spec["allow_missing_history"] and (query.any(1) & ~condition[6]).any():
            raise ValueError(
                "Requested region has no valid history; use a checkpoint trained for missing-history generation"
            )
        return condition, query, pixels.any(-1)

    def sample_tactile_noise(self, condition, token_valid, seed):
        ids = condition[2]
        codec, s = self.touch_v2.codec, self.tactile_spec
        shape = (ids.shape[0], s["future_frames"], ids.shape[1], codec.patches_per_site, codec.patch_dim)
        noise = torch.zeros(shape, device=self.device, dtype=self.dtype)
        # Stable per-region noise: optional palms/padding do not shift finger RNG.
        for b, row in enumerate(ids.tolist()):
            for slot, region in enumerate(row):
                if region >= 0:
                    generator = torch.Generator(device=self.device).manual_seed((seed + 1 + region * 1000003) % (2**63))
                    noise[b, :, slot] = torch.randn(
                        shape[1], shape[3], shape[4], device=self.device, dtype=self.dtype, generator=generator
                    )
        return noise.flatten(1, 3).masked_fill(~token_valid[..., None], 0)

    def decode_tactile_generation(self, tokens, future_valid, condition):
        ids, spatial = condition[2], condition[4]
        maps = self.touch_v2.codec.decode(tokens, ids.shape[1])
        maps = maps.masked_fill(~(future_valid[..., None, None, None] & spatial[:, None]), 0)
        keep = future_valid[0].any(0)
        selected_ids = ids[0, keep].tolist()
        return dict(
            future_tactile_maps=maps[0, :, keep].float().cpu().numpy(),
            future_tactile_valid=future_valid[0, :, keep].cpu().numpy(),
            tactile_region_names=[self.tactile_spec["region_names"][i] for i in selected_ids],
            tactile_region_ids=ids[0, keep].cpu().numpy(),
            tactile_history_available=condition[6][0, keep].cpu().numpy(),
            tactile_spatial_valid=spatial[0, keep].cpu().numpy(),
            tactile_representation="deform_gray",
            tactile_codec=self.tactile_spec["codec"],
            future_tactile_times_s=(
                torch.arange(1, self.tactile_spec["future_frames"] + 1)
                * self.tactile_spec["future_stride"]
                / self.tactile_spec["fps"]
            ).numpy(),
        )

    def load_legacy_touch_checkpoint(self, path):
        """Explicit initialization, not resume; audit every reused/reinitialized tensor."""
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(path, framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
        if meta.get("model_version") != "touchwam_v2" or meta.get("tactile_schema") != "sharpa_tactile_v1":
            raise RuntimeError("Expected a legacy touchwam_v2 geometry checkpoint")
        old, current = load_file(path), self.state_dict()
        prefix = self.new_weight_prefixes
        old_base = {k for k in old if not k.startswith(prefix)}
        base = {k for k in current if not k.startswith(prefix)}
        if old_base != base or any(old[k].shape != current[k].shape for k in base):
            raise RuntimeError("Legacy checkpoint V/A key or shape mismatch")
        reused = dict((k, old[k]) for k in base)
        compatible = ("touch_v2.history_encoder.", "touch_v2.blocks.", "touch_v2.time_mlp.")
        for key in current:
            if key.startswith(compatible) and key in old and old[key].shape == current[key].shape:
                reused[key] = old[key]
        missing, unexpected = self.load_state_dict(
            reused, strict=False, assign=any(p.is_meta for p in self.parameters())
        )
        if (
            set(missing) != set(current) - set(reused)
            or unexpected
            or any(t.is_meta for t in (*self.parameters(), *self.buffers()))
        ):
            raise RuntimeError("Legacy migration audit failed")
        with torch.no_grad():
            for block in self.touch_v2.blocks:
                block.video_gate.zero_()
                block.action_gate.zero_()
        self._apply_training_policy()
        return dict(
            source=path,
            reused=sorted(reused),
            initialized=sorted(set(current) - set(reused)),
            reset_gates=True,
            optimizer_resumed=False,
        )
