"""Optional-palm deformation generation using the existing V/A/T bridges."""

import json

import torch

from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture
from openwam.tactile.regions import CODEC, INTENSITY_CODEC, RGB_CODEC, REGION_NAMES, SCHEMA, collate_regions
from openwam.tactile.registry import (
    LEGACY_REGION_NAMES,
    PREVIOUS_REGION_NAMES,
    PREVIOUS_REGISTRY_SHA256,
    PREVIOUS_REGISTRY_VERSION,
    REGISTRY_SHA256,
    REGISTRY_VERSION,
    region_spec,
    validate_registry,
)

from .architecture import TouchWAMArchitecture
from .regional_modules import RegionalTactileFlowStream, globally_normalized_sum, regional_flow_loss


@register_architecture(
    "touchwam_regions",
    framework="dual_system",
    variant="touchwam_regions",
    note="Named deformation regions with optional palms and independent observation/query masks.",
)
class RegionalTouchWAMArchitecture(TouchWAMArchitecture):
    model_version = "touchwam_regions_v1"
    supports_video_only_training = True
    # prepare_inputs only collates touch packets and runs the frozen visual/text
    # encoders. Trainable tactile frontends run in compute_loss after the wait.
    supports_frozen_input_overlap = True

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
            rgb_enabled=False,
            rgb_map_size=128,
            rgb_patch_size=64,
            observation_encoder=None,
        )
        provided = dict(cfg.get("tactile", {}))
        if set(provided) - set(defaults):
            raise ValueError(f"Unknown regional tactile settings: {sorted(set(provided) - set(defaults))}")
        defaults.update(provided)
        s = self.tactile_spec = defaults
        if s["observation_encoder"] is None:
            s.pop("observation_encoder")
        else:
            from .heterogeneous import normalize_encoder_spec
            s["observation_encoder"] = normalize_encoder_spec(s["observation_encoder"], list(s["sensor_names"]))
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
            "rgb_map_size",
            "rgb_patch_size",
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
        validate_registry(s["region_names"])
        if s["codec"] not in (CODEC, INTENSITY_CODEC):
            raise ValueError("The default tactile codec must remain deform/intensity; enable the separate RGB route")
        if type(s["rgb_enabled"]) is not bool:
            raise ValueError("rgb_enabled must be boolean")
        if not s["rgb_enabled"]:
            # Keep old checkpoint metadata/specs exactly loadable when this
            # optional route is disabled, including explicitly passed false.
            for key in ("rgb_enabled", "rgb_map_size", "rgb_patch_size"):
                s.pop(key)
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
        # Validate serialization at construction, before any optimizer steps.
        # The same property is used by actual safetensors saving/loading.
        self.checkpoint_metadata
        self.touch_v2 = RegionalTactileFlowStream(s, self.video_backbone.dim, self.action_backbone.dim)

    # training.touchwam_p1 switches (OpenWAMTrainer._configure_touchwam_p1); defaults = reference path.
    _validate_inputs = True
    _encoder_activity_mask = False
    _pinned_tactile_h2d = False

    def configure_training_fastpath(self, *, validate_inputs=None, encoder_activity_mask=None, pinned_tactile_h2d=None):
        """validate_inputs=False skips only checks that raise on bad input and read GPU values
        back to the host; every value that enters the forward/loss is computed identically."""
        if validate_inputs is not None:
            self._validate_inputs = bool(validate_inputs)
            if hasattr(self.touch_v2, "observation_encoder"):
                self.touch_v2.observation_encoder._validate_inputs = bool(validate_inputs)
        if encoder_activity_mask is not None:
            self._encoder_activity_mask = bool(encoder_activity_mask)
        if pinned_tactile_h2d is not None:
            self._pinned_tactile_h2d = bool(pinned_tactile_h2d)

    def _apply_training_policy(self):
        super()._apply_training_policy()
        if hasattr(self.touch_v2, "observation_encoder"):
            self.touch_v2.observation_encoder.apply_freeze_policy()

    def load_openwam_checkpoint(self, path):
        super().load_openwam_checkpoint(path)
        if hasattr(self.touch_v2, "observation_encoder"):
            self.ftp1_initialization_provenance = self.touch_v2.observation_encoder.load_pretrained()
            self.initialization_provenance = dict(openwam_checkpoint=str(path),
                openwam_tensors_preserved=True, tactile_observation_encoder=self.ftp1_initialization_provenance)
        self._apply_training_policy()

    def _codec_name(self, name=None):
        name = self.tactile_spec["codec"] if name is None else name
        # Already normalized legacy deform remains compatible with the
        # source-normalized intensity route. RGB is always explicitly selected.
        if name == CODEC and self.tactile_spec["codec"] == INTENSITY_CODEC:
            name = INTENSITY_CODEC
        self.touch_v2.get_codec(name)
        return name

    def _condition_codec(self, condition):
        return self._codec_name(condition[7] if len(condition) == 8 else None)

    def _predict_tactile(self, future, condition):
        return self.touch_v2.predict(future, self._condition_codec(condition))

    @property
    def checkpoint_metadata(self):
        metadata = dict(
            model_version=self.model_version,
            tactile_schema=SCHEMA,
            tactile_codec=self.tactile_spec["codec"],
            tactile_spec=json.dumps(self.tactile_spec, sort_keys=True),
        )
        if tuple(self.tactile_spec["region_names"]) == REGION_NAMES:
            metadata.update(tactile_region_registry=REGISTRY_VERSION, tactile_region_registry_sha256=REGISTRY_SHA256)
        return metadata

    @property
    def tactile_input_contract(self):
        s = self.tactile_spec
        contract = dict(
            model_version=self.model_version,
            schema=SCHEMA,
            codec=s["codec"],
            region_names=s["region_names"],
            region_registry=REGISTRY_VERSION if tuple(s["region_names"]) == REGION_NAMES else "legacy_12",
            region_registry_sha256=REGISTRY_SHA256 if tuple(s["region_names"]) == REGION_NAMES else None,
            region_identity="unique region_id; compose native sensors before model input",
            single_arm_gripper_side="right",
            sensor_names=s["sensor_names"],
            representation="contact_intensity" if s["codec"] == INTENSITY_CODEC else "deform_gray",
            channels=["contact_intensity" if s["codec"] == INTENSITY_CODEC else "deform_intensity"],
            intensity_units="source-normalized, not calibrated force",
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
        if s.get("rgb_enabled", False):
            contract["supported_codecs"] = [s["codec"], RGB_CODEC]
            contract["rgb"] = dict(codec=RGB_CODEC, representation="optical_rgb", channels=["red", "green", "blue"],
                                   map_size=s["rgb_map_size"], patch_size=s["rgb_patch_size"],
                                   units="image channel values [0,1], not force or deformation",
                                   tokens_per_region=self.touch_v2.rgb_codec.patches_per_site,
                                   routing="one codec per microbatch; independent image encoder/output head")
        if s.get("observation_encoder"):
            contract["observation_encoder"] = s["observation_encoder"]
            contract["observation_input"] = "tactile_observation_maps + tactile_observation_spatial_valid + tactile_encoder_ids"
            contract["observation_tokens_per_region"] = 4
            contract["history_and_prediction_resolution"] = "independent; original history never upsampled from prediction cache"
        return contract

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
        tactile_codec=None,
        observation_maps=None,
        observation_spatial_valid=None,
        encoder_ids=None,
    ):
        s, device = self.tactile_spec, self.device
        codec_name = self._codec_name(tactile_codec)
        codec = self.touch_v2.get_codec(codec_name)

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
            region_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            layout_present = torch.zeros_like(region_ids, dtype=torch.bool)
        if layout_present is None:
            raise ValueError("Explicit tactile_layout_present is required")
        ids = tensor(region_ids, 2, torch.long, "region IDs")
        present = tensor(layout_present, 2, torch.bool, "layout presence")
        if ids.shape != present.shape:
            raise ValueError("Invalid region layout")
        if self._validate_inputs:
            if ((ids < -1) | (ids >= len(s["region_names"]))).any():
                raise ValueError("Invalid region layout")
            if (present & (ids < 0)).any():
                raise ValueError("Padding IDs cannot be present")
            for row in ids:
                regions = row[row >= 0]
                if regions.unique().numel() != regions.numel():
                    raise ValueError("Duplicate region in one sample; compose native sensors first")
        palm_ids = [i for i, name in enumerate(s["region_names"]) if "palm" in region_spec(name).groups]
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
        shape = (batch_size, n, 1, codec.map_size, codec.map_size)
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
        if sensors.shape != ids.shape or (
                self._validate_inputs and ((sensors < 0) | (sensors >= len(s["sensor_names"]))).any()):
            raise ValueError("Unknown sensor identity")
        if maps is None:
            if valid is not None:
                raise ValueError("Observation validity requires maps")
            maps = torch.zeros(
                batch_size, s["history"], n, codec.channels, codec.map_size, codec.map_size,
                device=device, dtype=self.dtype
            )
            valid = torch.zeros(batch_size, s["history"], n, device=device, dtype=torch.bool)
        elif valid is None:
            raise ValueError("Maps require explicit observation validity")
        maps = torch.as_tensor(maps, device=device)
        if maps.dtype == torch.uint8:
            maps = maps.to(self.dtype) / 255
        elif not maps.is_floating_point():
            raise ValueError("Tactile maps must be uint8 or float [0,1]")
        else:
            maps = maps.to(self.dtype)
        if maps.ndim == 5 and batch_size == 1:
            maps = maps.unsqueeze(0)
        if maps.shape != (batch_size, s["history"], n, codec.channels, codec.map_size, codec.map_size):
            raise ValueError("History map shape differs from layout/configuration")
        valid = tensor(valid, 3, torch.bool, "observation validity")
        if valid.shape != maps.shape[:3]:
            raise ValueError("Observation mask shape differs from history")
        valid = valid & active[:, None]
        if augment and s["history_dropout"]:
            # Drop entire region histories, so completely stale sensors occur in training.
            valid &= ~(torch.rand(batch_size, 1, n, device=device) < s["history_dropout"])
        observation_kwargs = {}
        if s.get("observation_encoder") and n:
            if observation_maps is None or observation_spatial_valid is None or encoder_ids is None:
                raise ValueError("Heterogeneous input requires observation maps, ROI and explicit encoder IDs")
            encoder_ids = tensor(encoder_ids, 2, torch.long, "encoder IDs")
            if encoder_ids.shape != ids.shape:
                raise ValueError("Encoder IDs differ from regional layout")
            observation_maps = torch.as_tensor(observation_maps, device=device)
            if observation_maps.dtype == torch.uint8:
                observation_maps = observation_maps.to(self.dtype) / 255
            elif observation_maps.is_floating_point():
                observation_maps = observation_maps.to(self.dtype)
            else:
                raise ValueError("Observation maps must be uint8 or float [0,1]")
            if observation_maps.ndim == 5 and batch_size == 1:
                observation_maps = observation_maps.unsqueeze(0)
            if observation_maps.ndim != 6 or observation_maps.shape[:4] != maps.shape[:4]:
                raise ValueError("Observation history layout/channels differ from future codec")
            observation_spatial = tensor(observation_spatial_valid, 5, torch.bool, "observation spatial ROI")
            observation_spatial = observation_spatial & active[..., None, None, None]
            observation_kwargs = dict(observation_maps=observation_maps, observation_spatial=observation_spatial, encoder_ids=encoder_ids)
        elif not s.get("observation_encoder") and any(x is not None for x in (observation_maps, observation_spatial_valid, encoder_ids)):
            raise ValueError("Observation encoder inputs require an enabled heterogeneous encoder")
        history, mask, available = self.touch_v2.encode_history(maps, valid, ids, sensors, spatial, codec_name, **observation_kwargs)
        if s.get("observation_encoder") and self._encoder_activity_mask:
            # Same set as encoder_ids[available].unique(), as a fixed-size bitmask: no host sync.
            count = len(s["observation_encoder"]["sensors"])
            activity = torch.zeros(count, dtype=torch.long, device=device)
            if n:
                keep = available & (encoder_ids >= 0) & (encoder_ids < count)
                activity.scatter_reduce_(0, encoder_ids.clamp(0, count - 1).flatten(), keep.long().flatten(),
                                         reduce="amax")
            self._observed_encoder_mask = activity
        elif s.get("observation_encoder"):
            self._observed_encoder_ids = (encoder_ids[available].unique().detach() if n
                                          else ids.new_empty(0))
        condition = (history, mask, ids, active, spatial, sensors, available)
        return (*condition, codec_name) if s.get("rgb_enabled", False) else condition

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
        return requested, self.touch_v2.get_codec(self._condition_codec(condition)).pixel_mask(requested, spatial)

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
        tactile_codec=None,
        tactile_observation_maps=None,
        tactile_observation_spatial_valid=None,
        tactile_encoder_ids=None,
        tactile_query_mask=None,
        touch_condition=None,
        **kwargs,
    ):
        if "tactile_surface_ids" in kwargs:
            raise ValueError("surface_id was removed; use unique regional maps")
        if "future_tactile_maps" in kwargs or "future_tactile_valid" in kwargs:
            raise ValueError("Clean future tactile targets must not enter forward; use compute_loss")
        if touch_condition is None:
            touch_condition = self._condition(
                tactile_maps,
                tactile_valid,
                (noisy_actions if noisy_actions is not None else kwargs["latents"]).shape[0],
                region_ids=tactile_region_ids,
                layout_present=tactile_layout_present,
                spatial_valid=tactile_spatial_valid,
                sensor_ids=tactile_sensor_ids,
                palm_mode=tactile_palm_mode,
                tactile_codec=tactile_codec,
                observation_maps=tactile_observation_maps,
                observation_spatial_valid=tactile_observation_spatial_valid,
                encoder_ids=tactile_encoder_ids,
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
                tactile_codec,
                tactile_observation_maps,
                tactile_observation_spatial_valid,
                tactile_encoder_ids,
            )
        ):
            raise ValueError("Pass raw history/layout OR touch_condition")
        if tactile_query_mask is not None:
            if kwargs.get("tactile_future_valid") is not None:
                raise ValueError("Pass query mask OR token validity")
            kwargs["tactile_future_valid"] = self._query(touch_condition, tactile_query_mask)[1].any(-1)
        if touch_condition[2].shape[1] == 0:
            # Exact OpenWAM V/VA path: no tactile attention, embeddings or writeback.
            for key in ("noisy_tactile", "tactile_timestep", "tactile_future_valid"):
                value = kwargs.pop(key, None)
                if key != "tactile_timestep" and value is not None and value.numel():
                    raise ValueError("Nonempty future tactile tokens require a regional layout")
            video, action = DualSystemSelfAttnArchitecture.forward(self, noisy_actions, action_timestep, **kwargs)
            codec = self.touch_v2.get_codec(self._condition_codec(touch_condition))
            return video, action, video.new_empty(video.shape[0], 0, codec.patch_dim)
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
        history, history_valid, ids, _, spatial, sensors, _ = touch_condition[:7]
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
        if self._validate_inputs and not torch.isfinite(noisy_tactile).all():
            raise ValueError("Nonfinite noise in requested tactile regions")
        future, time = self.touch_v2.prepare_future(
            noisy_tactile, tactile_timestep, ids, sensors, spatial, self._condition_codec(touch_condition)
        )
        future = future.masked_fill(~tactile_future_valid[..., None], 0)
        return history, history_valid, future, time, tactile_future_valid

    def prepare_inputs(self, batch):
        samples = [batch] if isinstance(batch, dict) else list(batch)
        if not samples:
            raise ValueError("Cannot prepare an empty batch")
        s = self.tactile_spec
        modalities = [sample.get("training_modality", "video_action") for sample in samples]
        allowed = {"video_only", "video_action", "video_tactile", "video_action_tactile"}
        if set(modalities) - allowed:
            raise ValueError(f"Unknown training modalities: {set(modalities) - allowed}")
        with_action = [m in ("video_action", "video_action_tactile") for m in modalities]
        if any(with_action) and not all(with_action):
            raise ValueError("Use globally aligned microbatches; mixed action/no-action batches are not supported")
        codec_names = {self._codec_name(sample.get("tactile_codec")) for sample in samples
                       if sample.get("tactile_region_names")}
        if len(codec_names) > 1:
            raise ValueError("Use homogeneous tactile codecs per microbatch; RGB and intensity cannot be silently mixed")
        codec_name = next(iter(codec_names), self.tactile_spec["codec"])
        codec = self.touch_v2.get_codec(codec_name)
        expected_future = torch.arange(1, s["future_frames"] + 1, dtype=torch.float64) * s["future_stride"] / s["fps"]
        expected_history = torch.arange(1 - s["history"], 1, dtype=torch.float64) / s["fps"]
        base_samples = []
        for sample, modality, has_action in zip(samples, modalities, with_action):
            has_regions = bool(sample.get("tactile_region_names"))
            if modality in ("video_tactile", "video_action_tactile") and not has_regions:
                raise ValueError("A tactile training modality requires an explicit regional layout")
            if has_regions:
                if "future_tactile_times_s" in sample:
                    times = torch.as_tensor(sample["future_tactile_times_s"], dtype=torch.float64).cpu()
                    if times.shape != expected_future.shape or not torch.allclose(times, expected_future, atol=1e-6, rtol=0):
                        raise ValueError("Dataset/model future tactile times differ")
                else:
                    offsets = torch.as_tensor(sample.get("future_tactile_offsets", []), dtype=torch.float64).cpu()
                    fps = sample.get("tactile_fps")
                    if fps is None or fps <= 0 or offsets.shape != expected_future.shape or not torch.allclose(
                        offsets / fps, expected_future, atol=1e-6, rtol=0
                    ):
                        raise ValueError("Dataset/model tactile offsets or fps differ")
                if "tactile_history_times_s" in sample:
                    times = torch.as_tensor(sample["tactile_history_times_s"], dtype=torch.float64).cpu()
                    if times.shape != expected_history.shape or not torch.allclose(times, expected_history, atol=1e-6, rtol=0):
                        raise ValueError("Dataset/model tactile history times differ; resample causally in the reader")
            base = dict(sample)
            base["training_modality"] = "video_action" if has_action else "video_only"
            if not has_action:
                # Older V readers contain all-masked placeholders. Remove them
                # before preprocessing; no dummy action/proprio enters forward.
                for key in ("action_mask", "proprio_mask"):
                    if key in base and torch.as_tensor(base[key]).any():
                        raise ValueError(f"{modality} cannot contain a valid {key}")
                    base[key] = torch.zeros(1, dtype=torch.bool)
                base.pop("action", None)
                base.pop("proprio", None)
            base_samples.append(base)
        observation_options = {}
        if s.get("observation_encoder"):
            from openwam.tactile.regions import observation_batch_shape
            observation_options = dict(observation_map_size=observation_batch_shape(samples, s["observation_encoder"]),
                                       encoder_names=self.touch_v2.observation_encoder.encoder_names)
        tactile = collate_regions(
            samples, map_size=codec.map_size, history=s["history"], future_frames=s["future_frames"],
            region_names=s["region_names"], sensor_names=s["sensor_names"], codec=codec_name,
            **observation_options,
        )
        if self._pinned_tactile_h2d and torch.cuda.is_available():
            # One stream-ordered copy per tensor instead of a blocking (stream-syncing) copy
            # each time _condition/compute_loss moves them. Values are unchanged.
            tactile = {k: v.pin_memory().to(self.device, non_blocking=True)
                       if torch.is_tensor(v) and v.device.type == "cpu" else v for k, v in tactile.items()}
        result = DualSystemSelfAttnArchitecture.prepare_inputs(self, base_samples)
        result.update(tactile)
        result["tactile_codec"] = codec_name
        return result

    def compute_loss(self, *, actions=None, lambda_video=1.0, lambda_action=1.0, **inputs):
        """Real V/VA/VT/VAT paths with independently masked future supervision."""
        if "tactile_surface_ids" in inputs:
            raise ValueError("surface_id was removed; use unique regional maps")
        b = inputs["input_latents"].shape[0]
        no_action = inputs.pop("video_only_mask", None)
        if no_action is not None:
            # Only read on the host: keep it where the batch produced it (no copy + sync).
            no_action = (torch.as_tensor(no_action, device=self.device) if self._validate_inputs
                         else torch.as_tensor(no_action))
            if no_action.dtype != torch.bool or no_action.shape != (b,):
                raise ValueError("video_only_mask must be bool [B]")
            if no_action.any() and not no_action.all():
                raise ValueError("Use globally aligned microbatches; mixed action/no-action batches are not supported")
            if no_action.all() and self._validate_inputs:
                for key, enabled in (("action_is_pad", True), ("proprio_mask", False)):
                    value = inputs.get(key)
                    if value is not None and not (torch.as_tensor(value) == enabled).all():
                        raise ValueError(f"No-action samples contain valid {key}")
            if no_action.all():
                actions = None
        if actions is None:
            inputs.pop("proprio", None)
            inputs.pop("proprio_mask", None)
            inputs["_strict_video_only"] = True
        future_maps = inputs.pop("future_tactile_maps", None)
        future_valid = inputs.pop("future_tactile_valid", None)
        requested = inputs.pop("tactile_query_mask", None)
        condition = self._condition(
            inputs.pop("tactile_maps", None), inputs.pop("tactile_valid", None), b,
            region_ids=inputs.pop("tactile_region_ids", None),
            layout_present=inputs.pop("tactile_layout_present", None),
            spatial_valid=inputs.pop("tactile_spatial_valid", None),
            sensor_ids=inputs.pop("tactile_sensor_ids", None),
            palm_mode=inputs.pop("tactile_palm_mode", None), augment=self.training,
            tactile_codec=inputs.pop("tactile_codec", None),
            observation_maps=inputs.pop("tactile_observation_maps", None),
            observation_spatial_valid=inputs.pop("tactile_observation_spatial_valid", None),
            encoder_ids=inputs.pop("tactile_encoder_ids", None),
        )
        query, query_pixels = self._query(condition, requested)
        codec_name = self._condition_codec(condition)
        codec, s = self.touch_v2.get_codec(codec_name), self.tactile_spec
        if (future_maps is None) != (future_valid is None):
            raise ValueError("Future maps and validity must be provided together")
        if future_maps is None:
            future_maps = torch.zeros(*query.shape, codec.channels, codec.map_size, codec.map_size,
                                      device=self.device, dtype=self.dtype)
            future_valid = torch.zeros_like(query)
        else:
            future_maps = torch.as_tensor(future_maps, device=self.device)
            future_valid = torch.as_tensor(future_valid, device=self.device)
        if future_maps.shape[:3] != query.shape or future_valid.shape != query.shape or future_valid.dtype != torch.bool:
            raise ValueError("Future target shape or validity differs from query layout")
        if future_maps.dtype == torch.uint8:
            future_maps = future_maps.to(self.dtype) / 255
        elif future_maps.is_floating_point():
            future_maps = future_maps.to(self.dtype)
        else:
            raise ValueError("Tactile targets must be uint8 or float [0,1]")
        clean, labels = codec.encode(future_maps, future_valid & query, condition[4])
        label_pixels = codec.pixel_mask(labels, condition[4])
        if self.tactile_scheduler.timesteps is None:
            raise RuntimeError("Call init_training_schedulers before compute_loss")
        vs, acs, ts = self.video_scheduler, self.action_scheduler, self.tactile_scheduler
        lower = int(inputs.pop("min_timestep_boundary", 0) * len(vs.timesteps))
        upper = int(inputs.pop("max_timestep_boundary", 1) * len(vs.timesteps))
        if not 0 <= lower < upper <= len(vs.timesteps):
            raise ValueError("Invalid video timestep boundaries")
        vi = torch.randint(lower, upper, (b,))

        def noise_sample(value, scheduler, index):
            value = value.to(device=self.device, dtype=self.dtype)
            sigma = scheduler.sigmas[index].to(value).reshape(b, *([1] * (value.ndim - 1)))
            noise = torch.randn_like(value)
            return (1 - sigma) * value + sigma * noise, noise - value, scheduler.timesteps[index].to(value), noise

        noisy_v, target_v, time_v, _ = noise_sample(inputs["input_latents"], vs, vi)
        noisy_a = target_a = time_a = ai = None
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=self.dtype)
            action_valid = self._action_valid_mask(actions, inputs.get("action_is_pad"))
            if self._validate_inputs and not torch.isfinite(actions.masked_select(action_valid)).all():
                raise ValueError("Nonfinite valid action labels")
            actions = torch.where(action_valid, actions, 0)
            ai = torch.randint(len(acs.timesteps), (b,))
            noisy_a, target_a, time_a, _ = noise_sample(actions, acs, ai)
            # Inactive slots follow sigma * independent noise, matching the
            # inference analytic-noise path. They receive no action supervision.
            # Sanitization above prevents placeholder values from entering it.
        ti = torch.randint(len(ts.timesteps), (b,))
        noisy_t, target_t, time_t, noise_t = noise_sample(clean, ts, ti)
        # Queries are known from layout, not target validity. Unknown futures
        # receive pure noise and no loss; clean placeholder zeros never leak.
        noisy_t = torch.where(label_pixels, noisy_t, noise_t)
        noisy_t = torch.where(query_pixels, noisy_t, 0)
        if inputs.get("first_frame_latents") is not None:
            ref = inputs["first_frame_latents"]
            noisy_v[:, :, : ref.shape[2]] = ref
        forward_inputs = dict(inputs)
        for key in ("action_is_pad", "video_is_pad", "input_latents", "latents"):
            forward_inputs.pop(key, None)
        proprio_mask = forward_inputs.pop("proprio_mask", None)
        if proprio_mask is not None:
            forward_inputs["_proprio_sample_mask"] = proprio_mask
        predictions = self(
            noisy_a, time_a, noisy_tactile=noisy_t, tactile_timestep=time_t,
            tactile_future_valid=query_pixels.any(-1), touch_condition=condition,
            latents=noisy_v, timestep=time_v, **forward_inputs,
        )
        loss_v = self._compute_video_loss(predictions[0], target_v, vi, inputs, self.device)
        loss_a = self._regional_action_loss(predictions[1], target_a, ai, acs, inputs)
        loss_t = regional_flow_loss(codec, predictions[2], target_t, clean, label_pixels, query.shape[2], s["foreground_weight"])
        return dict(
            loss=lambda_video * loss_v + lambda_action * loss_a + s["loss_weight"] * loss_t,
            loss_video=(lambda_video * loss_v).detach(), loss_action=(lambda_action * loss_a).detach(),
            loss_tactile=(s["loss_weight"] * loss_t).detach(),
            loss_tactile_deform=(loss_t * (codec_name != RGB_CODEC)).detach(),
            loss_tactile_intensity=(loss_t * (codec_name != RGB_CODEC)).detach(),
            loss_tactile_rgb=(loss_t * (codec_name == RGB_CODEC)).detach(),
            tactile_labeled_regions=labels.any(1).sum().detach(), tactile_labeled_frames=labels.sum().detach(),
            **({"observation_encoder_active_mask": self._observed_encoder_mask}
               if s.get("observation_encoder") and self._encoder_activity_mask else
               {"observation_encoder_active_ids": self._observed_encoder_ids}
               if s.get("observation_encoder") else {}),
        )

    def _compute_video_loss(self, prediction, target, indices, inputs, device):
        # Base video loss is a batch mean of correctly masked sample means.
        # Remove dilution by samples with no supervised future visual frame,
        # using the same global labeled-sample denominator as A and T.
        loss = super()._compute_video_loss(prediction, target, indices, inputs, device)
        pad = inputs.get("video_is_pad")
        count = prediction.new_tensor(prediction.shape[0], dtype=torch.float32) if pad is None else (
            ~torch.as_tensor(pad, device=device, dtype=torch.bool)
        ).flatten(1).any(1).sum()
        return globally_normalized_sum(loss * prediction.shape[0], count)

    @staticmethod
    def _action_valid_mask(actions, pad):
        if pad is None:
            return torch.ones_like(actions, dtype=torch.bool)
        pad = torch.as_tensor(pad, device=actions.device)
        if pad.dtype != torch.bool:
            raise ValueError("Action padding must be boolean")
        if pad.shape == actions.shape[:2]:
            pad = pad[..., None].expand_as(actions)
        if pad.shape != actions.shape:
            raise ValueError("Action padding must be [B,T] or [B,T,D]")
        return ~pad

    def _regional_action_loss(self, prediction, target, indices, scheduler, inputs):
        # Always participate, even on ranks with zero action labels.
        if prediction is None:
            zero = inputs["input_latents"].new_zeros((), dtype=torch.float32)
            return globally_normalized_sum(zero, zero)
        valid = self._action_valid_mask(target, inputs.get("action_is_pad"))
        error = torch.where(valid, prediction.float() - target.float(), 0).square()
        count = valid.flatten(1).sum(1)
        per_sample = error.flatten(1).sum(1) / count.clamp_min(1)
        weights = scheduler.training_weight(indices).to(per_sample)
        return globally_normalized_sum((per_sample * weights).sum(), (count > 0).sum())

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
        codec, s = self.touch_v2.get_codec(self._condition_codec(condition)), self.tactile_spec
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
        codec_name = self._condition_codec(condition)
        codec = self.touch_v2.get_codec(codec_name)
        maps = codec.decode(tokens, ids.shape[1]) if ids.shape[1] else tokens.new_empty(
            ids.shape[0], self.tactile_spec["future_frames"], 0, codec.channels, codec.map_size, codec.map_size,
        )
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
            tactile_representation=("optical_rgb" if codec_name == RGB_CODEC else
                                    "contact_intensity" if codec_name == INTENSITY_CODEC else "deform_gray"),
            tactile_codec=codec_name,
            future_tactile_times_s=(
                torch.arange(1, self.tactile_spec["future_frames"] + 1)
                * self.tactile_spec["future_stride"]
                / self.tactile_spec["fps"]
            ).numpy(),
        )

    def load_regional_registry_checkpoint(self, path):
        """Explicit 12/64 -> 67 warm-start; old unsided gripper rows become right.

        v1 nonzero instance codes are removed, so this is not exact resume for
        models trained with repeated regions. Recompose data and start a new run.
        """
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(path, framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
        if meta.get("model_version") != self.model_version or meta.get("tactile_schema") != SCHEMA:
            raise RuntimeError("Expected a regional deform checkpoint")
        old_spec = json.loads(meta.get("tactile_spec", "{}"))
        names = tuple(old_spec.get("region_names", ()))
        if names not in (LEGACY_REGION_NAMES, PREVIOUS_REGION_NAMES):
            raise RuntimeError("Registry migration requires a known 12/64-region source")
        if names == PREVIOUS_REGION_NAMES and (
            meta.get("tactile_region_registry") != PREVIOUS_REGISTRY_VERSION
            or meta.get("tactile_region_registry_sha256") != PREVIOUS_REGISTRY_SHA256
        ):
            raise RuntimeError("Source registry metadata/hash mismatch")
        expected_spec = dict(self.tactile_spec, region_names=list(names))
        if (
            tuple(self.tactile_spec["region_names"]) != REGION_NAMES
            or old_spec != expected_spec
            or meta.get("tactile_codec") != self.tactile_spec["codec"]
        ):
            raise RuntimeError("Registry migration requires identical settings except 12/64 -> 67 region_names")
        old, current = load_file(path), self.state_dict()
        key = "touch_v2.region_embedding.weight"
        if old.keys() != current.keys() or any(old[k].shape != current[k].shape for k in current if k != key):
            raise RuntimeError("Registry migration tensor keys/shapes differ")
        if old[key].shape != (len(names), self.tactile_spec["width"]) or current[key].is_meta:
            raise RuntimeError("Registry migration requires materialized new region embeddings")
        expanded = current[key].detach().clone().to(old[key])
        expanded[: len(names)] = old[key]
        old[key] = expanded
        self.load_state_dict(old, strict=True, assign=any(p.is_meta for p in self.parameters()))
        if any(t.is_meta for t in (*self.parameters(), *self.buffers())):
            raise RuntimeError("Registry migration left meta tensors")
        self._apply_training_policy()
        return dict(
            source=path,
            registry=REGISTRY_VERSION,
            registry_sha256=REGISTRY_SHA256,
            reused=sorted(k for k in current if k != key),
            copied_region_rows=list(range(len(names))),
            initialized_region_rows=list(range(len(names), len(REGION_NAMES))),
            source_registry=meta.get("tactile_region_registry", "legacy_12"),
            gripper_policy="single arm maps to right; v1 rows 58:61 become right gripper" if len(names) == 64 else None,
            requires_region_composition=True,
            reset_gates=False,
            optimizer_resumed=False,
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
