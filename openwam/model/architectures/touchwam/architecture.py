"""TouchWAM V2: compact visual/action/tactile joint flow model.

Alpha's VA attention and state-dict paths are preserved. Six small tactile
blocks communicate through independent, zero-initialized residual bridges.
"""

from __future__ import annotations

import json
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.registry import register_architecture

from .modules import TactileFlowStream, tactile_flow_loss


@register_architecture(
    "touchwam_v2",
    framework="dual_system",
    variant="touchwam_v2",
    note="Compact third generative tactile stream with gated VA residual bridges (Wan).",
)
class TouchWAMArchitecture(DualSystemSelfAttnArchitecture):
    model_version = "touchwam_v2"
    new_weight_prefixes = ("touch_v2.",)
    supports_video_only_training = False
    preprocess_modules = (
        "video_backbone.text_encoder",
        "video_backbone.reason1",
        "video_backbone.vae",
        "video_backbone.image_encoder",
        "video_backbone.video_encoder",
    )

    def __init__(self, cfg=None):
        if cfg is None:
            raise ValueError("TouchWAM V2 requires a config")
        self.training_mode = cfg.get("training_mode", "joint_finetune")
        if self.training_mode not in ("new_only", "action_finetune", "joint_finetune"):
            raise ValueError("V2 training_mode must be new_only, action_finetune or joint_finetune")
        super().__init__(cfg)
        defaults = dict(
            history=4,
            map_size=32,
            patch_size=16,
            distance_scale_m=0.015,
            width=256,
            heads=8,
            layers=6,
            history_layers=2,
            bridge_layers=None,
            future_frames=8,
            future_stride=2,
            fps=20,
            loss_weight=1.0,
            contact_positive_weight=5.0,
            distance_background_weight=0.1,
        )
        spec = dict(cfg.get("tactile", {}))
        if set(spec) - set(defaults):
            raise ValueError(f"Unknown TouchWAM V2 settings: {sorted(set(spec) - set(defaults))}")
        defaults.update(spec)
        self.tactile_spec = defaults
        s = self.tactile_spec
        integers = (
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
        )
        if any(not isinstance(s[k], int) or isinstance(s[k], bool) or s[k] < 1 for k in integers):
            raise ValueError("Tactile dimensions, frame counts and strides must be positive integers")
        if s["width"] % s["heads"] or s["width"] % 2 or s["loss_weight"] < 0:
            raise ValueError("Invalid tactile width/heads/loss weight")
        if s["contact_positive_weight"] <= 0 or s["distance_background_weight"] < 0:
            raise ValueError("Invalid tactile pixel loss weights")
        depth = self.action_backbone.num_layers
        if s["bridge_layers"] is None:
            s["bridge_layers"] = [(i + 1) * depth // s["layers"] - 1 for i in range(s["layers"])]
        else:
            s["bridge_layers"] = list(s["bridge_layers"])
        layers = s["bridge_layers"]
        if (
            len(layers) != s["layers"]
            or layers != sorted(set(layers))
            or any(not isinstance(i, int) or i < 0 or i >= depth for i in layers)
        ):
            raise ValueError("bridge_layers must contain layers unique, ordered, existing VA layer indices")
        self.touch_v2 = TactileFlowStream(s, self.video_backbone.dim, self.action_backbone.dim)
        self.tactile_scheduler = ActionScheduler()
        self._apply_training_policy()

    @property
    def checkpoint_metadata(self):
        return dict(
            model_version=self.model_version,
            tactile_schema="sharpa_tactile_v1",
            tactile_codec="normalized_patches_v1",
            tactile_spec=json.dumps(self.tactile_spec, sort_keys=True),
        )

    @property
    def tactile_input_contract(self):
        from openwam.tactile.sharpa import SCHEMA, SITE_NAMES

        s = self.tactile_spec
        return dict(
            model_version=self.model_version,
            schema=SCHEMA,
            site_names=list(SITE_NAMES),
            history=s["history"],
            map_size=s["map_size"],
            fps=s["fps"],
            channels=["distance_along_normal_m", "contact_mask"],
            missing_sensor_mask="tactile_valid",
            missing_modality="bypass",
            future_frames=s["future_frames"],
            future_stride=s["future_stride"],
            future_contact_output="probability",
            codec="normalized_patches_v1",
        )

    def _apply_training_policy(self):
        for name, param in self.named_parameters():
            train = name.startswith(self.new_weight_prefixes)
            if self.training_mode == "action_finetune":
                train |= name.startswith(("action_backbone.", "proprio_encoder."))
            elif self.training_mode == "joint_finetune":
                train = True
            if name.startswith(tuple(prefix + "." for prefix in self.preprocess_modules)):
                train = False
            param.requires_grad_(train)

    def freeze_modules(self, names):
        if set(names) - set(self.preprocess_modules):
            raise ValueError("V2 uses training_mode for backbone freezing; freeze: may only name preprocess encoders")
        result = super().freeze_modules(names)
        self._apply_training_policy()
        return result

    def set_dtype_device(self, dtype, device):
        super().set_dtype_device(dtype, device)
        self.touch_v2.to(dtype=dtype, device=device)

    def init_training_schedulers(self, num_timesteps=1000):
        super().init_training_schedulers(num_timesteps)
        self.tactile_scheduler.set_timesteps(num_timesteps, training=True)

    def apply_compile_optimizations(self, compile_cfg):
        # The parent's compiled driver has only two streams: never select it.
        from openwam.model.compile_options import compile_enabled

        if compile_enabled(compile_cfg, default=False, strict=True):
            raise ValueError("TouchWAM V2 currently requires compile.enabled=false")

    @torch.no_grad()
    def training_diagnostics(self):
        return dict(
            model_version=self.model_version,
            training_mode=self.training_mode,
            touch_video_gates=[float(b.video_gate.tanh()) for b in self.touch_v2.blocks],
            touch_action_gates=[float(b.action_gate.tanh()) for b in self.touch_v2.blocks],
        )

    def _history(self, maps, valid, batch_size):
        s = self.tactile_spec
        if maps is None:
            if valid is not None:
                raise ValueError("tactile_valid requires tactile_maps")
            maps = torch.zeros(
                batch_size, s["history"], 10, 2, s["map_size"], s["map_size"], device=self.device, dtype=self.dtype
            )
            valid = torch.zeros(batch_size, s["history"], 10, device=self.device, dtype=torch.bool)
        elif valid is None:
            raise ValueError("tactile_maps requires tactile_valid")
        maps = torch.as_tensor(maps, device=self.device, dtype=self.dtype)
        valid = torch.as_tensor(valid, device=self.device, dtype=torch.bool)
        if maps.ndim == 5:
            maps, valid = maps.unsqueeze(0), valid.unsqueeze(0)
        if maps.shape[0] != batch_size:
            raise ValueError("Tactile and VA batch sizes differ")
        return self.touch_v2.encode_history(maps, valid)

    def prepare_inputs(self, batch):
        samples = [batch] if isinstance(batch, dict) else batch
        if any(sample.get("training_modality") == "video_only" for sample in samples):
            raise ValueError("TouchWAM requires video-action samples with explicit tactile validity")
        result = super().prepare_inputs(samples)
        for key in ("tactile_maps", "tactile_valid", "future_tactile_maps", "future_tactile_valid"):
            if any(key not in sample for sample in samples):
                raise ValueError(f"TouchWAM V2 training requires {key}")
            result[key] = torch.stack([torch.as_tensor(sample[key]) for sample in samples])
        expected_offsets = torch.arange(1, self.tactile_spec["future_frames"] + 1) * self.tactile_spec["future_stride"]
        for sample in samples:
            if "future_tactile_offsets" not in sample:
                raise ValueError("Dataset/model future tactile offsets differ")
            offsets = torch.as_tensor(sample["future_tactile_offsets"])
            # Accelerate may already have moved the whole sample to CUDA.
            if not torch.equal(offsets, expected_offsets.to(device=offsets.device)):
                raise ValueError("Dataset/model future tactile offsets differ")
        return result

    def forward(
        self,
        noisy_actions,
        action_timestep,
        *,
        noisy_tactile=None,
        tactile_timestep=None,
        tactile_future_valid=None,
        tactile_maps=None,
        tactile_valid=None,
        touch_condition=None,
        proprio=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **pipeline_inputs,
    ):
        if "future_tactile_maps" in pipeline_inputs or "future_tactile_valid" in pipeline_inputs:
            raise ValueError("Clean future tactile targets must not enter forward; use compute_loss")
        if noisy_actions is None:
            raise ValueError("TouchWAM V2 requires the action stream")
        if touch_condition is not None and (tactile_maps is not None or tactile_valid is not None):
            raise ValueError("Pass raw history OR touch_condition")
        vb, ab = self.video_backbone, self.action_backbone
        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        context, context_mask = pipeline_inputs.get("context"), pipeline_inputs.get("context_mask")
        if context is not None and context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            lengths = pipeline_inputs["seq_lens"].to(device=context.device)
            context_mask = torch.arange(context.shape[1], device=context.device)[None] < lengths[:, None]
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )
        if vstate.hidden_states.ndim != 3 or vstate.prefix_kv_len:
            raise ValueError("TouchWAM V2 requires a Wan-style [B,S,D] video stream without prefix KV")
        astate = ab.prepare_state(
            noisy_actions,
            action_timestep,
            context=context,
            context_mask=context_mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        history, history_valid = (
            self._history(tactile_maps, tactile_valid, noisy_actions.shape[0])
            if touch_condition is None
            else touch_condition
        )
        s, codec = self.tactile_spec, self.touch_v2.codec
        shape = (noisy_actions.shape[0], s["future_frames"] * codec.tokens_per_frame, codec.patch_dim)
        if noisy_tactile is None:
            noisy_tactile = noisy_actions.new_zeros(shape)
            tactile_future_valid = torch.zeros(shape[:2], dtype=torch.bool, device=noisy_actions.device)
            tactile_timestep = torch.zeros(noisy_actions.shape[0], device=noisy_actions.device)
        elif tactile_timestep is None or tactile_future_valid is None:
            raise ValueError("Noisy future touch requires timestep and token validity")
        if tactile_future_valid.shape != shape[:2] or tactile_future_valid.dtype != torch.bool:
            raise ValueError("tactile_future_valid must be bool [B,future_tokens]")
        noisy_tactile = noisy_tactile.masked_fill(~tactile_future_valid[..., None], 0)
        future, time = self.touch_v2.prepare_future(noisy_tactile, tactile_timestep)
        future = future.masked_fill(~tactile_future_valid[..., None], 0)
        driver = self.mot_driver
        per_frame = driver._video_tokens_per_frame(vstate)
        mask = driver._build_attention_mask(
            vstate.hidden_states.shape[1],
            astate.payload.x_action.shape[1],
            per_frame,
            device=vstate.hidden_states.device,
        )
        bridges = {layer: i for i, layer in enumerate(s["bridge_layers"])}
        clean_tokens = per_frame * max(1, int(pipeline_inputs.get("num_clean_prefix_frames", 0) or 0))
        for layer in range(vb.num_layers):
            vstate, astate = driver.step(
                layer,
                vstate,
                astate,
                mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            if layer in bridges:
                block = partial(self.touch_v2.blocks[bridges[layer]], clean_video_tokens=clean_tokens)
                args = (
                    future,
                    history,
                    tactile_future_valid,
                    history_valid,
                    time,
                    vstate.hidden_states,
                    astate.payload.x_action,
                )
                if use_gradient_checkpointing and self.training:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            future, vx, ax = checkpoint(block, *args, use_reentrant=False)
                    else:
                        future, vx, ax = checkpoint(block, *args, use_reentrant=False)
                else:
                    future, vx, ax = block(*args)
                vstate.hidden_states, astate.payload.x_action = vx, ax
        prediction = self.touch_v2.output(future).masked_fill(~tactile_future_valid[..., None], 0)
        return vb.finalize(vstate), ab.extract_prediction(astate), prediction

    def compute_loss(self, *, actions=None, lambda_video=1.0, lambda_action=1.0, **inputs):
        """Independent V/A/T noise levels, clean history, future-only tactile loss."""
        if actions is None:
            raise ValueError("TouchWAM V2 needs action targets")
        video_only = inputs.pop("video_only_mask", None)
        if video_only is not None and torch.as_tensor(video_only).any():
            raise ValueError("TouchWAM requires video-action samples with explicit tactile validity")
        future_maps = inputs.pop("future_tactile_maps").to(device=self.device, dtype=self.dtype)
        future_valid = inputs.pop("future_tactile_valid").to(device=self.device, dtype=torch.bool)
        if future_maps.shape[1] != self.tactile_spec["future_frames"]:
            raise ValueError("Future tactile target horizon differs from model config")
        clean_t, valid_t = self.touch_v2.codec.encode(future_maps, future_valid)
        batch_size = clean_t.shape[0]
        vs, acs, ts = self.video_scheduler, self.action_scheduler, self.tactile_scheduler
        if ts.timesteps is None:
            raise RuntimeError("Call init_training_schedulers before compute_loss")
        lower = int(inputs.pop("min_timestep_boundary", 0) * len(vs.timesteps))
        upper = int(inputs.pop("max_timestep_boundary", 1) * len(vs.timesteps))
        vi = torch.randint(lower, upper, (batch_size,))
        ai = torch.randint(len(acs.timesteps), (batch_size,))
        ti = torch.randint(len(ts.timesteps), (batch_size,))

        def noise_sample(clean, scheduler, indices):
            clean = clean.to(device=self.device, dtype=self.dtype)
            sigma = scheduler.sigmas[indices].to(device=self.device, dtype=self.dtype)
            sigma = sigma.reshape(batch_size, *([1] * (clean.ndim - 1)))
            noise = torch.randn_like(clean)
            timestep = scheduler.timesteps[indices].to(device=self.device, dtype=self.dtype)
            return (1 - sigma) * clean + sigma * noise, noise - clean, timestep

        noisy_v, target_v, time_v = noise_sample(inputs["input_latents"], vs, vi)
        noisy_a, target_a, time_a = noise_sample(actions, acs, ai)
        noisy_t, target_t, time_t = noise_sample(clean_t, ts, ti)
        ref = inputs.get("first_frame_latents")
        if ref is not None:
            noisy_v[:, :, : ref.shape[2]] = ref
        forward_inputs = dict(inputs)
        for key in ("action_is_pad", "video_is_pad", "input_latents"):
            forward_inputs.pop(key, None)
        proprio_mask = forward_inputs.pop("proprio_mask", None)
        if proprio_mask is not None:
            forward_inputs["_proprio_sample_mask"] = proprio_mask
        forward_inputs.update(latents=noisy_v, timestep=time_v)
        predictions = self(
            noisy_a,
            time_a,
            noisy_tactile=noisy_t,
            tactile_timestep=time_t,
            tactile_future_valid=valid_t,
            **forward_inputs,
        )
        loss_v = self._compute_video_loss(predictions[0], target_v, vi, inputs, self.device)
        loss_a = self._compute_action_loss(predictions[1], target_a, ai, acs, inputs, self.device)
        s = self.tactile_spec
        loss_t, loss_c, loss_d = tactile_flow_loss(
            self.touch_v2.codec,
            predictions[2],
            target_t,
            clean_t,
            valid_t,
            positive_weight=s["contact_positive_weight"],
            background_weight=s["distance_background_weight"],
        )
        return dict(
            loss=lambda_video * loss_v + lambda_action * loss_a + s["loss_weight"] * loss_t,
            loss_video=(lambda_video * loss_v).detach(),
            loss_action=(lambda_action * loss_a).detach(),
            loss_tactile=(s["loss_weight"] * loss_t).detach(),
            loss_tactile_contact=loss_c.detach(),
            loss_tactile_distance=loss_d.detach(),
        )

    def audit_openwam_checkpoint(self, path):
        """Header-only validation, also usable on a full-size meta skeleton."""
        from safetensors import safe_open

        expected = self.state_dict()
        new = {k for k in expected if k.startswith(self.new_weight_prefixes)}
        base = set(expected) - new
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if keys != base:
                raise RuntimeError(
                    f"OpenWAM warm start key mismatch: missing={sorted(base - keys)}, unexpected={sorted(keys - base)}"
                )
            for key in keys:
                if tuple(f.get_slice(key).get_shape()) != tuple(expected[key].shape):
                    raise RuntimeError(f"OpenWAM warm start shape mismatch: {key}")
        return dict(
            base_tensors=len(base),
            new_tensors=len(new),
            base_parameters=sum(expected[k].numel() for k in base),
            new_parameters=sum(expected[k].numel() for k in new),
        )

    def load_openwam_checkpoint(self, path):
        """Only new tactile tensors may be absent; Alpha paths/shapes must match."""
        from safetensors.torch import load_file

        self.audit_openwam_checkpoint(path)
        new = {k for k in self.state_dict() if k.startswith(self.new_weight_prefixes)}
        missing, unexpected = self.load_state_dict(
            load_file(path), strict=False, assign=any(p.is_meta for p in self.parameters())
        )
        if set(missing) != new or unexpected or any(t.is_meta for t in (*self.parameters(), *self.buffers())):
            raise RuntimeError("Incomplete OpenWAM -> TouchWAM V2 warm start")
        with torch.no_grad():
            for block in self.touch_v2.blocks:
                block.video_gate.zero_()
                block.action_gate.zero_()
        self._apply_training_policy()

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        from .inference import generate_three_stream

        return generate_three_stream(self, *args, **kwargs)

    def load_checkpoint(self, path, strict=True):
        from safetensors import safe_open

        if not strict:
            raise ValueError("V2 checkpoints require strict loading; use load_openwam_checkpoint for Alpha")
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        if any(metadata.get(k) != v for k, v in self.checkpoint_metadata.items()):
            raise RuntimeError("TouchWAM V2 checkpoint metadata/specification mismatch")
        super().load_checkpoint(path, strict=True)
        self._apply_training_policy()
