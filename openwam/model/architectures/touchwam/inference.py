"""Joint Euler inference for the compact third tactile stream."""

from __future__ import annotations

import time

import torch

from openwam.model.architectures.base import _assert_decode_video_supported, _expand_inputs_for_cfg


def _predict(model, actions, at, touch, shared, vt, scale, merge):
    touch_args = dict(noisy_tactile=touch, tactile_timestep=at)
    if scale == 1:
        return model(actions, at, **touch_args, **shared, timestep=vt)
    if not isinstance(shared.get("uncond_context"), torch.Tensor):
        raise ValueError("CFG requires uncond_context from the video preprocessor")
    if merge:
        expanded, ax, vx_t, ax_t = _expand_inputs_for_cfg(
            shared,
            action_latents=actions,
            v_timestep=vt,
            a_timestep=at,
        )
        expanded["touch_condition"] = tuple(torch.cat((x, x)) for x in shared["touch_condition"])
        expanded["tactile_future_valid"] = torch.cat((shared["tactile_future_valid"],) * 2)
        if "uncond_context_mask" in shared:
            expanded["context_mask"] = torch.cat((shared["uncond_context_mask"], shared["context_mask"]))
        predictions = model(
            ax, ax_t, noisy_tactile=torch.cat((touch, touch)), tactile_timestep=ax_t, **expanded, timestep=vx_t
        )
        pairs = [prediction.chunk(2) for prediction in predictions]
    else:
        conditional = model(actions, at, **touch_args, **shared, timestep=vt)
        uncond_inputs = dict(shared, context=shared["uncond_context"])
        # Respect separate text padding when supplied by a preprocessor.
        if "uncond_context_mask" in shared:
            uncond_inputs["context_mask"] = shared["uncond_context_mask"]
        unconditional = model(actions, at, **touch_args, **uncond_inputs, timestep=vt)
        pairs = zip(unconditional, conditional)
    return tuple(uncond + scale * (cond - uncond) for uncond, cond in pairs)


@torch.no_grad()
def generate_three_stream(
    model,
    schedule,
    prompt,
    *,
    tactile_maps=None,
    tactile_valid=None,
    tactile_region_ids=None,
    tactile_layout_present=None,
    tactile_query_mask=None,
    tactile_spatial_valid=None,
    tactile_sensor_ids=None,
    tactile_surface_ids=None,
    tactile_palm_mode=None,
    vace_video=None,
    first_frame_image=None,
    num_frames=9,
    action_num_frames=None,
    height=384,
    width=320,
    seed=42,
    tiled=True,
    input_video_latents=None,
    num_inference_steps=50,
    shift=5.0,
    dit_cache=None,
    decode_video=True,
    profile=False,
    vace_cache=None,
    prompt_embed_cache=None,
    proprio=None,
    cfg_scale=1.0,
    cfg_merge=False,
    active_action_mask=None,
):
    """Touch follows the action sigma schedule, including asynchronous plateaus.

    The architecture defines query availability: legacy V2 uses history;
    regional models use explicit layout and requests, never future labels.
    """
    model.eval()
    if dit_cache is not None:
        raise ValueError("TouchWAM V2 requires DiT cache disabled: the legacy cache stores only V/A predictions")
    if cfg_scale < 1:
        raise ValueError("cfg_scale must be >= 1")
    schedule = [(float(v), float(a)) for v, a in schedule]
    if len(schedule) < 2:
        raise ValueError("A denoising schedule needs at least two endpoints")
    nv, na = model.video_scheduler.num_train_timesteps, model.action_scheduler.num_train_timesteps
    if schedule[0] != (float(nv), float(na)) or schedule[-1] != (0.0, 0.0):
        raise ValueError("V2 generation from noise requires sigma endpoints 1 -> 0 for both V/A")
    if any(not (0 <= v <= nv and 0 <= a <= na) for v, a in schedule):
        raise ValueError("Denoising timesteps outside [0,num_train_timesteps]")
    if any(v1 > v0 or a1 > a0 for (v0, a0), (v1, a1) in zip(schedule, schedule[1:])):
        raise ValueError("Denoising schedule must be non-increasing")
    s = model.tactile_spec
    action_num_frames = int(action_num_frames if action_num_frames is not None else 33)
    if action_num_frames - 1 < s["future_frames"] * s["future_stride"]:
        raise ValueError("Action chunk must cover the configured future tactile horizon")
    start = time.perf_counter()
    device, dtype, vb = model.device, model.dtype, model.video_backbone
    shared = vb.preprocess_input_for_inference(
        prompt=prompt,
        vace_video=vace_video,
        first_frame_image=first_frame_image,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
        num_inference_steps=num_inference_steps,
        shift=shift,
        tiled=tiled,
        vace_cache=vace_cache,
        prompt_embed_cache=prompt_embed_cache,
        cfg_scale=cfg_scale,
        cfg_merge=cfg_merge,
    )
    if input_video_latents is not None:
        shared["latents"] = input_video_latents.to(device=device, dtype=dtype)
    if shared["latents"].shape[0] != 1:
        raise ValueError("generate currently accepts one observation; forward/compute_loss support batches")
    ref = shared.get("first_frame_latents")
    if ref is not None:
        shared["latents"] = shared["latents"].clone()
        shared["latents"][:, :, : ref.shape[2]] = ref
    if model.uses_proprioception:
        if proprio is None:
            raise ValueError("use_proprioception=True requires proprio")
        shared["proprio"] = torch.as_tensor(proprio, device=device, dtype=dtype)
    history, future_valid, token_valid = model.prepare_tactile_generation(
        tactile_maps, tactile_valid, region_ids=tactile_region_ids,
        layout_present=tactile_layout_present, query_mask=tactile_query_mask,
        spatial_valid=tactile_spatial_valid, sensor_ids=tactile_sensor_ids,
        surface_ids=tactile_surface_ids,
        palm_mode=tactile_palm_mode,
    )
    shared["touch_condition"] = history
    shared["tactile_future_valid"] = token_valid
    actions = torch.randn(
        1,
        action_num_frames - 1,
        model.action_dim,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    # A separate RNG preserves Alpha's initial action noise for paired checks.
    touch = model.sample_tactile_noise(history, token_valid, seed)
    inactive = model._resolve_inactive_action_dims(active_action_mask, device)
    inactive_noise = actions[..., inactive].clone() if inactive is not None else None
    for (tv, ta), (tv_next, ta_next) in zip(schedule, schedule[1:]):
        if tv == tv_next and ta == ta_next:
            continue
        vt = torch.tensor([tv], device=device, dtype=dtype)
        at = torch.tensor([ta], device=device, dtype=dtype)
        pv, pa, pt = _predict(model, actions, at, touch, shared, vt, cfg_scale, cfg_merge)
        shared["latents"] = shared["latents"] + pv * ((tv_next - tv) / nv)
        if ref is not None:
            shared["latents"][:, :, : ref.shape[2]] = ref
        actions = actions + pa * ((ta_next - ta) / na)
        touch = (touch + pt * ((ta_next - ta) / na)).masked_fill(~token_valid[..., None], 0)
        if inactive is not None:
            actions[..., inactive] = inactive_noise * (ta_next / na)
    if decode_video:
        _assert_decode_video_supported(vb)
        video = vb.decode_video(shared["latents"], tiled=tiled)
    else:
        video = None
    actions = actions[0].float().cpu().numpy()
    normalizer = getattr(model, "normalizer", None)
    if normalizer is not None:
        actions = normalizer.unnormalize(actions)
    result = dict(
        video=video,
        actions=actions,
        **model.decode_tactile_generation(touch, future_valid, history),
        future_tactile_offsets=torch.arange(1, s["future_frames"] + 1).numpy() * s["future_stride"],
        tactile_fps=s["fps"],
        tactile_prediction_kind="generated",
    )
    if profile:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["generation_seconds"] = time.perf_counter() - start
    return result
