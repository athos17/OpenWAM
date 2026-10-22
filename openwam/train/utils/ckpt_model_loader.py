"""Self-contained architecture construction for finetune / resume.

When ``training.finetune_ckpt_path`` or ``training.resume_ckpt_path`` is set,
the trainer builds the architecture from that checkpoint source alone:
module skeletons from the component specs saved in its ``config.yaml``,
weights from either the requested ``checkpoint_step_*.safetensors`` file or
the latest weights in the directory (finetune), or from accelerate's
``load_state`` after prepare (resume). The original pretrained backbone
directory (``model.video_backbone.model_path``) does not need to exist on the
training host.

The checkpoint's ``config.yaml`` is the RECONSTRUCTION base, not the run's
authority. On the finetune path the caller passes ``override_cfg`` (the live
Hydra config) and the ckpt's ``model`` section is layered under ``cfg.model``:
the ckpt supplies the ``components`` / ``tokenizer`` specs that only it has,
every value the operator set in ``configs/`` (or on the CLI) wins. The merged
result is written back into ``cfg.model``, so one config both builds the
architecture and is written by ``save_config`` into the new run dir. Resume
passes no ``override_cfg`` — it continues one run whose ``config.yaml`` is
already on disk, so the ckpt config stays authoritative there and any live
divergence is reported by :func:`warn_live_model_cfg_ignored` instead.

Train-side counterpart of the deploy loader (``openwam/deploy/model_loader.py``),
kept independent so training never imports deploy code.
"""

import glob
import logging
import os
import shutil

from omegaconf import DictConfig, OmegaConf, open_dict

from openwam.train.utils.checkpointing import find_latest_weights

logger = logging.getLogger(__name__)


def _resolve_ckpt_source(source: str) -> tuple[str, str | None]:
    """Return ``(ckpt_dir, explicit_weights)`` for a directory or safetensors file."""
    source = os.fspath(source)
    if source.endswith(".safetensors"):
        # Fail fast on a typo'd filename — otherwise safetensors only errors
        # after minutes of architecture-skeleton construction.
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Explicit checkpoint weights file does not exist: {source}")
        return os.path.dirname(os.path.abspath(source)), os.path.abspath(source)
    return source, None


# ----------------------------------------------------------------------------
# ckpt model config (base) + live model config (override)
# ----------------------------------------------------------------------------

# Keys whose live-vs-ckpt disagreement an override cannot express: each one
# selects WHICH module classes the checkpoint's weights belong to. The ckpt's
# ``components`` specs (module class + ctor kwargs per attr) survive the merge
# by construction, so overriding one of these would hand a skeleton built from
# the ckpt's classes to a different wrapper / architecture class — a state_dict
# topology no safetensors file matches. Value is the CLI hint for the fix.
_ARCH_IDENTITY_KEYS = (
    ("video_backbone.name", "model/video_backbone=<ckpt value>"),
    ("architecture.framework", "model=<ckpt value>"),
)

# Cap on a single value's printed length in the diff report: ``components`` is a
# multi-KB list of module specs and would bury the keys that actually changed.
_DIFF_VALUE_MAXLEN = 110


def _flatten_cfg(node, prefix: str = "") -> dict:
    """Flatten a plain-container config into ``{dotted.key: leaf}``.

    Lists stay leaves. ``components`` is a list of per-module spec dicts that
    only ever moves as a unit, and per-index keys would bury the real diff.
    """
    flat: dict = {}
    for key, val in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(val, dict) and val:
            flat.update(_flatten_cfg(val, prefix=f"{path}."))
        else:
            flat[path] = val
    return flat


def _short(val) -> str:
    text = repr(val)
    return text if len(text) <= _DIFF_VALUE_MAXLEN else text[: _DIFF_VALUE_MAXLEN - 3] + "..."


def diff_model_cfgs(ckpt_model, live_model) -> tuple[list, list, list]:
    """Compare two ``model`` configs key-by-key.

    Returns ``(overridden, live_only, ckpt_only)`` where ``overridden`` holds
    ``(dotted_key, ckpt_value, live_value)`` for keys both sides define with
    different values, and the other two hold ``(dotted_key, value)``.
    """
    base = _flatten_cfg(OmegaConf.to_container(ckpt_model, resolve=True) or {})
    over = _flatten_cfg(OmegaConf.to_container(live_model, resolve=True) or {})
    # Keys are unique, so tuple ordering never has to compare the values.
    overridden = sorted((k, base[k], over[k]) for k in over if k in base and base[k] != over[k])
    live_only = sorted((k, over[k]) for k in over if k not in base)
    ckpt_only = sorted((k, base[k]) for k in base if k not in over)
    return overridden, live_only, ckpt_only


def _check_arch_identity(ckpt_model, live_model, ckpt_dir: str) -> None:
    """Reject a live/ckpt disagreement on the keys that pick the module classes."""
    for key, hint in _ARCH_IDENTITY_KEYS:
        ckpt_val = OmegaConf.select(ckpt_model, key, default=None)
        live_val = OmegaConf.select(live_model, key, default=None)
        if ckpt_val is None or live_val is None or ckpt_val == live_val:
            continue
        raise ValueError(
            f"finetune config mismatch on model.{key}: live config says {live_val!r}, "
            f"{os.path.join(ckpt_dir, 'config.yaml')} says {ckpt_val!r}. This key selects which "
            "module classes the checkpoint's weights belong to, so it is the one thing the live "
            "config cannot override — the merged skeleton would match no safetensors file. Set "
            f"`{hint}` to the checkpoint's value, or point training.finetune_ckpt_path at a "
            "checkpoint of the architecture you want to train."
        )


def _log_model_cfg_diff(ckpt_dir: str, overridden: list, live_only: list, ckpt_only: list) -> None:
    """Report the merge on the launch terminal — an override must never be silent."""
    header = (
        f"[finetune] model config = {os.path.join(ckpt_dir, 'config.yaml')} (base) "
        f"+ live cfg.model (override): {len(overridden)} overridden, "
        f"{len(live_only)} live-only, {len(ckpt_only)} ckpt-only"
    )
    logger.info(header)
    # Plain print (rank-0 only; train.py silences print on other ranks) so the
    # merge is visible on the launch terminal even when logs are drowned out.
    print(header, flush=True)
    for key, ckpt_val, live_val in overridden:
        line = f"[finetune]   override  model.{key}: {_short(ckpt_val)} (ckpt) -> {_short(live_val)} (live)"
        logger.info(line)
        print(line, flush=True)
    # The other two buckets are counted in the header but only spelled out at
    # DEBUG: ``ckpt_only`` is the (always present, never actionable)
    # components/tokenizer spec tree, and logging is not rank-gated here, so at
    # INFO they would repeat once per rank on every launch.
    for key, live_val in live_only:
        logger.debug("[finetune]   live-only model.%s = %s", key, _short(live_val))
    for key, ckpt_val in ckpt_only:
        logger.debug("[finetune]   from-ckpt model.%s = %s", key, _short(ckpt_val))


def merge_ckpt_model_cfg(ckpt_cfg: DictConfig, cfg: DictConfig, ckpt_dir: str) -> DictConfig:
    """Layer the live run's ``model`` config on top of the checkpoint's.

    The ckpt's ``model`` section is the BASE: it alone carries the ``components``
    / ``tokenizer`` reconstruction specs that let the module skeletons be rebuilt
    without the original pretrained backbone dir. The live ``cfg.model`` is the
    OVERRIDE, so every value the operator set wins — including an explicit
    ``null``, which is how ``bridge_layers: null`` clears a ckpt's explicit list.
    Keys only the ckpt has survive; keys only the live config has are added.

    The merged result is written back into ``cfg.model`` so that exactly one
    config both builds the architecture here and is written by ``save_config``
    into the new run dir. Without the write-back the new checkpoint's
    ``config.yaml`` would advertise values the model was never built with, and
    deploying it would rebuild a different model than the one that trained.

    Returns the merged node as it now lives in ``cfg``.
    """
    ckpt_model = OmegaConf.select(ckpt_cfg, "model", default=None)
    live_model = OmegaConf.select(cfg, "model", default=None)
    if ckpt_model is None:
        # A ckpt config without a model section can't reconstruct anything; let
        # the live config carry the build and fail downstream if it can't.
        logger.warning("[finetune] %s has no `model:` section; using the live cfg.model unchanged.", ckpt_dir)
        return live_model
    if live_model is None:
        return ckpt_model

    _check_arch_identity(ckpt_model, live_model, ckpt_dir)
    _log_model_cfg_diff(ckpt_dir, *diff_model_cfgs(ckpt_model, live_model))

    merged = OmegaConf.merge(ckpt_model, live_model)
    with open_dict(cfg):
        cfg.model = merged
    return cfg.model


def warn_live_model_cfg_ignored(ckpt_cfg: DictConfig, cfg: DictConfig, ckpt_dir: str) -> None:
    """Report live ``model`` keys the resume path discards.

    Resume continues ONE run: its ``config.yaml`` is already on disk and
    ``setup_output_dir`` reuses the dir without rewriting it, so the ckpt config
    stays authoritative and a live override would make that file lie about the
    model being trained. It must still not be silent — a resumed run whose live
    yaml says something else is an operator error worth seeing.
    """
    ckpt_model = OmegaConf.select(ckpt_cfg, "model", default=None)
    live_model = OmegaConf.select(cfg, "model", default=None)
    if ckpt_model is None or live_model is None:
        return
    overridden, _live_only, _ckpt_only = diff_model_cfgs(ckpt_model, live_model)
    if not overridden:
        return
    msg = (
        f"[resume] live cfg.model disagrees with {os.path.join(ckpt_dir, 'config.yaml')} on "
        f"{len(overridden)} key(s). Resume continues the checkpoint's run, so the CKPT values win "
        "and the live ones are IGNORED (use training.finetune_ckpt_path to start a new run whose "
        "config overrides the checkpoint's):"
    )
    logger.warning(msg)
    print(msg, flush=True)
    for key, ckpt_val, live_val in overridden:
        line = f"[resume]   model.{key}: using {_short(ckpt_val)} (ckpt), ignoring {_short(live_val)} (live)"
        logger.warning(line)
        print(line, flush=True)


_DIT_PREFIX = "video_backbone.dit."
_SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def _safetensors_files(path: str) -> list[str]:
    """A single ``.safetensors`` file, or every ``*.safetensors`` shard in a directory."""
    files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) if os.path.isdir(path) else [path]
    if not files or not all(os.path.isfile(f) for f in files):
        raise FileNotFoundError(f"No safetensors weights at {path}")
    return files


def load_partial_finetune_weights(
    architecture, weights: str, *, load_prefixes=None, video_dit_path: str | None = None
) -> None:
    """Warm-start only part of the architecture; everything else keeps its fresh init.

    ``load_prefixes`` selects which ckpt keys are loaded (e.g. ``["video_backbone."]``
    leaves the action branch and proprio projection at the seeded random init).
    ``video_dit_path`` replaces ``video_backbone.dit.*`` with an external Wan DiT
    (e.g. the original Wan2.2-TI2V-5B shards), cast to the ckpt's per-tensor dtype
    so both initializations enter training at the same precision. Every loaded
    prefix must be covered completely: a key the sources do not supply raises.
    """
    import torch
    from safetensors import safe_open

    prefixes = tuple(load_prefixes) if load_prefixes else ("",)
    if video_dit_path and not _DIT_PREFIX.startswith(prefixes):
        raise ValueError(f"finetune_video_dit_path needs {_DIT_PREFIX} inside finetune_load_prefixes={list(prefixes)}")

    state_dict, ckpt_dit_dtypes = {}, {}
    with safe_open(weights, framework="pt") as f:
        for key in f.keys():
            if video_dit_path and key.startswith(_DIT_PREFIX):
                ckpt_dit_dtypes[key] = getattr(torch, _SAFETENSORS_DTYPES[f.get_slice(key).get_dtype()])
            elif key.startswith(prefixes):
                state_dict[key] = f.get_tensor(key)
    n_ckpt = len(state_dict)

    if video_dit_path:
        dit = {}
        for path in _safetensors_files(video_dit_path):
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    name = _DIT_PREFIX + key
                    dit[name] = f.get_tensor(key)
                    if name in ckpt_dit_dtypes:
                        dit[name] = dit[name].to(ckpt_dit_dtypes[name])
        if set(dit) != set(ckpt_dit_dtypes):
            raise ValueError(
                f"{video_dit_path} DiT keys differ from the ckpt's {_DIT_PREFIX}*: "
                f"missing={sorted(set(ckpt_dit_dtypes) - set(dit))[:5]} extra={sorted(set(dit) - set(ckpt_dit_dtypes))[:5]}"
            )
        live = architecture.state_dict()
        bad = [k for k, v in dit.items() if tuple(v.shape) != tuple(live[k].shape)]
        if bad:
            raise ValueError(f"{video_dit_path} DiT shape mismatch, e.g. {bad[0]}: {tuple(dit[bad[0]].shape)}")
        state_dict.update(dit)

    has_meta = any(p.device.type == "meta" for p in architecture.parameters())
    missing, unexpected = architecture.load_state_dict(state_dict, strict=False, assign=has_meta)
    not_loaded = [k for k in missing if k.startswith(prefixes)]
    if unexpected or not_loaded:
        raise RuntimeError(f"Partial finetune load failed: not_loaded={not_loaded[:10]}, unexpected={unexpected[:10]}")
    still_meta = [n for n, p in architecture.named_parameters() if p.device.type == "meta"]
    if still_meta:
        raise RuntimeError(f"Partial finetune load left meta (uninitialized) params, e.g. {still_meta[:5]}")

    fresh = sorted({".".join(k.split(".")[:2]) for k in missing})
    msg = (
        f"[finetune] partial load: {n_ckpt} tensors from {weights} (prefixes={list(prefixes)})"
        + (f", {len(state_dict) - n_ckpt} DiT tensors from {video_dit_path}" if video_dit_path else "")
        + f"; fresh init: {fresh}"
    )
    logger.info(msg)
    print(msg, flush=True)


def build_architecture_from_ckpt_dir(
    ckpt_dir: str,
    *,
    weights_required: bool,
    override_cfg: DictConfig = None,
    load_prefixes=None,
    video_dit_path: str | None = None,
):
    """Build the architecture from a self-contained checkpoint source.

    Skeletons come from the resolved model config's
    ``model.video_backbone.components`` specs (tokenizer resolved against
    ``<ckpt_dir>/tokenizer/``). With ``weights_required=True`` (finetune) either
    the explicit safetensors source or latest ``checkpoint_step_*.safetensors``
    in the source dir is loaded here; with
    ``weights_required=False`` (resume) safetensors are skipped entirely —
    accelerate's ``load_state`` restores a strict superset (module weights
    incl. frozen params) after prepare, and reading the latest safetensors
    would crash on a file truncated by a mid-save kill even though the run
    is still recoverable from its accel state.

    ``override_cfg`` is the live run config (finetune). When given, its ``model``
    section is layered over the ckpt's via :func:`merge_ckpt_model_cfg` and the
    merged result — written back into ``override_cfg.model`` — is what the
    architecture is built from. When omitted the ckpt's ``model`` section is used
    verbatim (resume, and standalone callers).

    ``load_prefixes`` / ``video_dit_path`` (finetune only) switch the weight load
    to :func:`load_partial_finetune_weights`.

    Returns ``(resolved_arch, architecture, ckpt_cfg)``.
    """
    from openwam.model import build_architecture, resolve_architecture_config

    ckpt_dir, explicit_weights = _resolve_ckpt_source(ckpt_dir)
    tag = "finetune" if weights_required else "resume"
    ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    if override_cfg is not None:
        model_cfg = merge_ckpt_model_cfg(ckpt_cfg, override_cfg, ckpt_dir)
    else:
        model_cfg = OmegaConf.select(ckpt_cfg, "model", default=None)
    if model_cfg is None:
        raise ValueError(
            f"{os.path.join(ckpt_dir, 'config.yaml')} has no `model:` section and neither does the "
            f"live run config, so there is nothing to build the architecture from. Point "
            f"training.{'finetune' if weights_required else 'resume'}_ckpt_path at a run directory "
            "written by this trainer (its config.yaml carries the model section and the component specs)."
        )
    resolved_arch = resolve_architecture_config(model_cfg)
    params = dict(resolved_arch.params)

    # Pin video_backbone params to a plain dict: OmegaConf would auto-promote
    # the ``_source`` assignment below into a DictConfig, and build_holder()
    # dispatches DictConfig sources to the training pipeline (which needs
    # cfg.training plus the pretrained model_path dir) instead of the
    # component-spec path.
    vb_params = params.get("video_backbone") or {}
    if not isinstance(vb_params, dict):
        vb_params = OmegaConf.to_container(vb_params, resolve=True) or {}
    params["video_backbone"] = vb_params
    # ``_source`` comes from the SAME resolved config the skeletons were sized
    # from, not from ckpt_cfg directly — otherwise a live override of a
    # video_backbone key (shift_video, from_scratch, ...) would reach the
    # architecture but not the backbone build that reads it back off ``_source``.
    vb_params["_source"] = OmegaConf.to_container(model_cfg.video_backbone, resolve=True)
    vb_params["_ckpt_dir"] = ckpt_dir
    # Finetune loads safetensors a few lines below, so a backbone may leave an
    # empty shell for that load to fill. Resume must not: `load_state` runs only
    # after `accelerator.prepare`, and `set_dtype_device` (openwam_trainer.py:126)
    # touches the module first — on a meta shell that raises "Cannot copy out of
    # meta tensor". Backbones without a shell path ignore the key.
    vb_params["_materialize_weights"] = not weights_required

    logger.info("[%s] building architecture from self-contained ckpt dir: %s", tag, ckpt_dir)
    architecture = build_architecture(resolved_arch.registry_name, params)

    if weights_required:
        weights = explicit_weights or find_latest_weights(ckpt_dir)
        logger.info("[%s] loading pretrained weights: %s", tag, weights)
        # Plain print so the warm-start is visible on the launch terminal even
        # when logger output is drowned out; non-main ranks have print disabled.
        print(f"[{tag}] loading pretrained weights: {weights}", flush=True)
        if load_prefixes or video_dit_path:
            load_partial_finetune_weights(
                architecture, weights, load_prefixes=load_prefixes, video_dit_path=video_dit_path
            )
        else:
            architecture.load_checkpoint(weights)
        print(f"[{tag}] pretrained weights loaded OK", flush=True)
    return resolved_arch, architecture, ckpt_cfg


def propagate_component_specs(ckpt_cfg: DictConfig, cfg: DictConfig) -> None:
    """Carry the ckpt's reconstruction specs into the live run cfg.

    save_video_backbone_deploy_assets() no-ops when model_path is unreadable,
    so without this the new run's config.yaml would lose the specs and its
    checkpoints would no longer be self-contained.

    The finetune path does not need this — :func:`merge_ckpt_model_cfg` already
    wrote the whole merged model config (specs included) back into ``cfg``. It
    remains for the resume path, which builds from the ckpt config verbatim.
    """
    for key in ("components", "tokenizer"):
        val = OmegaConf.select(ckpt_cfg, f"model.video_backbone.{key}", default=None)
        if val is not None and OmegaConf.select(cfg, f"model.video_backbone.{key}", default=None) is None:
            with open_dict(cfg):
                OmegaConf.update(cfg, f"model.video_backbone.{key}", val)
            logger.info("[self-contained] propagated model.video_backbone.%s specs from ckpt config", key)


# Tokenizer artifact dirs written by save_deploy_assets, per backbone family:
# Wan uses ``tokenizer/``, cosmos3_edge uses ``text_tokenizer/``. Relaying only
# the first would produce a checkpoint that carries the ``components`` marker
# (so it claims self-containment) but no tokenizer for the deploy build to read.
_CKPT_ARTIFACT_DIRS = ("tokenizer", "text_tokenizer")


def copy_ckpt_artifacts(ckpt_dir: str, output_dir: str) -> None:
    """Relay the checkpoint's tokenizer artifact dirs into the new run dir.

    Same reason as propagate_component_specs: keeps the self-containment chain
    alive when the tokenizer's original model_path source is unreachable. Only
    the dirs that exist are copied, so this no-ops per backbone as appropriate.
    """
    ckpt_dir, _explicit_weights = _resolve_ckpt_source(ckpt_dir)
    for name in _CKPT_ARTIFACT_DIRS:
        src = os.path.join(ckpt_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.isdir(src) and not os.path.isdir(dst):
            shutil.copytree(src, dst)
            logger.info("[self-contained] relayed tokenizer dir: %s -> %s", src, dst)
