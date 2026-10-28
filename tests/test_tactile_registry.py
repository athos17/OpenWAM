"""Frozen identities, native coverage, unique regions and sensor composition and safe warm-starts."""

import copy
import json
from pathlib import Path

import pytest
import torch

from openwam.model import build_architecture
from openwam.tactile.layout import SurfaceLayout, SurfaceSpec
from openwam.tactile.regions import collate_regions
from openwam.tactile.registry import (
    LEGACY_REGION_NAMES,
    PREVIOUS_REGION_NAMES,
    PREVIOUS_REGISTRY_SHA256,
    PREVIOUS_REGISTRY_VERSION,
    REGION_NAMES,
    REGIONS,
    REGISTRY_SHA256,
    REGISTRY_VERSION,
    region_spec,
    registry_document,
)
from tests.test_touchwam import open_gates, tiny_config
from tests.test_touchwam_regions import models as regional_models
from tests.test_touchwam_regions import regional_inputs, regional_loss_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def models(monkeypatch, tmp_path):
    return regional_models.__wrapped__(monkeypatch, tmp_path)


def test_frozen_registry_and_serialized_contract():
    assert REGISTRY_SHA256 == "dd12e94655597366eb2b99929220691fa95ade9a9dc261f01962460b2ced12c1"
    assert len(REGIONS) == 67 and REGION_NAMES[:12] == LEGACY_REGION_NAMES
    assert [r.id for r in REGIONS] == list(range(67))
    assert json.loads((ROOT / "docs/touchwam/region_registry_v2.json").read_text()) == json.loads(
        json.dumps(registry_document())
    )
    assert "palm" in region_spec("right.thumb.base_thenar").groups
    with pytest.raises(ValueError, match="Unregistered"):
        region_spec("right.index.tpi")


def test_layouts_keep_touchscale_coverage_and_gripper_identity():
    directory = ROOT / "configs/tactile_layouts"
    sharpa = SurfaceLayout.load(directory / "sharpa_ten_fingertips.json")
    touchscale = SurfaceLayout.load(directory / "touchscale_full_hand.json")
    gripper = SurfaceLayout.load(directory / "univtac_isaac51_gripper.json")
    assert len(sharpa.surfaces) == 10
    assert len(touchscale.surfaces) == 30
    assert sum(h * w for h, w in (s.native_shape for s in touchscale.surfaces)) == 1760
    assert len(touchscale.select("fingertips").surfaces) == 10
    assert len(touchscale.select("without_palm").surfaces) == 26
    assert not any("pinky.proximal" in s.region for s in touchscale.surfaces)
    assert {s.region for s in touchscale.surfaces if s.native_name == "thumb_base_thenar"} == {
        "right.thumb.base_thenar",
        "left.thumb.base_thenar",
    }
    assert [s.key for s in gripper.surfaces] == [58, 59]
    assert {s.representation for s in touchscale.surfaces} == {"touchscale_normal_force_v1"}
    assert {s.representation for s in gripper.surfaces} == {"univtac_indent_mm_v1"}
    assert [s.region for s in gripper.surfaces] == ["right.gripper.jaw_a.surface", "right.gripper.jaw_b.surface"]
    dual = SurfaceLayout.load(directory / "dual_gripper_deform_example.json")
    assert [s.key for s in dual.surfaces] == [58, 59, 64, 65]
    SurfaceLayout.load(directory / "single_arm_composed_deform_example.json")


def test_unique_regions_and_unsupported_instance_fields(tmp_path):
    fields = dict(
        region="generic.surface",
        sensor="new_sensor",
        representation="deform",
        units="uint8",
        native_shape=(7, 11),
        mapping_note="Other contact skin, fixed native coverage",
    )
    one = SurfaceSpec(native_key="pad/A", **fields)
    with pytest.raises(ValueError, match="Duplicate region"):
        SurfaceLayout("bad", (one, SurfaceSpec(native_key="pad/B", **fields)))
    with pytest.raises(TypeError, match="surface_id"):
        SurfaceSpec(native_key="pad/B", surface_id=1, **fields)
    with pytest.raises(ValueError, match="mapping_note"):
        SurfaceSpec(native_key="unknown", **dict(fields, mapping_note=""))
    assert "tactile_surface_ids" not in SurfaceLayout("one", (one,)).identity_fields()
    for name in ("gripper.jaw_a.surface", "gripper.jaw_b.surface", "gripper.base"):
        with pytest.raises(ValueError, match="Unregistered"):
            region_spec(name)


def test_canonical_config_and_contract(models):
    _, _, model = models
    assert tuple(model.tactile_spec["region_names"]) == REGION_NAMES
    assert model.tactile_input_contract["region_registry"] == REGISTRY_VERSION
    cfg = tiny_config()
    cfg["tactile"]["region_names"] = list(reversed(REGION_NAMES))
    with pytest.raises(ValueError, match="frozen"):
        build_architecture("touchwam_regions", cfg)


def test_checkpoint_binds_frozen_registry_hash(models, tmp_path):
    from safetensors.torch import save_file

    _, _, model = models
    metadata = dict(model.checkpoint_metadata, tactile_region_registry_sha256="different_definition")
    path = tmp_path / "incompatible.safetensors"
    save_file(model.state_dict(), str(path), metadata=metadata)
    with pytest.raises(RuntimeError, match="metadata"):
        model.load_checkpoint(str(path))


def test_composite_palm_group_and_full_hand_gradients(models):
    _, _, model = models
    layout = SurfaceLayout.load(ROOT / "configs/tactile_layouts/touchscale_full_hand.json")
    ids = torch.tensor([s.key for s in layout.surfaces])
    # Synthetic DEFORM fixture tests geometry/identity routing, not a force codec.
    args = regional_inputs(30)
    args["tactile_region_ids"] = ids[None].expand(2, -1)
    condition = model._condition(
        args["tactile_maps"],
        args["tactile_valid"],
        2,
        region_ids=args["tactile_region_ids"],
        layout_present=args["tactile_layout_present"],
        palm_mode="off",
    )
    assert condition[3].sum(1).tolist() == [26, 26]
    masked = [i for i, s in enumerate(layout.surfaces) if "palm" in region_spec(s.region).groups]
    assert masked == [2, 14, 17, 29]
    assert not condition[3][:, masked].any()
    losses = regional_loss_inputs(30)
    losses["tactile_region_ids"] = args["tactile_region_ids"]
    model.compute_loss(**losses)["loss"].backward()
    assert model.touch_v2.region_embedding.weight.grad[ids].abs().sum() > 0


def test_left_right_gripper_history_and_future_permute_together(models):
    _, _, model = models
    model.eval()
    open_gates(model)
    args = regional_inputs(2, batch=1)
    args["tactile_region_ids"][:] = 61
    with pytest.raises(ValueError, match="Duplicate"):
        model(**args)
    args["tactile_region_ids"] = torch.tensor([[58, 64]])
    before = model(**args)
    swapped = dict(args)
    for key in ("tactile_maps", "tactile_valid"):
        swapped[key] = args[key].flip(2)
    for key in ("tactile_region_ids", "tactile_layout_present"):
        swapped[key] = args[key].flip(1)
    swapped["noisy_tactile"] = args["noisy_tactile"].reshape(1, 2, 2, 4, 16).flip(2).flatten(1, 3)
    after = model(**swapped)
    for a, b in zip(before[:2], after[:2]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        before[2].reshape(1, 2, 2, 4, 16).flip(2), after[2].reshape(1, 2, 2, 4, 16), rtol=1e-5, atol=1e-6
    )
    # Left/right region identities keep independent RNG, stable under permutations.
    condition, _, mask = model.prepare_tactile_generation(
        args["tactile_maps"],
        args["tactile_valid"],
        region_ids=args["tactile_region_ids"],
        layout_present=args["tactile_layout_present"],
    )
    noise = model.sample_tactile_noise(condition, mask, 42).reshape(1, 2, 2, 4, 16)
    assert not torch.equal(noise[:, :, 0], noise[:, :, 1])
    swapped_condition, _, swapped_mask = model.prepare_tactile_generation(
        swapped["tactile_maps"],
        swapped["tactile_valid"],
        region_ids=swapped["tactile_region_ids"],
        layout_present=swapped["tactile_layout_present"],
    )
    torch.testing.assert_close(
        noise.flip(2), model.sample_tactile_noise(swapped_condition, swapped_mask, 42).reshape_as(noise), rtol=0, atol=0
    )


@pytest.mark.parametrize("merge", [False, True])
def test_left_right_gripper_generation_keeps_unique_regions(models, merge):
    _, _, model = models
    args = regional_inputs(2, batch=1)
    result = model.generate(
        schedule=[(1000, 1000), (0, 0)],
        prompt="pick",
        action_num_frames=5,
        decode_video=False,
        cfg_scale=2,
        cfg_merge=merge,
        proprio=args["proprio"],
        tactile_maps=args["tactile_maps"],
        tactile_valid=args["tactile_valid"],
        tactile_region_ids=torch.tensor([[58, 64]]),
        tactile_layout_present=args["tactile_layout_present"],
    )
    assert result["tactile_region_names"] == ["right.gripper.jaw_a.surface", "left.gripper.jaw_a.surface"]
    assert "tactile_surface_ids" not in result
    assert result["future_tactile_maps"].shape == (2, 2, 1, 8, 8)


def test_collate_unique_regions_and_reject_deprecated_identity():
    args = regional_loss_inputs(2)
    sample = {
        k: v[0]
        for k, v in args.items()
        if k
        in ("tactile_maps", "tactile_valid", "future_tactile_maps", "future_tactile_valid", "tactile_layout_present")
    }
    sample.update(
        tactile_region_names=["right.gripper.jaw_a.surface", "left.gripper.jaw_a.surface"],
        tactile_codec="sharpa_deform_gray_v1",
    )
    out = collate_regions([sample], map_size=8, history=2, future_frames=2)
    assert out["tactile_region_ids"].tolist() == [[58, 64]]
    assert "tactile_surface_ids" not in out
    sample["tactile_surface_ids"] = [0, 0]
    with pytest.raises(ValueError, match="removed"):
        collate_regions([sample], map_size=8, history=2, future_frames=2)


def test_unique_region_loss_and_history_codes(models):
    _, _, model = models
    args = regional_loss_inputs(2)
    args["tactile_region_ids"][:] = 61
    args["tactile_region_ids"] = torch.tensor([[58, 64], [58, 64]])
    args["tactile_maps"][:, :, 1] = args["tactile_maps"][:, :, 0]
    condition = model._condition(
        args["tactile_maps"],
        args["tactile_valid"],
        2,
        region_ids=args["tactile_region_ids"],
        layout_present=args["tactile_layout_present"],
    )
    history = condition[0].reshape(2, 2, 2, 4, -1)
    assert not torch.equal(history[:, :, 0], history[:, :, 1])
    model.compute_loss(**args)["loss"].backward()
    assert model.touch_v2.region_embedding.weight.grad[[58, 64]].abs().sum() > 0


def test_online_history_and_sidecar_reject_duplicate_regions(tmp_path):
    import numpy as np

    from openwam.tactile.deform_store import AlignedDeformStore
    from openwam.tactile.regions import DeformHistory
    from tests.test_deform_data import sidecar

    with pytest.raises(ValueError, match="duplicate"):
        DeformHistory(["generic.surface"] * 2, size=8)
    names = ["right.gripper.jaw_a.surface", "left.gripper.jaw_a.surface"]
    history = DeformHistory(names, size=8)
    value = history.append(0, {names[1]: np.zeros((8, 8), dtype=np.uint8)})
    assert "tactile_surface_ids" not in value
    assert value["tactile_valid"][-1].tolist() == [False, True]
    assert value["tactile_region_ids"].tolist() == [58, 64]
    meta = tmp_path / "meta"
    meta.mkdir()
    info = meta / "info.json"
    info.write_text("{}")
    path = sidecar(tmp_path / "touch", info)
    document = json.loads(path.read_text())
    document["surface_ids"] = [0] * 10
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="removed"):
        AlignedDeformStore(path, tmp_path, size=8, fps=30)
    document.pop("surface_ids")
    document["region_names"] = ["generic.surface"] * 10
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate"):
        AlignedDeformStore(path, tmp_path, size=8, fps=30)


def test_old_registry_checkpoint_load_and_explicit_migration_preserve_outputs(models, tmp_path):
    _, _, model = models
    cfg = tiny_config()
    cfg["tactile"]["region_names"] = list(LEGACY_REGION_NAMES)
    old = build_architecture("touchwam_regions", cfg)
    old.set_dtype_device(torch.float32, torch.device("cpu"))
    open_gates(old)
    path = tmp_path / "old_regions.safetensors"
    old.save_checkpoint(str(path))
    same = copy.deepcopy(old)
    same.load_checkpoint(str(path))
    with pytest.raises(RuntimeError, match="metadata"):
        model.load_checkpoint(str(path))
    report = model.load_regional_registry_checkpoint(str(path))
    assert not report["optimizer_resumed"] and report["initialized_region_rows"] == list(range(12, 67))
    args = regional_inputs()
    for a, b in zip(old(**args), model(**args)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    bad = copy.deepcopy(model)
    bad.tactile_spec["future_stride"] += 1
    with pytest.raises(RuntimeError, match="identical settings"):
        bad.load_regional_registry_checkpoint(str(path))


def test_registry_migration_through_training_loader(models, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir

    cfg = tiny_config()
    cfg["tactile"]["region_names"] = list(LEGACY_REGION_NAMES)
    old = build_architecture("touchwam_regions", cfg)
    weights = tmp_path / "old.safetensors"
    old.save_checkpoint(str(weights))
    OmegaConf.save(
        OmegaConf.create(
            {
                "model": {
                    "architecture": {
                        "framework": "dual_system",
                        "variant": "touchwam_regions",
                        "tactile": old.tactile_spec,
                    },
                    "video_backbone": {},
                }
            }
        ),
        tmp_path / "config.yaml",
    )
    live_tactile = dict(tiny_config()["tactile"], region_names=list(REGION_NAMES))
    live = OmegaConf.create(
        {
            "model": {
                "architecture": {"framework": "dual_system", "variant": "touchwam_regions", "tactile": live_tactile},
                "video_backbone": {},
            },
            "training": {"initialize_touch_registry": True},
            "project": {"seed": 42},
        }
    )
    monkeypatch.setattr(
        "openwam.model.resolve_architecture_config",
        lambda c: SimpleNamespace(
            registry_name="touchwam_regions", params=dict(tiny_config(), tactile=dict(c.architecture.tactile))
        ),
    )
    outputs = []
    for seed in (31, 73):
        torch.manual_seed(seed)
        before = torch.get_rng_state().clone()
        loaded = build_architecture_from_ckpt_dir(
            str(weights), weights_required=True, override_cfg=copy.deepcopy(live)
        )[1]
        assert torch.equal(before, torch.get_rng_state())
        outputs.append(loaded)
    for name, value in outputs[0].state_dict().items():
        torch.testing.assert_close(value, outputs[1].state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="finetune"):
        build_architecture_from_ckpt_dir(str(weights), weights_required=False, override_cfg=live)


def test_previous_64_registry_requires_explicit_warmstart(models, tmp_path):
    from safetensors.torch import save_file

    _, _, model = models
    open_gates(model)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    key = "touch_v2.region_embedding.weight"
    state[key] = state[key][:64].clone()
    old_spec = dict(model.tactile_spec, region_names=list(PREVIOUS_REGION_NAMES))
    meta = dict(
        model.checkpoint_metadata,
        tactile_spec=json.dumps(old_spec, sort_keys=True),
        tactile_region_registry=PREVIOUS_REGISTRY_VERSION,
        tactile_region_registry_sha256=PREVIOUS_REGISTRY_SHA256,
    )
    path = tmp_path / "previous.safetensors"
    save_file(state, str(path), metadata=meta)
    with pytest.raises(RuntimeError, match="metadata"):
        model.load_checkpoint(str(path))
    inputs = regional_inputs()
    before = model(**inputs)
    report = model.load_regional_registry_checkpoint(str(path))
    assert report["copied_region_rows"] == list(range(64))
    assert report["initialized_region_rows"] == [64, 65, 66]
    assert not report["optimizer_resumed"] and not report["reset_gates"]
    assert "single arm" in report["gripper_policy"]
    for a, b in zip(before, model(**inputs)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for k, v in state.items():
        torch.testing.assert_close(model.state_dict()[k][:64] if k == key else model.state_dict()[k], v, rtol=0, atol=0)
    meta["tactile_region_registry_sha256"] = "wrong"
    save_file(state, str(path), metadata=meta)
    with pytest.raises(RuntimeError, match="hash"):
        model.load_regional_registry_checkpoint(str(path))


def test_runtime_rejects_deprecated_surface_ids(models):
    _, _, model = models
    args = regional_inputs()
    with pytest.raises(ValueError, match="removed"):
        model(**dict(args, tactile_surface_ids=torch.zeros_like(args["tactile_region_ids"])))
    args = regional_loss_inputs()
    with pytest.raises(ValueError, match="removed"):
        model.compute_loss(**dict(args, tactile_surface_ids=torch.zeros_like(args["tactile_region_ids"])))


def test_composed_region_roi_reaches_model_and_predictions(models):
    import numpy as np

    from openwam.tactile.composition import SensorTile

    _, _, model = models
    surface = SurfaceSpec(
        "composed/a",
        "right.gripper.jaw_a.surface",
        "sharpa_real",
        "sharpa_deform_gray_v1",
        "uint8_intensity",
        (8, 8),
        tiles=(SensorTile("a", (4, 4), (0, 0)), SensorTile("b", (4, 4), (4, 4))),
    )
    frame = SurfaceLayout("fixture", (surface,)).compose(
        {"a": np.zeros((4, 4), np.uint8), "b": np.full((4, 4), 255, np.uint8)}, dtype="uint8"
    )
    args = regional_inputs(1, batch=1)
    args["tactile_region_ids"][:] = 58
    args["tactile_maps"][:] = torch.from_numpy(frame["maps"][surface.region]).float() / 255
    spatial = torch.from_numpy(frame["spatial_valid"][surface.region])[None, None, None]
    args["tactile_spatial_valid"] = spatial
    condition, query, mask = model.prepare_tactile_generation(
        args["tactile_maps"],
        args["tactile_valid"],
        region_ids=args["tactile_region_ids"],
        layout_present=args["tactile_layout_present"],
        spatial_valid=spatial,
    )
    assert condition[6].all() and condition[4].sum() == 32
    # Decoder keeps both measured tiles and masks both unmeasured quadrants.
    result = model.decode_tactile_generation(torch.zeros_like(args["noisy_tactile"]), query, condition)
    maps = result["future_tactile_maps"]
    assert np.all(maps[..., :4, 4:] == 0) and np.all(maps[..., 4:, :4] == 0)
    assert np.all(maps[..., :4, :4] == 0.5) and np.all(maps[..., 4:, 4:] == 0.5)
