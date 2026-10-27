"""Frozen identities, native coverage, repeated surfaces and safe warm-starts."""

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
    assert REGISTRY_SHA256 == "15ec04d9d4bc5262f2aa9af960e58595ec3e4329d4f40f3b33e3c52d7c2464a4"
    assert len(REGIONS) == 64 and REGION_NAMES[:12] == LEGACY_REGION_NAMES
    assert [r.id for r in REGIONS] == list(range(64))
    assert json.loads((ROOT / "docs/touchwam/region_registry_v1.json").read_text()) == json.loads(
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
    assert [s.key for s in gripper.surfaces] == [(58, 0), (59, 0)]
    assert {s.representation for s in touchscale.surfaces} == {"touchscale_normal_force_v1"}
    assert {s.representation for s in gripper.surfaces} == {"univtac_indent_mm_v1"}


def test_new_hardware_uses_repeated_slots_without_expanding_registry():
    fields = dict(
        region="generic.surface",
        sensor="new_sensor",
        representation="new_physical_quantity",
        units="declared",
        native_shape=(7, 11),
        mapping_note="Unmapped non-anthropomorphic contact skin",
    )
    one = SurfaceSpec(native_key="pad/A", surface_id=0, **fields)
    two = SurfaceSpec(native_key="pad/B", surface_id=1, **fields)
    layout = SurfaceLayout("new_hardware", (one, two))
    assert layout.identity_fields()["tactile_surface_ids"] == [0, 1]
    assert layout.identity_fields()["tactile_region_names"] == ["generic.surface"] * 2
    with pytest.raises(ValueError, match="Duplicate region"):
        SurfaceLayout("bad", (one, SurfaceSpec(native_key="pad/B", **fields)))
    with pytest.raises(ValueError, match="mapping_note"):
        SurfaceSpec(native_key="unknown", **dict(fields, mapping_note=""))


@pytest.mark.parametrize("value", [-1, 65536, 1.5, True])
def test_invalid_surface_ids_rejected(value):
    with pytest.raises(ValueError, match="surface_id"):
        SurfaceSpec("pad", "right.index.tip", "sensor", "deform", "uint8", (8, 8), value)


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
    ids = torch.tensor([s.key[0] for s in layout.surfaces])
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


def test_repeated_surface_history_and_future_permute_together(models):
    _, _, model = models
    model.eval()
    open_gates(model)
    args = regional_inputs(2, batch=1)
    args["tactile_region_ids"][:] = 61
    with pytest.raises(ValueError, match="Duplicate"):
        model(**args)
    args["tactile_surface_ids"] = torch.tensor([[1, 2]])
    before = model(**args)
    swapped = dict(args)
    for key in ("tactile_maps", "tactile_valid"):
        swapped[key] = args[key].flip(2)
    for key in ("tactile_region_ids", "tactile_surface_ids", "tactile_layout_present"):
        swapped[key] = args[key].flip(1)
    swapped["noisy_tactile"] = args["noisy_tactile"].reshape(1, 2, 2, 4, 16).flip(2).flatten(1, 3)
    after = model(**swapped)
    for a, b in zip(before[:2], after[:2]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        before[2].reshape(1, 2, 2, 4, 16).flip(2), after[2].reshape(1, 2, 2, 4, 16), rtol=1e-5, atol=1e-6
    )
    # Instance codes must actually distinguish otherwise-identical surfaces.
    condition, _, mask = model.prepare_tactile_generation(
        args["tactile_maps"],
        args["tactile_valid"],
        region_ids=args["tactile_region_ids"],
        layout_present=args["tactile_layout_present"],
        surface_ids=args["tactile_surface_ids"],
    )
    noise = model.sample_tactile_noise(condition, mask, 42).reshape(1, 2, 2, 4, 16)
    assert not torch.equal(noise[:, :, 0], noise[:, :, 1])
    swapped_condition, _, swapped_mask = model.prepare_tactile_generation(
        swapped["tactile_maps"],
        swapped["tactile_valid"],
        region_ids=swapped["tactile_region_ids"],
        layout_present=swapped["tactile_layout_present"],
        surface_ids=swapped["tactile_surface_ids"],
    )
    torch.testing.assert_close(
        noise.flip(2), model.sample_tactile_noise(swapped_condition, swapped_mask, 42).reshape_as(noise), rtol=0, atol=0
    )


@pytest.mark.parametrize("merge", [False, True])
def test_repeated_surface_generation_keeps_ids(models, merge):
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
        tactile_region_ids=torch.tensor([[58, 58]]),
        tactile_surface_ids=torch.tensor([[0, 1]]),
        tactile_layout_present=args["tactile_layout_present"],
    )
    assert result["tactile_region_names"] == ["gripper.jaw_a.surface"] * 2
    assert result["tactile_surface_ids"].tolist() == [0, 1]
    assert result["future_tactile_maps"].shape == (2, 2, 1, 8, 8)


def test_collate_repeated_surfaces_and_reject_duplicates():
    args = regional_loss_inputs(2)
    sample = {
        k: v[0]
        for k, v in args.items()
        if k
        in ("tactile_maps", "tactile_valid", "future_tactile_maps", "future_tactile_valid", "tactile_layout_present")
    }
    sample.update(
        tactile_region_names=["generic.surface"] * 2, tactile_surface_ids=[0, 1], tactile_codec="sharpa_deform_gray_v1"
    )
    out = collate_regions([sample], map_size=8, history=2, future_frames=2)
    assert out["tactile_region_ids"].tolist() == [[61, 61]]
    assert out["tactile_surface_ids"].tolist() == [[0, 1]]
    sample["tactile_surface_ids"] = [0, 0]
    with pytest.raises(ValueError, match="Duplicate"):
        collate_regions([sample], map_size=8, history=2, future_frames=2)


def test_repeated_surface_loss_and_history_codes(models):
    _, _, model = models
    args = regional_loss_inputs(2)
    args["tactile_region_ids"][:] = 61
    args["tactile_surface_ids"] = torch.tensor([[0, 1], [0, 1]])
    args["tactile_maps"][:, :, 1] = args["tactile_maps"][:, :, 0]
    condition = model._condition(
        args["tactile_maps"],
        args["tactile_valid"],
        2,
        region_ids=args["tactile_region_ids"],
        layout_present=args["tactile_layout_present"],
        surface_ids=args["tactile_surface_ids"],
    )
    history = condition[0].reshape(2, 2, 2, 4, -1)
    assert not torch.equal(history[:, :, 0], history[:, :, 1])
    model.compute_loss(**args)["loss"].backward()
    assert model.touch_v2.region_embedding.weight.grad[61].abs().sum() > 0


def test_online_history_and_sidecar_keep_native_surface_identity(tmp_path):
    import numpy as np

    from openwam.tactile.deform_store import AlignedDeformStore
    from openwam.tactile.regions import DeformHistory
    from tests.test_deform_data import sidecar

    history = DeformHistory(["generic.surface"] * 2, surface_ids=[0, 1], surface_keys=["pad_a", "pad_b"], size=8)
    value = history.append(0, {"pad_b": np.zeros((8, 8), dtype=np.uint8)})
    assert value["tactile_surface_ids"].tolist() == [0, 1]
    assert value["tactile_valid"][-1].tolist() == [False, True]
    assert value["tactile_region_ids"].tolist() == [61, 61]
    meta = tmp_path / "meta"
    meta.mkdir()
    info = meta / "info.json"
    info.write_text("{}")
    path = sidecar(tmp_path / "touch", info)
    document = json.loads(path.read_text())
    document["region_names"] = ["generic.surface"] * 10
    document["surface_ids"] = list(range(10))
    path.write_text(json.dumps(document))
    store = AlignedDeformStore(path, tmp_path, size=8, fps=30)
    sample = store.window(1, 0, 0, 40, 0, history=4, future_frames=2, future_stride=3)
    assert sample["tactile_surface_ids"].tolist() == list(range(10))
    document["surface_ids"][1] = 0
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="identities"):
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
    assert not report["optimizer_resumed"] and report["initialized_region_rows"] == list(range(12, 64))
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
