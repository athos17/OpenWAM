"""Composed sim/real datasets (scripts/sim2real/build_mix_datasets.py) and cache assembly
(scripts/precompute/build_encode_cache.py assemble).

Needs a converted Marvin bucket at ``$OPENWAM_TEST_MARVIN_DIR`` (default
``data/pick_place_2_3_openwam_v3``), used as both the "real" and the "sim" source;
skipped otherwise. Source cache rows are synthetic and encode their own row number.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from openwam.dataloader.utils import encode_cache as ec
from tests.dataloader.test_encode_cache import _write_cache

MARVIN_DIR = Path(os.environ.get("OPENWAM_TEST_MARVIN_DIR", "data/pick_place_2_3_openwam_v3")).resolve()
pytestmark = pytest.mark.skipif(not (MARVIN_DIR / "meta" / "info.json").is_file(), reason="Marvin bucket not available")
ROOT = Path(__file__).resolve().parents[2]
ROW = (4, 2)
OVERRIDES = ["dataloader=marvin", "dataloader.color_jitter.enabled=false"]


def _row(r: int) -> torch.Tensor:
    bits = torch.tensor([r % 30000, r // 30000, 7, 7, 7, 7, 7, 7], dtype=torch.int16).reshape(ROW)
    return bits.view(torch.bfloat16)


def _decode(row: torch.Tensor) -> int:
    bits = row.view(torch.int16).reshape(-1)
    return int(bits[0]) + 30000 * int(bits[1])


def _dataset(dataset_dir: Path):
    sys.path.insert(0, str(ROOT / "scripts" / "precompute"))
    import build_encode_cache as bec

    return bec, bec.build_dataset(bec.compose_cfg(OVERRIDES + [f"dataloader.dataset_dir={dataset_dir}"]))


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "scripts" / "sim2real"))
    import build_mix_datasets

    out = tmp_path_factory.mktemp("mix") / "root"
    build_mix_datasets.main(["--real", str(MARVIN_DIR), "--sim", str(MARVIN_DIR), "--out-root", str(out),
                             "--n-real", "4", "--seed", "1"])
    return out


def test_composed_datasets_share_stats_and_map_to_sources(mixed):
    stats = (mixed / "normalization_stats.npy").read_bytes()
    comp = json.loads((mixed / "composition.json").read_text())
    assert len(comp["sampled_real_episodes"]) == 4 and not set(comp["sampled_real_episodes"]) & set(comp["holdout_real_episodes"])
    for name, n_eps in (("sim87", 87), ("sim87_real4", 91), ("real4", 4)):
        root = mixed / name
        assert (root / "meta/normalization_stats.npy").read_bytes() == stats
        eps = json.loads((root / "meta/composition.json").read_text())["episodes"]
        assert [e["episode_index"] for e in eps] == list(range(n_eps))
        info = json.loads((root / "meta/info.json").read_text())
        assert info["total_episodes"] == n_eps and info["total_frames"] == sum(e["length"] for e in eps)
        head = root / "videos/observation.images.left_head/chunk-000/file-000.mp4"
        src = Path(eps[0]["source_dir"]) / f"videos/observation.images.left_head/chunk-000/file-{eps[0]['source_episode_index']:03d}.mp4"
        assert os.path.samefile(head, src)  # hard link, not a copy


def test_assemble_copies_the_matching_source_rows(mixed, tmp_path):
    bec, source = _dataset(MARVIN_DIR)
    src_cache = tmp_path / "src_latent"
    src_cache.mkdir()
    np.save(src_cache / "windows.npy", ec.window_table(source))
    _write_cache(src_cache, "latent", source, ROW, _row, {"windows.npy": None, "_rows": range(len(source))},
                 ec.latent_signature(source), "vae-fp")
    meta = json.loads((src_cache / "meta.json").read_text())
    meta["batch"] = 48  # real caches record the encode batch; assemble requires agreement
    meta["manifest_sha256"] = ec.manifest_digest(meta)
    (src_cache / "meta.json").write_text(json.dumps(meta))
    ec.verify_cache(src_cache, workers=2)

    target_dir = mixed / "sim87_real4"
    out = tmp_path / "assembled"
    argv = sys.argv
    sys.argv = ["build_encode_cache.py", "assemble", "--out", str(out), "--kinds", "latent", "--num-shards", "3",
                "--source", f"{MARVIN_DIR}={src_cache}", "--", *OVERRIDES, f"dataloader.dataset_dir={target_dir}"]
    try:
        bec.main()
    finally:
        sys.argv = argv
    ec.verify_cache(out / "latent", workers=2)

    _, target = _dataset(target_dir)
    cache = ec.LatentCache(out / "latent", target)  # signature + window table of the composed dataset
    comp = {e["episode_index"]: e for e in json.loads((target_dir / "meta/composition.json").read_text())["episodes"]}
    src_rows = {(int(source._eps_df.episode_index.iloc[e]), int(o)): r for r, (e, o) in enumerate(ec.window_table(source).tolist())}
    table = ec.window_table(target)
    rng = np.random.default_rng(0)
    for i in [0, len(table) - 1, *rng.integers(0, len(table), 300).tolist()]:
        e, o = table[i]
        c = comp[int(target._eps_df.episode_index.iloc[e])]
        assert _decode(cache.get(int(i))) == src_rows[(c["source_episode_index"], int(o))]
