import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.data.ftp1 import freeze_training_subset as freeze


def make_snapshot(tmp_path):
    source = tmp_path / "producer"
    bucket = source / "FreeTacMan" / "abc"
    freeze.atomic(bucket / "meta/info.json", dict(codebase_version="v3.0", fps=30, total_episodes=2,
                                                total_frames=3, splits={"train": "0:1"}))
    freeze.atomic(bucket / "meta/source_definition.json", dict(source_id="ftp1/FreeTacMan", training_ready=False))
    freeze.atomic(bucket / "meta/action_contract.pending.json", dict(
        schema="touchwam_robot_action_v1", source_id="ftp1/FreeTacMan", profile_sha256="a" * 64,
        action=dict(slots=[34, 37], units=["source_native_length", "1"]),
        verification=dict(status="pending_source_normalization", evidence=["published pose convention"])))
    (bucket / "data").mkdir()
    pq.write_table(pa.table({"episode_index": [0, 0, 1], "observation.state": [[1., .5], [3., 1.], [100., 10.]],
                            "observation.state.valid": [[True, True], [True, True], [True, True]]}),
                   bucket / "data/file.parquet")
    (bucket / "videos").mkdir()
    (bucket / "videos/native.mp4").write_bytes(b"immutable-media")
    artifact_paths = ["meta/info.json", "meta/source_definition.json", "meta/action_contract.pending.json"]
    freeze.atomic(bucket / "meta/openwam_adapter_status.json", dict(
        complete=True, action_status="awaiting_source_normalization", training_ready=False,
        artifact_sha256={name: freeze.sha(bucket / name) for name in artifact_paths}))
    entries = [dict(path=p.relative_to(bucket).as_posix(), bytes=p.stat().st_size, sha256=freeze.sha(p))
               for p in sorted(bucket.rglob("*")) if p.is_file()]
    digest = hashlib.sha256((json.dumps(entries, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()).hexdigest()
    freeze.atomic(bucket / "_PFS_PUBLICATION.json", dict(schema="touchwam.ftp1.pfs_publication.v1",
        complete=True, identity=dict(source="FreeTacMan", key="abc"), files=entries, file_count=len(entries),
        payload_bytes=sum(e["bytes"] for e in entries), publication_id=digest))
    snapshot = tmp_path / "snapshot.json"
    freeze.atomic(snapshot, dict(root=str(source), sources={"FreeTacMan": dict(buckets=1,
        buckets_snapshot=[dict(bucket="abc", info_sha256=freeze.sha(bucket / "meta/info.json"), episodes=2, frames=3)])}))
    return snapshot, bucket, tmp_path / "frozen"


def test_freezes_explicit_population_without_copying_payload(tmp_path):
    snapshot, original, output = make_snapshot(tmp_path)
    freeze.atomic(original.parent / "later/meta/info.json", {"new": True})
    result = freeze.freeze(snapshot, output)
    assert result["sources"]["FreeTacMan"]["buckets"] == 1
    assert not (output / "FreeTacMan/later").exists()
    assert os.path.samefile(original / "videos/native.mp4", output / "FreeTacMan/abc/videos/native.mp4")
    assert os.path.samefile(original / "data/file.parquet", output / "FreeTacMan/abc/data/file.parquet")
    assert not os.path.samefile(original / "meta/info.json", output / "FreeTacMan/abc/meta/info.json")
    assert freeze.freeze(snapshot, output) == result


def test_changed_snapshot_or_incomplete_publication_is_rejected(tmp_path):
    snapshot, original, output = make_snapshot(tmp_path)
    manifest = freeze.read(original / "_PFS_PUBLICATION.json")
    manifest["complete"] = False
    freeze.atomic(original / "_PFS_PUBLICATION.json", manifest)
    with pytest.raises(ValueError, match="Incomplete"):
        freeze.freeze(snapshot, output)


def test_train_only_exact_statistics_and_private_contracts(tmp_path):
    snapshot, original, output = make_snapshot(tmp_path)
    freeze.freeze(snapshot, output)
    result = freeze.finalize(output, "FreeTacMan")
    stats = np.load(output / "FreeTacMan/normalization_stats.npy", allow_pickle=True).item()
    assert result["train_frames"] == 2
    assert stats["eef"]["min"][34] == 1 and stats["eef"]["max"][34] == 3
    assert stats["eef"]["mean"][34] == 2
    # rot6d preserves identity even when the empirical observed range differs.
    assert stats["eef"]["min"][37] == -1 and stats["eef"]["max"][37] == 1
    assert not (original / "meta/action_contract.json").exists()
    assert freeze.read(original / "meta/source_definition.json")["training_ready"] is False
    assert freeze.finalize(output, "FreeTacMan") == result


def test_interrupted_metadata_finalization_resumes_and_rejects_other_edits(tmp_path, monkeypatch):
    snapshot, original, output = make_snapshot(tmp_path)
    freeze.freeze(snapshot, output)
    real_atomic = freeze.atomic
    interrupted = False

    def fail_once(path, value):
        nonlocal interrupted
        if Path(path).name == "source_definition.json" and not interrupted:
            interrupted = True
            raise OSError("interrupted")
        return real_atomic(path, value)

    monkeypatch.setattr(freeze, "atomic", fail_once)
    with pytest.raises(OSError, match="interrupted"):
        freeze.finalize(output, "FreeTacMan")
    assert (output / "FreeTacMan/normalization_intent.json").exists()
    assert freeze.finalize(output, "FreeTacMan")["complete"]


def test_archive_only_profiles_never_receive_verified_contract(tmp_path):
    snapshot, original, output = make_snapshot(tmp_path)
    freeze.freeze(snapshot, output)
    status_path = output / "FreeTacMan/abc/meta/openwam_adapter_status.json"
    status = freeze.read(status_path)
    status["action_status"] = "pending_verified_action_profile"
    freeze.atomic(status_path, status)
    result = freeze.finalize(output, "FreeTacMan")
    assert result["training_ready"] is False
    assert result["status"] == "archive_only"
    assert not (output / "FreeTacMan/abc/meta/action_contract.json").exists()


def make_bound_publication(tmp_path):
    snapshot, original, output = make_snapshot(tmp_path)
    freeze.freeze(snapshot, output)
    normalized = freeze.finalize(output, "FreeTacMan")
    root = output / "FreeTacMan"
    freeze.atomic(root / "reader_validation.json", {"abc": {"status": "passed"}})
    freeze.atomic(root / "abc/meta/tactile_rgb.json", {"regions": ["right.gripper.jaw_a.surface"]})
    freeze.atomic(root / "abc/meta/episodes/episode-index.json", {"episodes": [0, 1]})
    files = [root / p for p in ("normalization_stats.npy", "normalization_complete.json", "reader_validation.json")]
    files.extend(p for p in (root / "abc/meta").rglob("*") if p.is_file())
    record = dict(schema=freeze.SCHEMA, source="FreeTacMan", status="complete", ready=True, training_ready=True,
                  buckets_expected=1, buckets_completed=1, episodes=2, frames=3,
                  training_modality="video_action_tactile", snapshot_sha256=freeze.sha(output / "snapshot.json"),
                  publication_inventory_sha256=freeze.sha(output / "publication_inventory.json"),
                  normalization_sha256=normalized["normalization_sha256"],
                  reader_validation_sha256=freeze.sha(root / "reader_validation.json"),
                  metadata_sha256={p.relative_to(root).as_posix(): freeze.sha(p) for p in files})
    freeze.atomic(root / "publication.json", record)
    return root


@pytest.mark.parametrize("relative", ["abc/meta/action_contract.json", "abc/meta/source_definition.json",
                                    "abc/meta/tactile_rgb.json", "abc/meta/episodes/episode-index.json",
                                    "normalization_stats.npy"])
def test_training_admission_rejects_changed_frozen_metadata(tmp_path, relative):
    from openwam.dataloader.publication import publication_contract
    root = make_bound_publication(tmp_path)
    assert publication_contract(root)["frames"] == 3
    path = root / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen FTP metadata changed"):
        publication_contract(root)


def test_training_admission_rejects_changed_snapshot_and_added_metadata(tmp_path):
    from openwam.dataloader.publication import publication_contract
    root = make_bound_publication(tmp_path)
    freeze.atomic(root / "abc/meta/new.json", {"unexpected": True})
    with pytest.raises(ValueError, match="metadata population changed"):
        publication_contract(root)
    (root / "abc/meta/new.json").unlink()
    path = root.parent / "snapshot.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="snapshot/publication inventory changed"):
        publication_contract(root)


def test_validation_cache_identity_binds_all_metadata_and_resolved_tactile_spec(tmp_path):
    snapshot, original, output = make_snapshot(tmp_path)
    freeze.freeze(snapshot, output)
    bucket = output / "FreeTacMan/abc"
    spec = {"map_size": 32, "observation_encoder": {"kind": "ftp1_heterogeneous_v1"}}
    identity = freeze.validation_identity(bucket, spec)
    assert freeze.validation_identity(bucket, dict(spec)) == identity
    freeze.atomic(bucket / "meta/tactile_rgb.json", {"changed_roi": [0, 0, 10, 10]})
    assert freeze.validation_identity(bucket, spec) != identity
    identity = freeze.validation_identity(bucket, spec)
    assert freeze.validation_identity(bucket, {**spec, "map_size": 64}) != identity
    path = bucket / "meta/source_definition.json"
    path.write_text(path.read_text() + " ")
    assert freeze.validation_identity(bucket, spec) != identity


def test_legacy_cache_migration_requires_unchanged_sealed_metadata_and_model(tmp_path):
    root = make_bound_publication(tmp_path)
    repo = tmp_path / "repo"
    model = repo / "configs/model/tactile.yaml"
    model.parent.mkdir(parents=True)
    model.write_text("map_size: 32\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "model baseline"], cwd=repo, check=True)
    result = freeze.read(root / "reader_validation.json")
    spec = {"map_size": 32}
    assert freeze.legacy_cache_proof(root, repo, result, spec, "HEAD")
    changed = {"abc": {**result["abc"], "extra": "unaccepted"}}
    with pytest.raises(ValueError, match="accepted sample result changed"):
        freeze.legacy_cache_proof(root, repo, changed, spec, "HEAD")
    model.write_text("map_size: 64\n")
    with pytest.raises(subprocess.CalledProcessError):
        freeze.legacy_cache_proof(root, repo, result, spec, "HEAD")
    model.write_text("map_size: 32\n")
    path = root / "abc/meta/tactile_rgb.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="accepted metadata changed"):
        freeze.legacy_cache_proof(root, repo, result, spec, "HEAD")
