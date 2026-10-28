"""Same-sized but different Sharpa caches must not pass the training gate."""
import hashlib
import json

import pytest

from openwam.dataloader.publication import source_publication_contract


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    name = "task--season"
    info = dict(codebase_version="v3.0", total_episodes=1, total_frames=2)
    for root in (source, cache):
        partition = root / "partitions/task/season"
        write(partition / "meta/info.json", info)
        (root / "buckets").mkdir()
        (root / "buckets" / name).symlink_to(partition, target_is_directory=True)
        write(root / "publication.json", dict(status="complete", episodes=1, frames=2,
                                              buckets_expected=1, buckets_completed=1))
    bucket = source / "buckets" / name
    episode_path = bucket / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    episode_path.write_bytes(b"fixture episode metadata; this gate hashes rather than decodes it")
    manifest = cache / "buckets" / name / "tactile/manifest.json"
    write(manifest, {"episodes": {"0": {"frames": 2}}})
    entry = dict(source_info_sha256=sha(bucket / "meta/info.json"),
                 source_episode_metadata_sha256=hashlib.sha256(json.dumps([
                     (str(episode_path.relative_to(bucket)), sha(episode_path))]).encode()).hexdigest(),
                 manifest_sha256=sha(manifest))
    catalog = dict(status="complete", pending=[], source_manifest_sha256=sha(source / "publication.json"),
                   bucket_manifest_by_name={name: entry})
    write(cache / "tactile_catalog.json", catalog)
    bind_catalog(cache)
    return source, cache, bucket, manifest


def bind_catalog(cache):
    path = cache / "publication.json"
    pub = json.loads(path.read_text())
    pub.update(schema="touchwam_sharpa_external_tactile_catalog_v1",
               tactile_catalog_sha256=sha(cache / "tactile_catalog.json"))
    write(path, pub)


def contract(source, cache):
    return source_publication_contract(dict(dataset_dir=str(source / "buckets"), tactile_root=str(cache)))


def test_binding_accepts_exact_source_and_records_catalog_digest(tmp_path):
    source, cache, _, _ = fixture(tmp_path)
    binding = contract(source, cache)["tactile"]["binding"]
    assert binding["buckets_verified"] == 1
    assert binding["catalog_sha256"] == sha(cache / "tactile_catalog.json")


@pytest.mark.parametrize("change", ["source", "info", "episodes", "manifest", "catalog", "names", "pending", "alias"])
def test_binding_rejects_equal_count_mismatches(tmp_path, change):
    source, cache, bucket, manifest = fixture(tmp_path)
    if change in ("source", "info", "episodes", "manifest", "catalog"):
        path = {"source": source / "publication.json", "info": bucket / "meta/info.json",
                "episodes": bucket / "meta/episodes/chunk-000/file-000.parquet",
                "manifest": manifest, "catalog": cache / "tactile_catalog.json"}[change]
        with path.open("ab") as f:
            f.write(b" ")
    elif change in ("names", "pending"):
        path = cache / "tactile_catalog.json"
        document = json.loads(path.read_text())
        if change == "names":
            document["bucket_manifest_by_name"]["wrong--season"] = document["bucket_manifest_by_name"].pop("task--season")
        else:
            document["pending"] = ["task--season"]
        write(path, document)
        bind_catalog(cache)
    else:
        alias = cache / "buckets/task--season"
        other = cache / "partitions/other/season"
        write(other / "meta/info.json", json.loads((bucket / "meta/info.json").read_text()))
        write(other / "tactile/manifest.json", json.loads(manifest.read_text()))
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        contract(source, cache)
