"""Every cluster replica owns a unique part of the original population."""

import importlib.util
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).resolve().parents[2] / "scripts/data/native/cluster_sharpa_cpu.py"
    spec = importlib.util.spec_from_file_location("sharpa_cpu_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replicas_partition_before_filtering_completed_work():
    module = launcher()
    population = list(range(495))
    completed = set(population[::3])
    seen = set()
    for rank in range(4):
        actual, _, command = module.shard_command({"RANK": str(rank), "WORLD_SIZE": "4",
                                                  "TOUCHWAM_SHARPA_WORKERS": "4"})
        assert actual == rank and "--pending-only" in command
        index = int(command[command.index("--shard-index") + 1])
        count = int(command[command.index("--shard-count") + 1])
        selected = set(population[index::count]) - completed
        assert not seen & selected
        seen |= selected
    assert seen == set(population) - completed


@pytest.mark.parametrize("env", [{}, {"RANK": "4", "WORLD_SIZE": "4"},
                                  {"RANK": "0", "WORLD_SIZE": "1", "TOUCHWAM_SHARPA_WORKERS": "0"}])
def test_missing_or_invalid_replica_identity_fails(env):
    with pytest.raises(ValueError):
        launcher().shard_command(env)


def test_master_waits_for_worker_receipt(tmp_path, monkeypatch):
    module = launcher()
    waited = []

    def worker_finishes(seconds):
        waited.append(seconds)
        assert module.finish_replica(1, 2, tmp_path, "job-current", 0) == 0

    monkeypatch.setattr(module.time, "sleep", worker_finishes)
    assert module.finish_replica(0, 2, tmp_path, "job-current", 0) == 0
    assert len(waited) == 1


def test_old_job_receipts_cannot_complete_new_job(tmp_path):
    module = launcher()
    module.finish_replica(1, 2, tmp_path, "job-old", 0)
    with pytest.raises(TimeoutError, match="refusing early success"):
        module.finish_replica(0, 2, tmp_path, "job-current", 0, timeout=0)


def test_worker_failure_is_not_reported_as_success(tmp_path):
    module = launcher()
    module.finish_replica(1, 2, tmp_path, "job-current", 7)
    assert module.finish_replica(0, 2, tmp_path, "job-current", 0) == 7


def test_wrong_replica_population_fails(tmp_path):
    module = launcher()
    module.finish_replica(1, 3, tmp_path, "job-current", 0)
    with pytest.raises(ValueError, match="different replica population"):
        module.finish_replica(0, 2, tmp_path, "job-current", 0)
