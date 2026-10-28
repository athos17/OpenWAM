#!/usr/bin/env python3
"""Resume Sharpa tactile extraction on disjoint CPU job replicas.

Every replica uses the same complete, sorted partition list and a unique
NODE_RANK/RANK. Filter completed manifests only AFTER assigning partitions.
Do not run overlapping helper jobs: shared-filesystem flock is not assumed to
coordinate different nodes. Original video/action files remain read-only.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def shard_command(environ, python=sys.executable):
    rank = int(environ.get("NODE_RANK", environ.get("RANK", "-1")))
    world = int(environ.get("WORLD_SIZE", "0"))
    workers = int(environ.get("TOUCHWAM_SHARPA_WORKERS", "2"))
    if world < 1 or not 0 <= rank < world or workers < 1:
        raise ValueError("Expected unique CPU replica rank, positive world size and worker count")
    root = Path(environ.get("TOUCHWAM_DATA_ROOT", "/mnt/pfs/v89atp/workspace/dataset/touchwam_pretrain_v1_20261028"))
    command = [python, str(Path(__file__).with_name("run_native_shards.py")),
               "--kind", "sharpa", "--output", str(root / "sharpadex"),
               "--shard-index", str(rank), "--shard-count", str(world),
               "--workers", str(workers), "--pending-only"]
    return rank, root, command


def finish_replica(rank, world, root, job_id, exit_code, *, timeout=3600, poll_seconds=5):
    """Keep master alive until every shard exits; the platform stops its peers.

    Receipts live under a unique scheduler job ID, so an earlier retry cannot
    satisfy this barrier. This is completion coordination, not task locking.
    """
    if not job_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError("A safe unique scheduler job ID is required for the completion barrier")
    folder = root / "logs/sharpa_jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rank{rank}.json"
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(dict(rank=rank, world=world, job_id=job_id, exit_code=exit_code)))
    temporary.replace(path)
    if rank != 0 or exit_code:
        return exit_code
    deadline = time.monotonic() + timeout
    while True:
        complete = 0
        for other in range(world):
            receipt = folder / f"rank{other}.json"
            if not receipt.is_file():
                continue
            status = json.loads(receipt.read_text())
            if any(status.get(k) != v for k, v in (("rank", other), ("world", world), ("job_id", job_id))):
                raise ValueError("Completion receipt belongs to a different replica population")
            if status["exit_code"]:
                return status["exit_code"]
            complete += 1
        if complete == world:
            return 0
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Only {complete}/{world} Sharpa replicas completed; refusing early success")
        time.sleep(poll_seconds)


def main():
    rank, root, command = shard_command(os.environ)
    world = int(os.environ["WORLD_SIZE"])
    job_id = os.environ.get("AIHC_JOB_ID")
    if not job_id:
        raise ValueError("AIHC_JOB_ID is required; do not reuse completion receipts across jobs")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"sharpa_resume_rank{rank}.log"
    print(json.dumps({"rank": rank, "command": command, "log": str(log_path)}), flush=True)
    with log_path.open("a", buffering=1) as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    code = finish_replica(rank, world, root, job_id, result.returncode)
    print(json.dumps({"rank": rank, "exit_code": code, "all_shards_complete": rank == 0 and code == 0}), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
