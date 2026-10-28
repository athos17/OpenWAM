#!/usr/bin/env python3
"""Four CPU replicas complete eight disjoint Sharpa extraction shards.

dev-cpu owns shard 0 of 9. This job never starts training or claims a GPU.
The scheduler injects NODE_RANK/RANK and WORLD_SIZE. Per-bucket locks and
converter receipts make retries safe without changing the data population.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    rank = int(os.environ.get("NODE_RANK", os.environ.get("RANK", "-1")))
    if int(os.environ.get("WORLD_SIZE", "0")) != 4 or not 0 <= rank < 4:
        raise RuntimeError("Expected four CPU replicas with unique node ranks")
    root = Path(os.environ.get("TOUCHWAM_DATA_ROOT", "/mnt/pfs/v89atp/workspace/dataset/touchwam_pretrain_v1_20261028"))
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    workers = []
    per_shard_workers = int(os.environ.get("TOUCHWAM_SHARPA_WORKERS_PER_SHARD", "1"))
    if per_shard_workers < 1:
        raise ValueError("TOUCHWAM_SHARPA_WORKERS_PER_SHARD must be positive")
    for shard in (2 * rank + 1, 2 * rank + 2):
        log = (logs / f"sharpa_cpu_shard{shard}.log").open("a", buffering=1)
        cmd = [sys.executable, str(Path(__file__).with_name("run_native_shards.py")),
               "--kind", "sharpa", "--output", str(root / "sharpadex"),
               "--shard-index", str(shard), "--shard-count", "9", "--workers", str(per_shard_workers)]
        workers.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    try:
        codes = [proc.wait() for proc, _ in workers]
        print(json.dumps({"node": rank, "exit_codes": codes}), flush=True)
        if any(codes):
            raise SystemExit(1)
    finally:
        for proc, log in workers:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()


if __name__ == "__main__":
    main()
