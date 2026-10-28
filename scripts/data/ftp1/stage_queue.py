#!/usr/bin/env python3
"""Bounded FTP staging producer and shared-PFS CPU conversion consumers.

The producer runs where /dataset is mounted. CPU workers need PFS only. Fixed
rank/hash partitioning ensures different nodes never write the same bucket.
POSIX flock additionally protects local duplicate processes and restarts; it is
not relied on for distributed scheduling. Completed receipts allow resumption.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import socket
import time
import traceback
from pathlib import Path

from convert import VERSION, atomic_json, convert_group, log
from stage import archive_identity, load_manifest, stage_group
from validate import validate_bucket

SCHEMA = "touchwam.ftp1.stage_queue.v2"


def key_for(group):
    return hashlib.sha256(group["group"].encode()).hexdigest()[:20]


def assigned_rank(key, worker_count):
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    return int(key, 16) % worker_count


def worker_identity(args, state):
    """Refuse launcher defaults that could make every node believe it is rank 0."""
    if state.get("schema") != SCHEMA:
        raise ValueError("Unknown/legacy stage queue schema; create a v2 queue before resuming")
    rank = args.rank if args.rank is not None else os.environ.get("RANK")
    world_size = args.world_size if args.world_size is not None else os.environ.get("WORLD_SIZE")
    if rank is None or world_size is None:
        raise ValueError("Explicit --rank/--world-size or launcher RANK/WORLD_SIZE are required")
    rank, world_size = int(rank), int(world_size)
    if world_size != state["worker_count"] or not 0 <= rank < world_size:
        raise ValueError("Worker rank/count differs from the fixed queue partition")
    if args.run_id != state["run_id"]:
        raise ValueError("Worker run identity differs from the queue")
    return rank, world_size


def bucket_for(output, source, group):
    task = group["group"].split("/", 1)[1].removesuffix(".zarr")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task).strip("_")
    return Path(output) / source / (safe + "-" + hashlib.sha256(task.encode()).hexdigest()[:8])


def reuse_completed(args, source, group):
    """Avoid copying a task already converted with this exact source/definition."""
    bucket = bucket_for(args.output, source, group)
    receipt = bucket / "meta/conversion_complete.json"
    if not receipt.exists():
        return None
    definition = json.loads((Path(args.definitions) / f"{source}.json").read_text())
    files = sorted(p for p in (Path(args.input) / source).iterdir() if ".tar" in p.name)
    _, fingerprint = archive_identity(files)
    signature = hashlib.sha256(
        json.dumps([VERSION, definition, fingerprint, group], sort_keys=True).encode()
    ).hexdigest()
    conversion = json.loads(receipt.read_text())
    if conversion.get("signature") != signature:
        raise ValueError(f"Existing conversion definition changed: {bucket}")
    validation = validate_bucket(bucket, Path(args.definitions) / "regions")
    return {
        "schema": SCHEMA,
        "group": group["group"],
        "conversion": conversion,
        "validation": validation,
        "completed_at": time.time(),
        "reused_without_staging": True,
    }


def producer(args):
    if args.worker_count < 1 or not args.run_id.strip():
        raise ValueError("A positive fixed worker_count and nonempty run_id are required")
    queue = Path(args.queue)
    queue.mkdir(parents=True, exist_ok=True)
    for sub in ["ready", "claims", "completed", "errors"]:
        (queue / sub).mkdir(exist_ok=True)
    jobs = [
        (s["dataset"], g)
        for s in json.loads(Path(args.census).read_text())
        if s["dataset"] in args.sources and s["dataset"] != "sharpa"
        for g in s["groups"]
        if g["frames"] > 0
    ]
    jobs.sort(key=lambda j: j[1]["end_offset"] - j[1]["offset"])
    identity = {
        "schema": SCHEMA,
        "worker_count": args.worker_count,
        "run_id": args.run_id,
        # Preserve the mounted path spelling: a login host may resolve a PFS
        # symlink to an alias that is not mounted inside cluster workers.
        "output": str(Path(args.output).absolute()),
        "definitions": str(Path(args.definitions).absolute()),
        "jobs": [{"source": s, "group": g} for s, g in jobs],
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state_path = queue / "state.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("fingerprint") != fingerprint:
        raise ValueError("Queue identity differs; use a new queue directory")
    with (queue / "producer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(
            identity,
            fingerprint=fingerprint,
            staging_complete=False,
            max_pending=args.max_pending,
            updated_at=time.time(),
        )
        atomic_json(state_path, state)
        for source, group in jobs:
            key = key_for(group)
            if (queue / "completed" / f"{key}.json").exists():
                continue
            existing = reuse_completed(args, source, group)
            if existing is not None:
                atomic_json(queue / "completed" / f"{key}.json", existing)
                log("queue_task_reused", source=source, group=group["group"])
                continue
            ready = queue / "ready" / f"{key}.json"
            if ready.exists():
                continue
            while True:
                pending = sum(not (queue / "completed" / p.name).exists() for p in (queue / "ready").glob("*.json"))
                if pending < args.max_pending:
                    break
                time.sleep(10)
            manifest = stage_group(args.input, args.staging_root, source, group, args.segment_mib * 1024**2)
            atomic_json(
                ready,
                {
                    "schema": SCHEMA,
                    "fingerprint": fingerprint,
                    "source": source,
                    "group": group,
                    "manifest": str(manifest.absolute()),
                },
            )
            log("queue_task_ready", source=source, group=group["group"], manifest=str(manifest))
        state.update(staging_complete=True, updated_at=time.time())
        atomic_json(state_path, state)
        log("queue_staging_complete", tasks=len(jobs), queue=str(queue))


def worker(args):
    queue = Path(args.queue)
    deadline = time.monotonic() + args.wait_timeout
    while not (queue / "state.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("No staging producer state arrived")
        time.sleep(5)
    state = json.loads((queue / "state.json").read_text())
    rank, world_size = worker_identity(args, state)
    own_keys = {key_for(j["group"]) for j in state["jobs"] if assigned_rank(key_for(j["group"]), world_size) == rank}
    identity = state["fingerprint"]
    while True:
        state = json.loads((queue / "state.json").read_text())
        if state["fingerprint"] != identity:
            raise ValueError("Stage queue changed during worker execution")
        processed = False
        for path in sorted((queue / "ready").glob("*.json")):
            key = path.stem
            if key not in own_keys:
                continue
            done = queue / "completed" / path.name
            if done.exists():
                continue
            with (queue / "claims" / f"{key}.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if done.exists():
                    continue
                row = json.loads(path.read_text())
                if row["fingerprint"] != identity or key_for(row["group"]) != key:
                    raise ValueError("Queue task identity mismatch")
                error_file = queue / "errors" / path.name
                previous = json.loads(error_file.read_text()) if error_file.exists() else {}
                if previous.get("attempts", 0) >= args.max_attempts:
                    continue
                atomic_json(
                    queue / "claims" / f"{key}.json",
                    {
                        "hostname": socket.gethostname(),
                        "pid": os.getpid(),
                        "rank": rank,
                        "worker_count": world_size,
                        "run_id": state["run_id"],
                        "started_at": time.time(),
                        "group": row["group"]["group"],
                    },
                )
                try:
                    # The converter verifies every staged segment and full range
                    # before indexing; no original archive path is dereferenced.
                    load_manifest(row["manifest"], verify=False)
                    definition = Path(state["definitions"]) / f"{row['source']}.json"
                    result = convert_group(
                        (
                            "/original-archive-not-mounted",
                            state["output"],
                            str(definition),
                            row["group"],
                            args.encoder_threads,
                            row["manifest"],
                        )
                    )
                    validation = validate_bucket(result["bucket"], Path(state["definitions"]) / "regions")
                    atomic_json(
                        done,
                        {
                            "schema": SCHEMA,
                            "group": row["group"]["group"],
                            "conversion": result,
                            "validation": validation,
                            "completed_at": time.time(),
                        },
                    )
                    if error_file.exists():
                        error_file.unlink()
                    log(
                        "queue_task_complete",
                        group=row["group"]["group"],
                        frames=result["frames"],
                        bucket=result["bucket"],
                    )
                    processed = True
                except Exception as exc:
                    error = {
                        "attempts": previous.get("attempts", 0) + 1,
                        "group": row["group"]["group"],
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "updated_at": time.time(),
                    }
                    atomic_json(error_file, error)
                    log("queue_task_error", **error)
                    time.sleep(5)
        done_keys = {p.stem for p in (queue / "completed").glob("*.json")}
        if state["staging_complete"] and own_keys <= done_keys:
            log("queue_worker_complete", rank=rank, tasks=len(own_keys))
            return
        exhausted = [
            json.loads(p.read_text())
            for p in (queue / "errors").glob("*.json")
            if p.stem in own_keys and json.loads(p.read_text()).get("attempts", 0) >= args.max_attempts
        ]
        if exhausted:
            raise RuntimeError(f"Queue has tasks exceeding retry budget: {exhausted}")
        if not processed:
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("producer")
    p.add_argument("--input", required=True)
    p.add_argument("--staging-root", required=True)
    p.add_argument("--queue", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--definitions", required=True)
    p.add_argument("--census", required=True)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--max-pending", type=int, default=2)
    p.add_argument("--segment-mib", type=int, default=1024)
    p.add_argument("--worker-count", type=int, required=True)
    p.add_argument("--run-id", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--queue", required=True)
    w.add_argument("--encoder-threads", type=int, default=2)
    w.add_argument("--max-attempts", type=int, default=3)
    w.add_argument("--wait-timeout", type=int, default=3600)
    w.add_argument("--rank", type=int, help="Defaults to launcher RANK; no implicit rank-0 fallback")
    w.add_argument("--world-size", type=int, help="Defaults to launcher WORLD_SIZE")
    w.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.mode == "producer":
        if args.max_pending < 1:
            parser.error("max-pending must be positive")
        producer(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
