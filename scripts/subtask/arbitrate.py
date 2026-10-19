"""Automatic arbitration of segments the two labelers disagree on (replaces manual review).

Selects rule segments that are kept under ``vocab.py --trust`` but carry ``disagree_object``
or ``disagree_success``. Each judge model sees a dense strip (10 head frames + 5 wrist
frames around the grasp) plus the wrist grasp/lift keyframes and must choose between the
two candidate names (or name another object / say none). Judges that agree decide the
segment; otherwise it is excluded. Writes ``<out>/overrides_auto.csv`` (manual
``overrides.csv`` rows take precedence in vocab.py).

Example:
  python scripts/subtask/arbitrate.py --dataset data/pick_place_2_3_openwam_v3 --out outputs/subtask
  python scripts/subtask/vocab.py --dataset data/pick_place_2_3_openwam_v3 --out outputs/subtask --trust gpt-6-astra
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARBITRATION_KIND, episode_table, label_path  # noqa: E402
from label import COLORS, _CAT_TEXT, cached, call, object_name  # noqa: E402
from review_app import dense_strip  # noqa: E402

TASK = """A robot's {hand} hand picked something up in this segment. Two annotators disagree.
Image 1 is a strip: rows 1-2 are 10 head-camera frames across the whole segment (frame numbers
top-left; the robot's left arm is on the image left), row 3 shows 5 {hand} wrist-camera frames
from 2 s before the grasp to the release. Images 2-3 are the {hand} wrist camera 1.3 s and 0.7 s
before the grasp, larger (small objects such as coffee capsules are visible there and hidden later).
Candidate A: "{a}"   Candidate B: "{b}"
Object categories:
{categories}
Colors: {colors}
Decide what the {hand} hand carried, and whether it ended inside the bin.
Return JSON: {{"choice": "A"|"B"|"other"|"none",
  "category": str|null, "other_category": str|null, "color": str|null,   // only for "other"
  "lifted": bool, "placed_in_bin": bool, "reason": str, "confidence": float}}"""

DECIDING = {"disagree_object", "disagree_success"}
EXCLUDED = {"no_object", "failed", "not_in_bin", "multiple_objects", "short", "not_idle",
            "trust_low_conf", "trust_hand_mismatch", "manual_exclude"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/subtask"))
    ap.add_argument("--judges", nargs="+", default=["gpt-6-astra", "gpt-6.1-sol"])
    ap.add_argument("--trust", default="gpt-6-astra", help="the model vocab.py --trust used (candidate A)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    out = args.out
    for var in ("LLM_API_KEY", "LLM_BASE_URL"):
        if not os.environ.get(var):
            sys.exit(f"missing environment variable {var}")

    # Select from labels computed WITHOUT earlier automatic verdicts (they would hide the
    # disagreement flags); manual overrides.csv rows still apply and mark segments reviewed.
    auto = out / "overrides_auto.csv"
    if auto.exists():
        auto.unlink()
    import vocab
    vocab.main(["--dataset", str(args.dataset), "--out", str(out), "--trust", args.trust])
    labels = pd.read_parquet(out / "segment_labels.parquet")
    rules = pd.read_parquet(out / "segments_rules.parquet").set_index(["episode_index", "seg_id"])
    kf = pd.read_parquet(out / "keyframes.parquet").set_index(["episode_index", "seg_id", "camera", "tag"]).path
    eps = episode_table(args.dataset).set_index("episode_index", drop=False)

    todo = []
    for r in labels[labels.type == "PICK"].itertuples():
        flags = set(f for f in r.flags.split(",") if f)
        if flags & DECIDING and not flags & EXCLUDED and "reviewed" not in flags:
            todo.append(r)
    print(f"{len(todo)} segments to arbitrate")

    # dense strips (shared with review_app.py; built for any segment missing one)
    jobs, items = defaultdict(list), {}
    for r in todo:
        rule = rules.loc[(r.episode_index, r.seg_id)]
        ev = json.loads(rule.events)
        it = {"seg": int(r.seg_id), "type": "PICK", "hand": rule.hand, "start": int(rule.start), "end": int(rule.end),
              "grasp": ev["grasp"], "release": ev["release"]}
        items[(r.episode_index, r.seg_id)] = it
        if not (out / f"review_app/dense/ep{r.episode_index:03d}_{int(rule.start):05d}.jpg").exists():
            jobs[int(r.episode_index)].append(it)
    if jobs:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(dense_strip, [(args.dataset, out, eps.loc[ep], v) for ep, v in jobs.items()]))

    calls = []
    for r in todo:
        rule = rules.loc[(r.episode_index, r.seg_id)]
        wrist = "lwrist" if rule.hand == "left" else "rwrist"
        imgs = [out / f"review_app/dense/ep{r.episode_index:03d}_{int(rule.start):05d}.jpg",
                out / kf[(r.episode_index, r.seg_id, wrist, "pre40")], out / kf[(r.episode_index, r.seg_id, wrist, "pre20")]]
        a, b = r.object, r.object_secondary
        text = TASK.format(hand=rule.hand, a=a, b=b, categories=_CAT_TEXT, colors=", ".join(COLORS))
        seg = rule.copy()
        seg["episode_index"], seg["seg_id"] = r.episode_index, r.seg_id
        for j in args.judges:
            path = label_path(out, j, pd.Series(seg), kind=ARBITRATION_KIND)
            calls.append((r, j, path, (lambda t=text, i=imgs, j=j: call(j, t, i))))
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(cached, p, fn): (r.episode_index, r.seg_id, j) for r, j, p, fn in calls}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                results[futs[f]] = f.result()[0]["result"]
            except Exception as e:  # noqa: BLE001 - failed judge -> segment excluded below
                print("FAILED", futs[f], str(e)[:150])
            if k % 20 == 0 or k == len(futs):
                print(f"{k}/{len(futs)} judge calls  {time.time() - t0:.0f}s", flush=True)

    rows, verdicts = [], Counter()
    for r in todo:
        rule = rules.loc[(r.episode_index, r.seg_id)]
        votes = []
        for j in args.judges:
            v = results.get((r.episode_index, r.seg_id, j))
            if v is None:
                votes.append(None)
                continue
            ch = str(v.get("choice", "")).strip().lower()
            name = {"a": r.object, "b": r.object_secondary}.get(ch)
            if ch == "other":
                name = object_name(v) if v.get("category") else None
            ok = bool(v.get("lifted")) and bool(v.get("placed_in_bin")) and ch != "none" and name not in (None, "", "NONE", "none")
            votes.append(name if ok else "EXCLUDE")
        frame = json.loads(rule.events)["grasp"]
        if None not in votes and len(set(votes)) == 1 and votes[0] != "EXCLUDE":
            rows.append([r.episode_index, frame, votes[0], "true", "true", "false", f"auto-arbitration: {votes[0]}"])
            verdicts["keep:" + ("A" if votes[0] == r.object else "B" if votes[0] == r.object_secondary else "other")] += 1
        else:
            why = "judges excluded" if set(votes) == {"EXCLUDE"} else f"judges split {votes}"
            rows.append([r.episode_index, frame, "", "", "", "true", f"auto-arbitration: {why}"])
            verdicts["exclude:" + ("both" if set(votes) == {"EXCLUDE"} else "split")] += 1
    with open(out / "overrides_auto.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_index", "frame", "object", "lifted", "placed", "exclude", "note"])
        w.writerows(rows)
    print(dict(verdicts), "->", out / "overrides_auto.csv")


if __name__ == "__main__":
    main()
