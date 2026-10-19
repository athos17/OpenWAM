"""P1: multimodal LLM labels for rule segments (OpenAI-compatible chat API).

Two passes, every call cached as JSON under ``<out>/llm/`` (re-runs skip done calls):
  episode  one call per episode on the numbered overview sheet -> object inventory
           + a preliminary object per PICK (primary model only)
  segment  one call per segment and model on its keyframes, given the inventory

Credentials come from the environment: LLM_API_KEY, LLM_BASE_URL.

Example:
  python scripts/subtask/label.py --out outputs/subtask --pass episode --episodes 0,30,70
  python scripts/subtask/label.py --out outputs/subtask --pass segment --models gpt-6.1-sol gpt-6-astra
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import label_path  # noqa: E402

CONTEXT = (
    "You label teleoperated demonstrations of a bimanual robot with two dexterous five-finger hands. "
    "The robot puts tabletop objects into a black wire-mesh bin that stands on a white table. "
    "The head camera looks down at the table from behind the robot: the robot's LEFT arm is on the LEFT "
    "side of head images and its RIGHT arm on the right. Wrist cameras sit on the wrist and look at the "
    "palm and fingers. People may stand in the background or reach in to rearrange objects or the bin. "
    "Name objects as '<color> <category>' with plain common nouns (e.g. 'green sponge', 'yellow tape roll', "
    "'blue cup', 'water bottle', 'green tennis ball'); add a size or shape word only when two objects in "
    "the episode would otherwise share a name. Never name the bin as an object. Reply with JSON only."
)

EPISODE_TASK = """The image is an overview sheet. Each tile is the head-camera frame at the moment a hand
closes on an object; the tile label #k is the segment id. Hand per segment: {hands}.
Return:
{{"inventory": [{{"name": str, "category": str, "color": str, "note": str}}],
  "segments": [{{"seg_id": int, "object": str, "confidence": float}}]}}
"inventory" lists every distinct object picked or visible on the table during the episode (one entry per
physical object). Each "object" must be an inventory name. confidence is 0-1."""

# Fixed taxonomy: the final object name is assembled from (color, category) in code, so the
# same kind of object always gets the same name. Hints target known confusions in this data.
CATEGORIES = {
    "coffee capsule": "small (~3 cm) metal coffee pod with a colored domed top (red, purple, gold, ...); "
                      "it is usually completely hidden in the closed fingers - look at the wrist frames BEFORE the grasp",
    "bottle cap": "a loose plastic screw cap that is not attached to a bottle",
    "water bottle": "clear plastic bottle of water (blue or red cap); always call it water bottle",
    "drink bottle": "plastic bottle with a printed colored label or colored body (e.g. purple yogurt drink, red cola-like bottle); NOT a cup",
    "cup": "open plastic drinking cup without a lid",
    "can": "metal drink can",
    "jar": "container with a screw lid",
    "tape roll": "roll of adhesive tape", "tape measure": "retractable tape measure",
    "tennis ball": "fuzzy tennis ball", "ball": "any other ball", "sponge": "kitchen sponge / scrubber",
    "block": "toy cube or block", "die": "dice", "toy figure": "plastic animal / dinosaur / character figure",
    "toy car": "toy vehicle", "tissue pack": "soft pack of tissues", "tissue box": "box of tissues",
    "stapler": "stapler", "scissors": "scissors", "pipette": "plastic pipette / dropper / syringe",
    "spray bottle": "bottle with trigger sprayer", "spray nozzle": "detached sprayer head",
    "eraser": "eraser", "marker": "marker or pen", "brush": "brush or comb", "remote control": "remote control",
    "phone": "phone", "toy fruit": "toy fruit / vegetable", "cloth": "cloth or towel", "padlock": "padlock",
    "coin": "coin", "glue stick": "glue stick", "other": "none of the above (then give other_category)",
}
COLORS = ["red", "pink", "purple", "blue", "cyan", "green", "yellow", "orange", "brown", "black", "white",
          "gray", "clear", "gold", "silver"]
_CAT_TEXT = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())

PICK_TASK = """One pick by the robot's {hand} hand (rule-detected, may be wrong). Images in order:
1 head camera at segment start (the robot is about to go for the object)
2 head camera at the grasp
3-5 {hand} wrist camera 2.0 s, 1.3 s and 0.7 s before the grasp (the object is usually visible here,
    in front of or between the fingers, even when it is hidden later)
6 {hand} wrist camera while lifting
7 head camera about 1 s after the hand opens over the bin
Identify the object the {hand} hand carries. Object categories:
""" + _CAT_TEXT + """
Colors: """ + ", ".join(COLORS) + """ (the dominant color of the object; for capsules the color of the top).
A full lift to the bin almost always carries something: answer "none" only if the hand is clearly empty
in the wrist frames before AND after the grasp and nothing disappears from the table.
Return:
{{"category": str,                // one key of the category list
  "other_category": str|null,     // plain noun when category is "other"
  "color": str|null,              // one of the colors
  "hand_used": "left"|"right"|"both"|"none",
  "object_seen_in": [int],        // image numbers where the carried object is visible
  "grasp_success": bool,          // the object was lifted
  "placed_in_bin": bool,          // it is inside the bin in image 7
  "multiple_objects": bool,       // more than one object carried at once
  "human_intervention": bool,     // a person touches objects or the bin during the segment
  "table_objects_at_start": [{{"category": str, "color": str|null, "side": "left"|"right"|"center"}}],
  "anomaly": null | str,          // e.g. dropped, regrasp, wrong detection, no object
  "confidence": float}}"""


def object_name(r: dict) -> str:
    """Canonical name from a v3 answer: '<color> <category>' (water bottle: no color)."""
    cat = str(r.get("category") or "").strip().lower()
    if cat in ("", "none", "no object") or r.get("hand_used") == "none" and not r.get("grasp_success"):
        return "NONE" if cat in ("", "none", "no object") else cat
    if cat == "other":
        cat = str(r.get("other_category") or "object").strip().lower()
    if cat == "water bottle":
        return cat
    color = str(r.get("color") or "").strip().lower()
    return f"{color} {cat}" if color in COLORS else cat


def call_pick(model, text, images):
    out = call(model, text, images)
    res = out["result"]
    res["object"] = object_name(res)
    res["table_objects_at_start"] = [
        {"name": object_name(o), "side": o.get("side")} for o in res.get("table_objects_at_start", []) if isinstance(o, dict)]
    return out


STILL_TASK = """The robot looks idle in this segment (rule-detected). Images: head camera at start, middle, end.
Return:
{{"robot_idle": bool, "human_present": bool, "human_rearranging": bool,
  "table_objects": [{{"name": str, "side": "left"|"right"|"center"}}],
  "note": null | str}}"""

_lock = threading.Lock()
_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}


def b64(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def call(model: str, text: str, images: list, timeout: int = 300, retries: int = 5) -> dict:
    key, base = os.environ["LLM_API_KEY"], os.environ["LLM_BASE_URL"].rstrip("/")
    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": b64(p)}} for p in images]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": CONTEXT}, {"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
    }).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               # Cloudflare in front of the relay rejects the default Python UA (error 1010).
               "User-Agent": "openwam-subtask-labeler/0.1"}
    err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(base + "/v1/chat/completions", body, headers)
            resp = json.load(urllib.request.urlopen(req, timeout=timeout))
            msg = resp["choices"][0]["message"]["content"]
            parsed = json.loads(msg[msg.find("{"):msg.rfind("}") + 1])
            u = resp.get("usage") or {}
            with _lock:
                _usage["calls"] += 1
                _usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                _usage["completion_tokens"] += u.get("completion_tokens", 0)
            return {"model": model, "result": parsed, "usage": u}
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:  # OSError: dropped connections
            err = e
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404):
                break
            time.sleep(min(60, 2 ** attempt * 3) * (0.5 + random.random()))
    with _lock:
        _usage["errors"] += 1
    raise RuntimeError(f"{model}: {err!r}")


def cached(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text()), False
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    tmp.replace(path)
    return out, True


def episode_jobs(out, segs, kf, model):
    for ep, g in segs.groupby("episode_index"):
        picks = g[g.type == "PICK"]
        if picks.empty:
            continue
        sheet = kf[(kf.episode_index == ep) & (kf.tag == "overview")].path.iloc[0]
        hands = ", ".join(f"#{s.seg_id} {s.hand}" for s in picks.itertuples())
        text = EPISODE_TASK.format(hands=hands)
        yield out / "llm" / "episode" / model / f"ep{ep:03d}.json", (lambda t=text, p=out / sheet: call(model, t, [p]))


def inventory_text(out, ep, model) -> str:
    path = out / "llm" / "episode" / model / f"ep{ep:03d}.json"
    if not path.exists():
        return "(unknown)"
    inv = json.loads(path.read_text())["result"].get("inventory", [])
    return json.dumps([i.get("name") for i in inv], ensure_ascii=False)


def segment_jobs(out, segs, kf, models, inv_model):
    order = {"PICK": [("head", "start"), ("head", "grasp"), (None, "pre60"), (None, "pre40"), (None, "pre20"),
                      (None, "lift"), ("head", "after")],
             "STILL": [("head", "start"), ("head", "mid"), ("head", "end")]}
    kf_idx = kf.set_index(["episode_index", "seg_id", "camera", "tag"]).path
    for s in segs.itertuples():
        if s.type not in order:  # TRANSIT carries no instruction
            continue
        wrist = "lwrist" if s.hand == "left" else "rwrist"
        paths = [out / kf_idx[(s.episode_index, s.seg_id, cam or wrist, tag)] for cam, tag in order[s.type]]
        fn = call_pick if s.type == "PICK" else call
        text = PICK_TASK.format(hand=s.hand) if s.type == "PICK" else STILL_TASK
        for m in models:
            yield label_path(out, m, s), (lambda m=m, t=text, p=paths, fn=fn: fn(m, t, p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("outputs/subtask"))
    ap.add_argument("--pass", dest="which", choices=["episode", "segment"], required=True)
    ap.add_argument("--models", nargs="+", default=["gpt-6.1-sol", "gpt-6-astra"])
    ap.add_argument("--inventory-model", default="gpt-6.1-sol")
    ap.add_argument("--episodes", type=str, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    for var in ("LLM_API_KEY", "LLM_BASE_URL"):
        if not os.environ.get(var):
            sys.exit(f"missing environment variable {var}")

    segs = pd.read_parquet(args.out / "segments_rules.parquet")
    kf = pd.read_parquet(args.out / "keyframes.parquet")
    if args.episodes:
        keep = [int(e) for e in args.episodes.split(",")]
        segs, kf = segs[segs.episode_index.isin(keep)], kf[kf.episode_index.isin(keep)]
    if args.which == "episode":
        jobs = list(episode_jobs(args.out, segs, kf, args.inventory_model))
    else:
        jobs = list(segment_jobs(args.out, segs, kf, args.models, args.inventory_model))
    todo = [(p, fn) for p, fn in jobs if not p.exists()]
    print(f"{len(jobs)} calls, {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(cached, p, fn): p for p, fn in todo}
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:  # keep going; failures are listed at the end and re-run next time
                failed.append((futs[f].name, str(e)[:200]))
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {time.time() - t0:.0f}s  usage={_usage}", flush=True)
    for name, e in failed:
        print("FAILED", name, e)


if __name__ == "__main__":
    main()
