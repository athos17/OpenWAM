"""P1: merge the two models' labels, canonicalize object names, build prompts.

Steps
  1. per PICK: both models' object / success / flags; agreement check
  2. one LLM call maps every raw object string to a canonical '<color> <category>' name
  3. merge a failed attempt into the next PICK of the same hand + object (a regrasp is
     part of the same subtask); other failures / empty-hand detections are flagged
  4. prompts from fixed templates; per-segment candidate counts; order-rule statistics
Outputs: subtasks.parquet, segment_labels.parquet (pre-merge, rule seg ids), subtask_prompts.json,
vocab_map.json, stats.json

Manual review corrections go in ``<out>/overrides.csv`` keyed by the RULE segment id
with columns ``episode_index,frame,object,lifted,placed,exclude,note``; ``frame`` is any frame inside the
rule segment (e.g. its grasp), so rows survive re-segmentation -- empty cells keep the model
labels; ``object`` is a canonical name or NONE; lifted/placed/exclude are true/false.

Example:
  python scripts/subtask/vocab.py --dataset data/pick_place_2_3_openwam_v3 --out outputs/subtask
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EEF_XYZ, episode_table, label_path, load_state  # noqa: E402
from label import cached, call  # noqa: E402

MAIN_PROMPT = "Pick all items into the bin"
STILL_PROMPT = "Stay still."
CANONICAL = "Pick up the {obj} with the {hand} hand and put it in the bin."
NOHAND = "Put the {obj} into the bin."
VARIANTS = (
    "Use your {hand} hand to place the {obj} in the bin.",
    "Grab the {obj} and drop it into the bin.",
    "Pick the {obj} up and put it into the bin.",
    "Move the {obj} into the bin with your {hand} hand.",
)

VOCAB_TASK = """Below are raw object names produced by two annotators for objects that a robot put into a bin,
with occurrence counts. Map every raw name to a canonical name so that the same physical kind of object
always gets the same name and different kinds get different names. Canonical names are '<color> <category>'
with plain common nouns (e.g. 'green sponge', 'yellow tape roll', 'blue cup', 'water bottle',
'green tennis ball'). Merge synonyms and over-specific variants ('green rectangular sponge' -> 'green sponge',
'clear water bottle' -> 'water bottle'); keep genuinely different objects apart. Map non-objects
('none', 'nothing', 'unknown', empty hand) to "NONE".
Return {{"map": {{raw: canonical}}, "canonical": [{{"name": str, "description": str}}]}}
Raw names:
{names}"""


def norm(s) -> str:
    return " ".join(str(s or "").lower().replace("_", " ").split())


def load_overrides(out: Path) -> list:
    """[(source, (episode, frame), row)]: overrides_auto.csv (arbitrate.py) first, then the
    reviewer's overrides.csv, so a manual row replaces an automatic one for the same segment."""
    found = []
    for name in ("overrides_auto.csv", "overrides.csv"):
        path = out / name
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
        seen = set()
        for r in df.to_dict("records"):
            key = (int(r["episode_index"]), int(r["frame"]))
            if key in seen:
                raise ValueError(f"{name}: duplicate row for {key}")
            seen.add(key)
            found.append((name, key, {k: v.strip() for k, v in r.items() if k not in ("episode_index", "frame")}))
    return found


def as_bool(v: str):
    v = v.strip().lower()
    if not v:
        return None
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"overrides.csv: not a boolean: {v!r}")


def apply_override(row: dict, o: dict, canon) -> None:
    """Replace model-derived fields of one pre-merge row with a reviewer's decision."""
    if o.get("object"):
        o = {**o, "object": "NONE" if norm(o["object"]) in ("none", "no object") else canon(o["object"])}
    flags = set(row["flags"]) - {"disagree_object", "disagree_success", "low_conf", "hand_mismatch"}
    if o.get("object"):
        row["object"] = row["object_secondary"] = o["object"]
        flags.discard("no_object")
        if o["object"] == "NONE":
            flags.add("no_object")
    lifted, placed, exclude = as_bool(o.get("lifted", "")), as_bool(o.get("placed", "")), as_bool(o.get("exclude", ""))
    if lifted is not None:
        flags.discard("failed")
        if not lifted:
            flags.add("failed")
    if placed is not None:
        flags.discard("not_in_bin")
        if lifted is not False and not placed:
            flags.add("not_in_bin")
    if exclude is not None:
        # A reviewer's keep/exclude decision supersedes every automatic exclusion reason.
        flags -= {"manual_exclude", "not_idle", "short", "multiple_objects"}
        if exclude:
            flags.add("manual_exclude")
    flags.add("reviewed")
    row["flags"] = sorted(flags)
    row["review_note"] = o.get("note", "")


def load_labels(out: Path, models, segs: pd.DataFrame) -> dict:
    labels = {}
    for m in models:
        for s in segs.itertuples():
            p = label_path(out, m, s)
            if p.exists():
                labels[(m, s.episode_index, s.seg_id)] = json.loads(p.read_text())["result"]
    return labels


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/subtask"))
    ap.add_argument("--models", nargs="+", default=["gpt-6.1-sol", "gpt-6-astra"])
    ap.add_argument("--vocab-model", default="gpt-6.1-sol")
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--trust", default=None, help="let this model decide object/success/flags (no manual review)")
    ap.add_argument("--trust-min-conf", type=float, default=0.6, help="--trust: exclude below this confidence")
    args = ap.parse_args(argv)
    primary, secondary = args.models[0], args.models[1]

    segs = pd.read_parquet(args.out / "segments_rules.parquet")
    labels = load_labels(args.out, args.models, segs)
    missing = [(m, r.episode_index, r.seg_id) for r in segs.itertuples() for m in args.models
               if (m, r.episode_index, r.seg_id) not in labels]
    if missing:
        sys.exit(f"{len(missing)} segment labels missing (e.g. {missing[:3]}); rerun label.py --pass segment")

    # ---- 2. canonical vocabulary --------------------------------------------------------
    raw = Counter()
    for (m, ep, seg), lab in labels.items():
        if "object" in lab:
            raw[norm(lab["object"])] += 1
    names = "\n".join(f"{n} ({c})" for n, c in sorted(raw.items(), key=lambda x: -x[1]))
    vocab, _ = cached(args.out / "vocab_map.json", lambda: call(args.vocab_model, VOCAB_TASK.format(names=names), []))
    vmap = {norm(k): v for k, v in vocab["result"]["map"].items()}
    unmapped = [n for n in raw if n not in vmap]
    if unmapped:
        print(f"WARNING {len(unmapped)} raw names not mapped; kept verbatim: {unmapped[:10]}")
    # Reviewer aliases (canonical_aliases.json: {name: final_name}) applied after the LLM map;
    # merges kinds the models cannot tell apart from these views (e.g. the black toy animals).
    alias_path = args.out / "canonical_aliases.json"
    aliases = {norm(k): v for k, v in json.loads(alias_path.read_text()).items()} if alias_path.exists() else {}

    def canon(s):
        c = vmap.get(norm(s), norm(s))
        return aliases.get(norm(c), aliases.get(norm(s), c))
    overrides = load_overrides(args.out)
    by_seg, source = {}, {}
    for name, (ep, frame), o in overrides:
        hit = segs[(segs.episode_index == ep) & (segs.start <= frame) & (segs.end > frame)]
        if len(hit) != 1:
            sys.exit(f"{name}: ep {ep} frame {frame} matches {len(hit)} rule segments")
        key = (ep, int(hit.seg_id.iloc[0]))
        if source.get(key) == name:
            sys.exit(f"{name}: two rows hit rule segment {key}")
        by_seg[key], source[key] = o, name

    # ---- 1. per-segment merge of the two models ------------------------------------------
    rows = []
    for r in segs.itertuples():
        a, b = labels[(primary, r.episode_index, r.seg_id)], labels[(secondary, r.episode_index, r.seg_id)]
        row = {"episode_index": r.episode_index, "seg_id": r.seg_id, "rule_seg_ids": [int(r.seg_id)], "type": r.type,
               "start": r.start, "end": r.end, "hand": r.hand, "events": r.events,
               "flags": [f for f in r.flags.split(",") if f], "object_secondary": None, "review_note": ""}
        if args.trust:
            # One model decides; the other only adds informational disagreement flags.
            t, o = (a, b) if args.trust == primary else (b, a)
        if r.type == "PICK" and args.trust:
            ot, oo = canon(t.get("object")), canon(o.get("object"))
            row["object"], row["object_secondary"] = ot, oo
            if ot == "NONE":
                row["flags"].append("no_object")
            elif ot != oo:
                row["flags"].append("disagree_object")
            if not t.get("grasp_success"):
                row["flags"].append("failed")
            elif not t.get("placed_in_bin"):
                row["flags"].append("not_in_bin")
            if bool(t.get("grasp_success")) != bool(o.get("grasp_success")) or \
                    bool(t.get("placed_in_bin")) != bool(o.get("placed_in_bin")):
                row["flags"].append("disagree_success")
            if t.get("multiple_objects"):
                row["flags"].append("multiple_objects")
            if t.get("human_intervention"):
                row["flags"].append("human")
            if float(t.get("confidence", 0)) < args.trust_min_conf:
                row["flags"].append("trust_low_conf")
            if t.get("hand_used") not in (r.hand, "both"):
                row["flags"].append("trust_hand_mismatch")
            side_objs = [canon(x.get("name")) for x in t.get("table_objects_at_start", [])
                         if x.get("side") in (r.hand, "center")]
            row["n_candidates_same_side"] = len(set(side_objs) | {ot})
        elif r.type == "PICK":
            oa, ob = canon(a["object"]), canon(b["object"])
            row["object"] = oa
            row["object_secondary"] = ob
            lifted = [bool(a.get("grasp_success")), bool(b.get("grasp_success"))]
            placed = [bool(a.get("placed_in_bin")), bool(b.get("placed_in_bin"))]
            if oa == "NONE" and ob == "NONE":
                row["flags"].append("no_object")
            elif oa != ob:
                row["flags"].append("disagree_object")
            if not any(lifted):
                row["flags"].append("failed")
            elif not all(lifted):
                row["flags"].append("disagree_success")
            if any(lifted) and not any(placed):
                row["flags"].append("not_in_bin")
            if a.get("multiple_objects") and b.get("multiple_objects"):
                row["flags"].append("multiple_objects")
            if a.get("human_intervention") or b.get("human_intervention"):
                row["flags"].append("human")
            if min(float(a.get("confidence", 0)), float(b.get("confidence", 0))) < args.min_conf:
                row["flags"].append("low_conf")
            if a.get("hand_used") not in (r.hand, "both"):
                row["flags"].append("hand_mismatch")
            side_objs = [canon(o.get("name")) for o in a.get("table_objects_at_start", [])
                         if o.get("side") in (r.hand, "center")]
            row["n_candidates_same_side"] = len(set(side_objs) | {oa})
        else:
            row["object"] = None
            judges = (t,) if args.trust else (a, b)
            if any(j.get("human_rearranging") for j in judges):
                row["flags"].append("human")
            if not all(j.get("robot_idle") for j in judges):
                row["flags"].append("not_idle")
        if (r.episode_index, r.seg_id) in by_seg:
            apply_override(row, by_seg[(r.episode_index, r.seg_id)], canon)
        rows.append(row)
    df = pd.DataFrame(rows)
    pre = df.copy()
    pre["flags"] = [",".join(sorted(set(f))) for f in pre["flags"]]
    pre.to_parquet(args.out / "segment_labels.parquet")
    print(f"applied {len(overrides)} review overrides")

    # ---- 3. fold failed attempts into the following regrasp ------------------------------
    merged = []
    for ep, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("start").to_dict("records")
        out = []
        pending = None  # a failed attempt waiting for its regrasp
        for row in g:
            if pending is not None:
                if row["type"] == "PICK" and row["hand"] == pending["hand"] and row["object"] == pending["object"]:
                    row["start"] = pending["start"]
                    row["rule_seg_ids"] = pending["rule_seg_ids"] + row["rule_seg_ids"]
                    row["flags"] = sorted(set(row["flags"]) | {"regrasp"})
                    pending = None
                else:
                    out.append(pending)
                    pending = None
            if row["type"] == "PICK" and "failed" in row["flags"] and "no_object" not in row["flags"]:
                pending = row
                continue
            out.append(row)
        if pending is not None:
            out.append(pending)
        for i, row in enumerate(out):
            row["seg_id"] = i
        merged.extend(out)
    df = pd.DataFrame(merged)
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("start")
        assert (g.start.to_numpy()[1:] == g.end.to_numpy()[:-1]).all(), f"episode {ep} not tiled"

    # ---- 4. prompts ------------------------------------------------------------------------
    exclude = {"no_object", "failed", "not_in_bin", "multiple_objects", "manual_exclude", "short", "not_idle",
               "trust_low_conf", "trust_hand_mismatch"}
    prompts, variants = [], []
    for row in df.itertuples():
        if row.type == "STILL":
            prompts.append((STILL_PROMPT, STILL_PROMPT))
            variants.append([])
        elif row.object and row.object != "NONE":
            prompts.append((CANONICAL.format(obj=row.object, hand=row.hand), NOHAND.format(obj=row.object)))
            variants.append([v.format(obj=row.object, hand=row.hand) for v in VARIANTS])
        else:
            prompts.append((None, None))
            variants.append([])
    df["prompt"] = [p[0] for p in prompts]
    df["prompt_nohand"] = [p[1] for p in prompts]
    df["variants"] = variants
    df["exclude"] = [bool(exclude & set(f)) or p[0] is None for f, p in zip(df["flags"], prompts)]
    df["flags"] = [",".join(sorted(set(f))) for f in df["flags"]]

    # same canonical name for two different objects inside one episode -> cannot be commanded apart
    dup = []
    for ep, g in df[(df.type == "PICK") & ~df.exclude].groupby("episode_index"):
        c = Counter(g.object)
        dup += [(ep, o, n) for o, n in c.items() if n > 1]

    # ---- order-rule statistic (short static episodes): is the next object the nearest one? ----
    eps = episode_table(args.dataset).set_index("episode_index", drop=False)
    hits = total = 0
    for ep, g in df[(df.type == "PICK") & ~df.exclude & (df.episode_index < 64)].groupby("episode_index"):
        state = load_state(args.dataset, eps.loc[ep])
        g = g.sort_values("start")
        for hand, gh in g.groupby("hand"):
            grasp_xy = [state[json.loads(e)["grasp"], EEF_XYZ[hand]][:2] for e in gh.events]
            for k in range(len(gh) - 1):
                here = state[int(gh.start.iloc[k]), EEF_XYZ[hand]][:2]
                d = [np.linalg.norm(grasp_xy[j] - here) for j in range(k, len(gh))]
                total += 1
                hits += int(np.argmin(d) == 0)

    df.to_parquet(args.out / "subtasks.parquet")
    used = df[~df.exclude]
    objects = sorted(used[used.type == "PICK"].object.unique())
    registry = {
        "main_prompt": MAIN_PROMPT,
        "still_prompt": STILL_PROMPT,
        "templates": {"canonical": CANONICAL, "nohand": NOHAND, "variants": list(VARIANTS)},
        "objects": objects,
        "prompts": sorted({MAIN_PROMPT, STILL_PROMPT} | set(used.prompt.dropna()) | set(used.prompt_nohand.dropna())
                          | {v for vs in used.variants for v in vs}),
    }
    (args.out / "subtask_prompts.json").write_text(json.dumps(registry, ensure_ascii=False, indent=1))

    pick = df[df.type == "PICK"]
    flag_counts = Counter(f for fs in df["flags"] for f in fs.split(",") if f)
    stats = {
        "segments": int(len(df)), "pick": int(len(pick)), "still": int((df.type == "STILL").sum()),
        "pick_used": int((~pick.exclude).sum()), "excluded": int(df.exclude.sum()),
        "excluded_frames_frac": float(((df.end - df.start) * df.exclude).sum() / (df.end - df.start).sum()),
        "flags": dict(flag_counts),
        "objects": len(objects), "object_counts": dict(Counter(used[used.type == "PICK"].object).most_common()),
        "prompts": len(registry["prompts"]),
        "decision_points_ge2_candidates": int((pick[~pick.exclude].n_candidates_same_side >= 2).sum()),
        "nearest_first_rule_short_eps": {"hits": hits, "total": total, "acc": hits / max(total, 1)},
        "duplicate_names_in_episode": dup[:50],
    }
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("object_counts", "duplicate_names_in_episode")}, indent=1))
    print("top objects:", list(stats["object_counts"].items())[:25])
    print(f"duplicate names within an episode: {len(dup)}")


if __name__ == "__main__":
    main()
