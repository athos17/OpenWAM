"""P0: rule-based pick events and candidate subtask segments from proprio.

Per hand, a *carry* is one large EEF lift (see ``detect_carries``) with events
grasp / peak / release.

Segments tile every episode with no gaps. One PICK per carry; a PICK ends
``--boundary-after-release-sec`` after its release (the object is in the bin), so
the retreat and the approach to the next object belong to the next subtask.
STILL = both arms and both hands below the idle thresholds for ``--still-sec``,
right after a subtask ends (or at the episode start / end). Motion after the
last object, or before an idle wait, is TRANSIT (no instruction). The LLM pass assigns objects.

Example:
  python scripts/subtask/segment.py --dataset data/pick_place_2_3_openwam_v3 --out outputs/subtask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EEF_XYZ, FPS, episode_table, flexion_columns, load_state  # noqa: E402

HANDS = ("left", "right")


def runs(mask: np.ndarray, min_len: int, merge_gap: int = 0) -> list:
    """[start, end) runs of True with length >= min_len; runs separated by < merge_gap merged first."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    spans = [[int(a), int(b)] for a, b in zip(idx[::2], idx[1::2])]
    merged = []
    for s in spans:
        if merged and s[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if b - a >= min_len]


def detect_carries(z, closure, args) -> list:
    """Carries for one hand: one per large EEF lift.

    Lift heights are bimodal (table-level wiggles < 0.09 m, carries over the bin rim
    > 0.15 m), so every z peak with prominence >= ``--z-prominence`` is one carry.
    Hand closure is only auxiliary: loose grips (balls, bottles) barely close the hand
    and some objects are released with the thumb alone, so it cannot delimit carries.

    grasp   last frame at the lowest z before the lift (hand at the object)
    peak    the z peak (over the bin)
    release steepest closure drop around/after the peak; if the hand barely opens,
            the frame where the descent from the peak starts
    """
    n = len(z)
    look = int(args.baseline_sec * FPS)
    sc = uniform_filter1d(closure, 5)
    dc = np.gradient(sc)
    peaks, props = find_peaks(z, prominence=args.z_prominence, distance=int(0.7 * FPS))
    carries = []
    for p, prom in zip(peaks, props["prominences"]):
        lo = max(0, p - look, carries[-1]["peak"] + 1 if carries else 0)
        seg = z[lo:p + 1]
        g = lo + int(np.flatnonzero(seg <= seg.min() + 0.01)[-1])
        if z[p] - z[g] < args.min_lift:
            continue
        open_lvl = float(sc[max(lo, g - int(2 * FPS)):g + 1].min())
        hold = float(np.median(sc[g:p + 1]))
        w0, w1 = max(g + 1, p - int(0.5 * FPS)), min(n, p + int(2.5 * FPS))
        r = w0 + int(np.argmin(dc[w0:w1]))
        dropped = sc[w0:r + 1].max() - sc[r:w1].min() >= 0.05
        if not dropped:
            after = np.flatnonzero(z[p:w1] < z[p] - 0.02)
            r = p + int(after[0]) if after.size else w1 - 1
        carries.append({"grasp": int(g), "peak": int(p), "release": int(max(r, p)), "open": open_lvl,
                        "hold": hold, "amp": hold - open_lvl, "lift": float(z[p] - z[g]),
                        "prominence": float(prom), "open_at_end": bool(not dropped), "n_peaks": 1})
    return carries


def move_boundary(segs: list, carries: list, near: int, to: int, window: int = int(5 * FPS)) -> bool:
    """Reviewer edit: move the segment boundary closest to ``near`` (within 5 s) to frame ``to``.
    Refused when a neighbour would become empty or a PICK would lose its own grasp."""
    cands = [i for i in range(1, len(segs)) if abs(segs[i]["start"] - near) <= window]
    if not cands:
        print(f"  boundary edit near {near}: no boundary within 5 s, skipped")
        return False
    i = min(cands, key=lambda k: abs(segs[k]["start"] - near))
    a, b = segs[i - 1], segs[i]
    ok = a["start"] < to < b["end"]
    for sg in (a, b):
        if ok and sg["type"] == "PICK":
            g = carries[sg["carry"]]["grasp"]
            lo, hi = (a["start"], to) if sg is a else (to, b["end"])
            ok = lo <= g < hi
    if not ok:
        print(f"  boundary edit {segs[i]['start']} -> {to}: would break a neighbouring segment, skipped")
        return False
    a["end"] = b["start"] = int(to)
    a["flags"].append("boundary_edited")
    b["flags"].append("boundary_edited")
    return True


def segment_episode(state: np.ndarray, flex: dict, norm: dict, args, boundary_edits=()) -> tuple:
    n = len(state)
    smooth = max(1, int(args.speed_smooth_sec * FPS))
    sig = {}
    for hand in HANDS:
        lo, hi = norm[hand]
        closure = (state[:, flex[hand]].mean(1) - lo) / (hi - lo)
        xyz = state[:, EEF_XYZ[hand]]
        speed = np.r_[0.0, np.linalg.norm(np.diff(xyz, axis=0), axis=1) * FPS]
        fingers = np.r_[0.0, np.abs(np.diff(closure)) * FPS]
        sig[hand] = {"closure": closure, "z": xyz[:, 2], "speed": uniform_filter1d(speed, smooth),
                     "finger_speed": uniform_filter1d(fingers, smooth)}
    # Still = EVERY arm and EVERY hand below the thresholds (no summing, no gap merging):
    # slow continuous motion must not count as idle.
    idle = np.ones(n, dtype=bool)
    for hand in HANDS:
        idle &= (sig[hand]["speed"] < args.still_speed) & (sig[hand]["finger_speed"] < args.still_finger_speed)
    still = runs(idle, int(args.still_sec * FPS))

    carries = []
    for hand in HANDS:
        for c in detect_carries(sig[hand]["z"], sig[hand]["closure"], args):
            carries.append({"hand": hand, **c})
    carries.sort(key=lambda c: c["grasp"])
    for k, c in enumerate(carries):
        c["carry_id"] = k
        c["overlap"] = bool(k and carries[k - 1]["hand"] != c["hand"] and c["grasp"] < carries[k - 1]["release"])

    segs = []

    def add(kind, a, b, carry=None, flags=()):
        if b > a:
            segs.append({"type": kind, "start": int(a), "end": int(b), "carry": carry, "flags": list(flags)})

    def carve(a, limit, end):
        """Tile [a, end): idle runs inside [a, limit) become STILL, the motion before each idle run
        is TRANSIT (retreat / waiting moves that serve no subtask). Returns the new cursor."""
        cur = a
        for s0, e0 in still:
            s0, e0 = max(s0, cur), min(e0, limit)
            if e0 - s0 >= int(args.still_sec * FPS):
                add("TRANSIT", cur, s0)
                add("STILL", s0, e0)
                cur = e0
        if end == n and cur < n:  # nothing follows: the rest of the episode carries no instruction
            add("TRANSIT", cur, n)
            cur = n
        return cur

    # A subtask ends once its object has been let go into the bin (release + a short settle);
    # the retreat and the approach to the next object belong to the NEXT subtask, unless the
    # robot first idles (then: TRANSIT retreat, STILL wait, and the approach starts the next one).
    settle, pre = int(args.boundary_after_release_sec * FPS), int(args.still_before_grasp_sec * FPS)
    cursor = carve(0, carries[0]["grasp"] - pre, carries[0]["grasp"]) if carries else carve(0, n, n)
    for k, c in enumerate(carries):
        nxt = carries[k + 1]["grasp"] if k + 1 < len(carries) else None
        flags = ["open_at_end"] if c["open_at_end"] else []
        boundary = min(c["release"] + settle, n)
        if nxt is not None and boundary >= nxt:
            # the next pick (other hand) grasps before this one is done: cut before its grasp
            boundary = max(c["grasp"] + 1, nxt - int(1.0 * FPS))
            flags.append("overlap")
        add("PICK", cursor, boundary, k, flags)
        cursor = carve(boundary, nxt - pre, nxt) if nxt is not None else carve(boundary, n, n)
    for near, to in boundary_edits:
        move_boundary(segs, carries, near, to)
    for sg in segs:
        if sg["type"] == "PICK" and sg["end"] - sg["start"] < int(2.0 * FPS):
            sg["flags"].append("short")
    # merge accidental duplicates (PICK split by an overlap) are kept; sanity: contiguous tiling
    for s0, s1 in zip(segs, segs[1:]):
        assert s0["end"] == s1["start"], (s0, s1)
    assert segs[0]["start"] == 0 and segs[-1]["end"] == n
    return carries, segs, sig, still


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/subtask"))
    ap.add_argument("--episodes", type=str, default=None, help="comma list; default all")
    ap.add_argument("--z-prominence", type=float, default=0.12, help="m; lift heights are bimodal around 0.09-0.15")
    ap.add_argument("--min-lift", type=float, default=0.10, help="m of z rise between grasp and peak")
    ap.add_argument("--baseline-sec", type=float, default=8.0, help="max look-back for the grasp")
    ap.add_argument("--still-speed", type=float, default=0.025, help="m/s, every EEF (idle noise p90 ~0.02)")
    ap.add_argument("--still-finger-speed", type=float, default=0.1, help="closure units/s, every hand")
    ap.add_argument("--boundary-after-release-sec", type=float, default=0.3, help="subtask ends this long after release")
    ap.add_argument("--still-sec", type=float, default=1.5)
    ap.add_argument("--still-before-grasp-sec", type=float, default=1.5, help="no STILL this close to a grasp")
    ap.add_argument("--speed-smooth-sec", type=float, default=0.2)
    args = ap.parse_args(argv)

    eps = episode_table(args.dataset)
    flex = flexion_columns(args.dataset)
    states = {int(r["episode_index"]): load_state(args.dataset, r) for _, r in eps.iterrows()}
    norm = {}
    for hand in HANDS:
        allc = np.concatenate([s[:, flex[hand]].mean(1) for s in states.values()])
        norm[hand] = (float(np.percentile(allc, 1)), float(np.percentile(allc, 99)))
    selected = [int(e) for e in args.episodes.split(",")] if args.episodes else sorted(states)
    edits = {}
    if (args.out / "boundary_edits.csv").exists():  # written by the viewer's edit mode
        be = pd.read_csv(args.out / "boundary_edits.csv")
        for r in be.itertuples():
            edits.setdefault(int(r.episode_index), []).append((int(r.near_frame), int(r.new_frame)))
        print(f"applying {len(be)} reviewer boundary edits")

    carry_rows, seg_rows, sig_dump = [], [], {}
    for ep in selected:
        carries, segs, sig, _ = segment_episode(states[ep], flex, norm, args, edits.get(ep, ()))
        for c in carries:
            carry_rows.append({"episode_index": ep, **c})
        for i, s in enumerate(segs):
            c = carries[s["carry"]] if s["type"] == "PICK" and s["carry"] is not None else None
            seg_rows.append({
                "episode_index": ep, "seg_id": i, "type": s["type"], "start": s["start"], "end": s["end"],
                "hand": c["hand"] if c else None, "carry_id": s["carry"],
                "events": json.dumps({k: int(c[k]) for k in ("grasp", "peak", "release")} if c else {}),
                "flags": ",".join(s["flags"]),
            })
        sig_dump[ep] = {h: {k: v.astype(np.float32) for k, v in sig[h].items() if k != "finger_speed"} for h in HANDS}

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(carry_rows).to_parquet(args.out / "carries_rules.parquet")
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_parquet(args.out / "segments_rules.parquet")
    np.savez_compressed(args.out / "signals.npz", **{
        f"{ep}/{h}/{k}": v for ep, d in sig_dump.items() for h, dd in d.items() for k, v in dd.items()})
    (args.out / "segment_params.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "closure_norm": norm}, indent=2))

    short = seg_df[seg_df.episode_index < 64]
    long_ = seg_df[seg_df.episode_index >= 64]
    for name, df in (("short(0-63)", short), ("long(64-86)", long_)):
        if len(df):
            print(f"{name}: eps={df.episode_index.nunique()} PICK={int((df.type == 'PICK').sum())} "
                  f"(L={int(((df.type == 'PICK') & (df.hand == 'left')).sum())}, "
                  f"R={int(((df.type == 'PICK') & (df.hand == 'right')).sum())}) "
                  f"STILL={int((df.type == 'STILL').sum())} flagged={int((df['flags'] != '').sum())}")


if __name__ == "__main__":
    main()
