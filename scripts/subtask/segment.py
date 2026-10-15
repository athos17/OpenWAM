"""P0: rule-based pick events and candidate subtask segments from proprio.

Per hand, a *carry* is one large EEF lift (see ``detect_carries``) with events
grasp / peak / release.

Segments tile every episode with no gaps. One PICK per carry; the boundary
between consecutive PICKs is the slowest moment (summed EEF speed) between the
previous release and the next grasp, so the retreat after a release belongs to
the finished subtask. Sustained stillness (both arms) of at least ``--still-sec``
becomes a STILL segment. The LLM pass later assigns objects and verifies.

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


def segment_episode(state: np.ndarray, flex: dict, norm: dict, args) -> tuple:
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
    vsum = sig["left"]["speed"] + sig["right"]["speed"]
    # Still = arms AND fingers still (a hand paused at an object while closing is not idle).
    fsum = sig["left"]["finger_speed"] + sig["right"]["finger_speed"]
    still = runs((vsum < args.still_speed) & (fsum < args.still_finger_speed), int(args.still_sec * FPS),
                 merge_gap=int(0.5 * FPS))

    carries = []
    for hand in HANDS:
        for c in detect_carries(sig[hand]["z"], sig[hand]["closure"], args):
            carries.append({"hand": hand, **c})
    carries.sort(key=lambda c: c["grasp"])
    for k, c in enumerate(carries):
        c["carry_id"] = k
        c["overlap"] = bool(k and carries[k - 1]["hand"] != c["hand"] and c["grasp"] < carries[k - 1]["release"])

    def still_in(a, b):
        return [(max(s, a), min(e, b)) for s, e in still if min(e, b) - max(s, a) >= int(args.still_sec * FPS)]

    segs = []

    def add(kind, a, b, carry=None, flags=()):
        if b > a:
            segs.append({"type": kind, "start": int(a), "end": int(b), "carry": carry, "flags": list(flags)})

    # Leading gap: STILL runs before the first grasp, then PICK starts.
    first = carries[0]["grasp"] if carries else n
    cursor = 0
    pre, post = int(args.still_before_grasp_sec * FPS), int(0.5 * FPS)
    lead = still_in(0, first - pre)
    if lead and lead[0][0] <= int(0.5 * FPS):
        add("STILL", 0, lead[-1][1])
        cursor = lead[-1][1]
    for k, c in enumerate(carries):
        nxt = carries[k + 1]["grasp"] if k + 1 < len(carries) else n
        a, b = c["release"], nxt
        flags = ["overlap"] if (k + 1 < len(carries) and carries[k + 1]["overlap"]) else []
        if c["open_at_end"]:
            flags.append("open_at_end")
        if b <= a:  # next grasp (other hand) before this release
            boundary = max(c["grasp"] + 1, nxt - int(1.0 * FPS)) if k + 1 < len(carries) else n
            add("PICK", cursor, boundary, k, flags)
            cursor = boundary
            continue
        # A pause right before a grasp is the hand positioned at the object, not idling.
        gap_still = still_in(a + post, b - pre if k + 1 < len(carries) else b)
        if gap_still:
            add("PICK", cursor, gap_still[0][0], k, flags)
            add("STILL", gap_still[0][0], gap_still[-1][1])
            cursor = gap_still[-1][1]
        elif k + 1 < len(carries):
            lo, hi = a + int(0.2 * FPS), b - int(0.3 * FPS)
            boundary = lo + int(np.argmin(vsum[lo:hi])) if hi > lo else (a + b) // 2
            add("PICK", cursor, boundary, k, flags)
            cursor = boundary
        else:
            add("PICK", cursor, n, k, flags)
            cursor = n
    if cursor < n:
        if segs and segs[-1]["type"] == "STILL" or not carries:
            if segs and segs[-1]["type"] == "STILL":
                segs[-1]["end"] = n  # trailing motion after the final idle stretch stays idle-labeled
            else:
                add("STILL", cursor, n)
        else:
            add("PICK", cursor, n, len(carries) - 1)
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
    ap.add_argument("--still-speed", type=float, default=0.04, help="m/s, summed over both EEFs")
    ap.add_argument("--still-finger-speed", type=float, default=0.15, help="closure units/s, summed over hands")
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

    carry_rows, seg_rows, sig_dump = [], [], {}
    for ep in selected:
        carries, segs, sig, _ = segment_episode(states[ep], flex, norm, args)
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
