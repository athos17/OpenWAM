"""HTML review page: per-episode timeline (closure / EEF z / segments) + keyframes + labels.

Uses subtasks.parquet when present (after vocab.py), else segments_rules.parquet (P0).
Open ``<out>/review.html`` in a browser; images are referenced relative to it.

Example:
  python scripts/subtask/review.py --out outputs/subtask --episodes 0,30,66,70,80
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import label_path  # noqa: E402

COLORS = {"left": "#2b6cb0", "right": "#c53030"}
# Flags that need a human decision; "human" (a person in view) and "overlap" are informational.
REVIEW_FLAGS = {"disagree_object", "disagree_success", "low_conf", "failed", "not_in_bin", "no_object",
                "hand_mismatch", "multiple_objects", "not_idle", "open_at_end"}
SEG_FILL = {"PICK": {"left": "#bee3f8", "right": "#fed7d7"}, "STILL": "#e2e8f0"}


def polyline(y: np.ndarray, x0: float, w: float, y0: float, h: float, lo: float, hi: float, step: int) -> str:
    y = y[::step]
    xs = x0 + np.arange(len(y)) * w / max(len(y) - 1, 1)
    ys = y0 + h - (np.clip(y, lo, hi) - lo) / (hi - lo) * h
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))


def timeline(ep: int, sig, segs: pd.DataFrame, width: int = 1400) -> str:
    n = len(sig[f"{ep}/left/z"])
    step = max(1, n // 2000)
    h, pad = 46, 4
    parts = [f'<svg viewBox="0 0 {width} {3 * h + 4 * pad + 18}" width="100%">']
    sx = lambda f: f / n * width  # noqa: E731
    for s in segs.itertuples():
        fill = SEG_FILL["STILL"] if s.type == "STILL" else SEG_FILL["PICK"][s.hand]
        if getattr(s, "exclude", False):
            fill = "#fefcbf"
        x, w = sx(s.start), sx(s.end) - sx(s.start)
        parts.append(f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{3 * h + 3 * pad}" fill="{fill}" '
                     f'stroke="#fff" stroke-width="1"><title>#{s.seg_id} {s.type} {s.hand or ""} '
                     f'{html.escape(str(getattr(s, "object", "") or ""))} [{s.start},{s.end})</title></rect>')
        parts.append(f'<text x="{x + 2:.1f}" y="{3 * h + 3 * pad + 14}" font-size="11">#{s.seg_id}</text>')
        ev = json.loads(s.events) if isinstance(s.events, str) and s.events else {}
        for k, f in ev.items():
            parts.append(f'<line x1="{sx(f):.1f}" x2="{sx(f):.1f}" y1="0" y2="{3 * h + 3 * pad}" '
                         f'stroke="#555" stroke-dasharray="{"2,2" if k == "peak" else "none"}" stroke-width="0.7"/>')
    for i, (key, lo, hi, label) in enumerate((("closure", -0.1, 1.1, "closure"), ("z", 0.65, 1.15, "EEF z"),
                                              ("speed", 0.0, 0.6, "EEF speed"))):
        y0 = i * (h + pad)
        parts.append(f'<text x="2" y="{y0 + 11}" font-size="10" fill="#333">{label}</text>')
        for hand in ("left", "right"):
            pts = polyline(sig[f"{ep}/{hand}/{key}"], 0, width, y0, h, lo, hi, step)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS[hand]}" stroke-width="1.1"/>')
    parts.append("</svg>")
    return "".join(parts)


def label_cell(out: Path, model: str, rule_seg) -> str:
    p = label_path(out, model, rule_seg)
    if not p.exists():
        return "<i>—</i>"
    r = json.loads(p.read_text())["result"]
    keep = ("object", "hand_used", "grasp_success", "placed_in_bin", "human_intervention", "anomaly", "confidence",
            "robot_idle", "human_rearranging", "note")
    return "<br>".join(f"<b>{k}</b>: {html.escape(str(r[k]))}" for k in keep if k in r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("outputs/subtask"))
    ap.add_argument("--episodes", type=str, default=None)
    ap.add_argument("--models", nargs="+", default=["gpt-6.1-sol", "gpt-6-astra"])
    ap.add_argument("--only-flagged", action="store_true", help="list only segments with REVIEW_FLAGS")
    ap.add_argument("--name", default="review.html")
    args = ap.parse_args(argv)

    final = args.out / "subtasks.parquet"
    segs = pd.read_parquet(final if final.exists() else args.out / "segments_rules.parquet")
    kf = pd.read_parquet(args.out / "keyframes.parquet")
    rules = pd.read_parquet(args.out / "segments_rules.parquet")
    sig = np.load(args.out / "signals.npz")
    eps = [int(e) for e in args.episodes.split(",")] if args.episodes else sorted(segs.episode_index.unique())

    body = []
    for ep in eps:
        g = segs[segs.episode_index == ep].sort_values("start")
        body.append(f'<h2 id="ep{ep}">Episode {ep} <small>({int(g.end.max())} frames, {len(g)} segments)</small></h2>')
        body.append(timeline(ep, sig, g))
        rows = []
        for s in g.itertuples():
            flags = getattr(s, "flags", "") or ""
            if args.only_flagged and not (REVIEW_FLAGS & set(flags.split(","))):
                continue
            # keyframes / LLM labels are keyed by rule-stage seg ids; vocab.py may merge
            # several rule segments into one, so map back through the rule segment starts.
            r = rules[(rules.episode_index == ep) & (rules.start >= s.start) & (rules.start < s.end)]
            k = kf[(kf.episode_index == ep) & kf.seg_id.isin(r.seg_id) & kf.tag.isin(["grasp", "after", "mid"])]
            imgs = "".join(f'<img loading="lazy" src="{html.escape(p)}" title="{html.escape(p)}">' for p in k.path)
            obj = html.escape(str(getattr(s, "object", "") or ""))
            prompt = html.escape(str(getattr(s, "prompt", "") or ""))
            excl = ' class="excl"' if getattr(s, "exclude", False) else ""
            src_ids = sorted(set(k.seg_id))
            last = r[r.seg_id == src_ids[-1]].iloc[0] if src_ids else None
            cells = "".join(f"<td>{label_cell(args.out, m, last) if last is not None else ''}</td>" for m in args.models)
            rows.append(f"<tr{excl}><td>#{s.seg_id}<br>{s.type}<br>{s.hand or ''}<br>[{s.start},{s.end})<br>"
                        f"{(s.end - s.start) / 30:.1f}s</td><td><b>{obj}</b><br>{prompt}<br>"
                        f'<span class="flag">{html.escape(flags)}</span></td><td class="imgs">{imgs}</td>{cells}</tr>')
        heads = "".join(f"<th>{m}</th>" for m in args.models)
        body.append(f"<table><tr><th>seg</th><th>label</th><th>keyframes</th>{heads}</tr>{''.join(rows)}</table>")

    nav = " ".join(f'<a href="#ep{e}">{e}</a>' for e in eps)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Subtask review</title>
<style>
body{{font:13px system-ui,sans-serif;margin:16px;color:#1a202c;background:#fff}}
table{{border-collapse:collapse;width:100%;margin-bottom:28px}} td,th{{border:1px solid #cbd5e0;padding:4px;vertical-align:top}}
td.imgs img{{height:120px;margin:1px}} .flag{{color:#c05621;font-weight:600}} tr.excl{{background:#fffff0}}
nav{{position:sticky;top:0;background:#fff;padding:6px 0;border-bottom:1px solid #e2e8f0;line-height:1.8}}
</style></head><body><h1>Subtask review</h1>
<p>Timeline: closure / EEF z / EEF speed — <span style="color:#2b6cb0">left</span>,
<span style="color:#c53030">right</span>; shaded = segments (blue L pick, red R pick, grey still, yellow excluded); lines = grasp / peak (dashed) / release.</p>
<nav>{nav}</nav>{''.join(body)}</body></html>"""
    (args.out / args.name).write_text(page)
    print(args.out / args.name)


if __name__ == "__main__":
    main()
