"""Compare TouchWAM a800 runs (P0 baseline vs P1 variants): speed per modality, memory, losses.

    python scripts/perf/touchwam_a800/compare.py p0 p0r p1 p1v [--root /gaozt-test1/lyh/perf_runs/touchwam_a800]
        [--from-step 21] [--to-step N] [--window 100] [--json out.json]

Each TAG resolves to the newest <root>/<TAG>/<timestamp>/ holding step_trace_rank*.jsonl.
- Speed: per step, the median over ranks of each stage (GPU time where recorded) and the max over
  ranks of the step (the step really waits for the slowest rank); grouped by the step's modality.
- Memory: per-step peaks (training.step_timing.peak_per_step) per modality, max over ranks.
- Losses: debug_loss_history.csv of each run against the first run. The runs share seed and samples,
  so differences come only from nondeterministic kernels. Step by step they compound in bf16, so long
  runs are judged on per-modality means over --window steps; a repeat of the baseline (p0r) gives the
  noise floor.
- All runs are cut to the steps every run has (or --to-step): step k is the same batch in each run, so
  runs of different lengths are only comparable on their common steps.
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics as st

MODES = ("V", "VA", "VAT", "VT")
STAGES = ("data_wait_cpu_s", "encode_text_gpu_s", "encode_video_gpu_s", "prepare_inputs_gpu_s",
          "forward_gpu_s", "ds_backward_gpu_s", "ds_step_gpu_s", "backward_cpu_s", "metrics_reduce_gpu_s",
          "step_cpu_s")
LOSSES = ("loss", "loss_video", "loss_action", "loss_tactile", "grad_norm")


def run_dir(root, tag):
    dirs = sorted(d for d in glob.glob(os.path.join(root, tag, "*")) if glob.glob(os.path.join(d, "step_trace_rank*.jsonl")))
    if not dirs:
        raise SystemExit(f"no step traces under {root}/{tag}")
    return dirs[-1]


def mode(record):
    s = record.get("samples", {})
    if s.get("samples_video_action_tactile"):
        return "VAT"
    if s.get("samples_video_tactile"):
        return "VT"
    return "V" if s.get("samples_video_only") else "VA"


def load(path, lo, hi=None):
    traces = {}
    for p in glob.glob(os.path.join(path, "step_trace_rank*.jsonl")):
        rank = int(p.rsplit("rank", 1)[1].split(".")[0])
        traces[rank] = {r["step"]: r for r in map(json.loads, open(p))}
    steps = sorted(k for k in set.intersection(*(set(t) for t in traces.values())) if k >= lo and (hi is None or k <= hi))
    rows = []
    for k in steps:
        recs = [traces[r][k] for r in sorted(traces)]
        row = {"step": k, "mode": mode(recs[0]), "ranks": len(recs)}
        for key in STAGES:
            if key == "forward_gpu_s":
                vals = [r.get("compute_loss_gpu_s", 0.0) - r.get("prepare_inputs_gpu_s", 0.0) for r in recs]
            else:
                vals = [r.get(key, 0.0) for r in recs]
            row[key] = st.median(vals)
            row[key + "_min"] = min(vals)
        # Wall time of the step for the job: the slowest rank's loop iteration.
        row["iter_max_s"] = max(r.get("step_cpu_s", 0.0) + r.get("data_wait_cpu_s", 0.0) for r in recs)
        row["peak_alloc_gib"] = max(r.get("peak_allocated_gib", 0.0) for r in recs)
        row["peak_reserved_gib"] = max(r.get("peak_reserved_gib", 0.0) for r in recs)
        rows.append(row)
    return rows


def summarize(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["mode"]].append(r)
    n = len(rows)
    out = {"steps": n, "mean_iter_s": st.mean(r["iter_max_s"] for r in rows),
           "median_iter_s": st.median(r["iter_max_s"] for r in rows), "modes": {}}
    for m in MODES:
        rs = by.get(m, [])
        if not rs:
            continue
        d = {"n": len(rs), "share": len(rs) / n}
        for key in STAGES + ("iter_max_s",):
            d[key] = st.median(r[key] for r in rs)
        d["forward_min_gpu_s"] = st.median(r["forward_gpu_s_min"] for r in rs)
        d["mean_iter_s"] = st.mean(r["iter_max_s"] for r in rs)
        d["peak_alloc_gib"] = max(r["peak_alloc_gib"] for r in rs)
        d["peak_reserved_gib"] = max(r["peak_reserved_gib"] for r in rs)
        out["modes"][m] = d
    # Production modality shares (r2 steps 101-250) reweight the subset's sampled mix.
    prod = {"V": 0.30, "VA": 0.37, "VAT": 0.27, "VT": 0.06}
    if set(prod) <= set(out["modes"]):
        out["prod_weighted_iter_s"] = sum(prod[m] * out["modes"][m]["mean_iter_s"] for m in prod)
    return out


def losses(path):
    p = os.path.join(path, "debug_loss_history.csv")
    if not os.path.exists(p):
        return {}
    return {int(r["step"]): {k: float(r[k]) for k in LOSSES if r.get(k) not in (None, "")} for r in csv.DictReader(open(p))}


def loss_diff(ref, other):
    common = sorted(set(ref) & set(other))
    worst = {}
    for k in LOSSES:
        diffs = [(abs(other[s][k] - ref[s][k]) / max(abs(ref[s][k]), 1e-8), s) for s in common if k in ref[s] and k in other[s]]
        if diffs:
            worst[k] = {"max_rel": max(diffs)[0], "at_step": max(diffs)[1],
                        "median_rel": st.median(d for d, _ in diffs),
                        "first_step_rel_gt_1e-3": next((s for d, s in sorted(diffs, key=lambda x: x[1]) if d > 1e-3), None)}
    return {"steps": len(common), "worst": worst}


def loss_windows(ref, other, modes, window):
    """Per-modality means over consecutive windows of steps; worst relative gap per loss."""
    buckets = collections.defaultdict(lambda: ([], []))
    for s in sorted(set(ref) & set(other) & set(modes)):
        for k in LOSSES:
            if k in ref[s] and k in other[s]:
                a, b = buckets[(modes[s], (s - 1) // window, k)]
                a.append(ref[s][k])
                b.append(other[s][k])
    worst = {}
    for (m, w, k), (a, b) in buckets.items():
        ra, rb = st.mean(a), st.mean(b)
        rel = abs(rb - ra) / max(abs(ra), 1e-8)
        if rel > worst.get((m, k), {"rel": -1})["rel"]:
            worst[(m, k)] = {"rel": rel, "steps": f"{w * window + 1}-{(w + 1) * window}", "n": len(a), "ref": ra, "other": rb}
    return {f"{m}/{k}": v for (m, k), v in sorted(worst.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--root", default="/gaozt-test1/lyh/perf_runs/touchwam_a800")
    ap.add_argument("--from-step", type=int, default=21)
    ap.add_argument("--to-step", type=int, help="last step compared (default: the last step every run has)")
    ap.add_argument("--window", type=int, default=100, help="steps per loss-mean window")
    ap.add_argument("--json")
    args = ap.parse_args()
    paths = {tag: run_dir(args.root, tag) for tag in args.tags}
    rows = {tag: load(path, args.from_step, args.to_step) for tag, path in paths.items()}
    hi = min(r[-1]["step"] for r in rows.values() if r)
    if any(r and r[-1]["step"] > hi for r in rows.values()):
        print(f"comparing the common steps {args.from_step}-{hi}")
    report = {}
    ref_losses = ref_modes = None
    for tag in args.tags:
        s = summarize([r for r in rows[tag] if r["step"] <= hi])
        s["dir"] = paths[tag]
        cur = {k: v for k, v in losses(paths[tag]).items() if k <= hi}
        if ref_losses is None:
            ref_losses, ref_modes = cur, {r["step"]: r["mode"] for r in rows[tag]}
        elif cur:
            s["loss_vs_" + args.tags[0]] = loss_diff(ref_losses, cur)
            s["loss_windows_vs_" + args.tags[0]] = loss_windows(ref_losses, cur, ref_modes, args.window)
        report[tag] = s
    cols = ("n", "data_wait_cpu_s", "encode_text_gpu_s", "encode_video_gpu_s", "forward_min_gpu_s", "forward_gpu_s",
            "ds_backward_gpu_s", "ds_step_gpu_s", "iter_max_s", "mean_iter_s", "peak_alloc_gib")
    short = ("n", "wait", "text", "vae", "fwd_mn", "fwd_md", "bwd", "dsstep", "it_med", "it_mean", "peakA")
    print("%-8s %-4s " % ("run", "mode") + " ".join("%7s" % c for c in short))
    for tag, s in report.items():
        for m, d in s["modes"].items():
            print("%-8s %-4s " % (tag, m) + " ".join("%7d" % d[c] if c == "n" else "%7.3f" % d[c] for c in cols))
        print("%-8s all  steps=%d mean=%.3f median=%.3f prod-weighted=%s  %s" % (
            tag, s["steps"], s["mean_iter_s"], s["median_iter_s"],
            "%.3f" % s["prod_weighted_iter_s"] if "prod_weighted_iter_s" in s else "-", s["dir"]))
    base = report[args.tags[0]]
    for tag in args.tags[1:]:
        s = report[tag]
        if "prod_weighted_iter_s" in s and "prod_weighted_iter_s" in base:
            print("%s vs %s: prod-weighted %.3f -> %.3f s (%+.1f%%)" % (
                tag, args.tags[0], base["prod_weighted_iter_s"], s["prod_weighted_iter_s"],
                100 * (s["prod_weighted_iter_s"] / base["prod_weighted_iter_s"] - 1)))
        lv = s.get("loss_vs_" + args.tags[0])
        if lv:
            print("  losses over %d steps: " % lv["steps"] + ", ".join(
                "%s max_rel %.2e@%d" % (k, v["max_rel"], v["at_step"]) for k, v in lv["worst"].items()))
        lw = s.get("loss_windows_vs_" + args.tags[0])
        if lw:
            print("  %d-step per-modality means, worst gap: " % args.window + ", ".join(
                "%s %.2e@%s" % (k, v["rel"], v["steps"]) for k, v in lw.items() if k.endswith("/loss")))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
