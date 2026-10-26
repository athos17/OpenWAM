#!/usr/bin/env python3
"""Compare per-step loss curves of fixed-seed runs (training.loss_log_csv).

    python scripts/perf/compare_loss.py --ref base_a=<csv> --noise base_b=<csv> --test s7d=<csv> \
        [--window 25] [--plot out.png]

Each CSV is a run's ``debug_loss_history.csv``. ``--noise`` is a repeat of the
reference config (same seed, same code): its gap to the reference is the
run-to-run floor from non-deterministic kernels. The test run passes when its
gap to the reference stays within that floor (``--tolerance`` x the noise gap,
per metric, on window means), which is what "no training difference" means for
a change that is only meant to be faster.
"""

from __future__ import annotations

import argparse
import csv
import sys

METRICS = ("loss", "loss_video", "loss_action", "grad_norm")


def read(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[int(r["step"])] = {m: float(r[m]) for m in METRICS if m in r}
    return rows


def window_means(rows, steps, window):
    out = []
    for i in range(0, len(steps), window):
        chunk = steps[i : i + window]
        out.append({m: sum(rows[s][m] for s in chunk) / len(chunk) for m in METRICS})
    return out


def gap(a, b):
    """Mean |relative difference| of window means, per metric."""
    return {m: sum(abs(x[m] - y[m]) / max(abs(x[m]), 1e-12) for x, y in zip(a, b)) / len(a) for m in METRICS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--noise", required=True)
    p.add_argument("--test", required=True, action="append")
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--skip", type=int, default=0, help="ignore the first N steps")
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument("--plot")
    a = p.parse_args()

    def named(spec):
        name, path = spec.split("=", 1)
        return name, read(path)

    runs = [named(a.ref), named(a.noise)] + [named(t) for t in a.test]
    steps = sorted(set.intersection(*(set(r) for _, r in runs)))
    steps = [s for s in steps if s > a.skip]
    if not steps:
        sys.exit("no common steps")
    means = {name: window_means(rows, steps, a.window) for name, rows in runs}
    ref, noise = runs[0][0], runs[1][0]

    print(f"common steps {steps[0]}..{steps[-1]} ({len(steps)}), window {a.window}")
    header = "window".ljust(12) + "".join(f"{name:>12s}" for name, _ in runs)
    for metric in ("loss", "loss_video", "loss_action"):
        print(f"\n{metric} (window mean)\n{header}")
        for i in range(len(means[ref])):
            lo = steps[i * a.window]
            hi = steps[min((i + 1) * a.window, len(steps)) - 1]
            print(f"{lo:>5d}-{hi:<6d}" + "".join(f"{means[n][i][metric]:12.5f}" for n, _ in runs))

    floor = gap(means[ref], means[noise])
    print(f"\nmean |relative gap| of window means vs {ref}")
    print("run".ljust(14) + "".join(f"{m:>13s}" for m in METRICS))
    print(noise.ljust(14) + "".join(f"{floor[m]:13.4%}" for m in METRICS) + "   (noise floor)")
    ok = True
    for name, _ in runs[2:]:
        g = gap(means[ref], means[name])
        verdict = all(g[m] <= a.tolerance * floor[m] + 1e-9 for m in ("loss", "loss_video", "loss_action"))
        ok &= verdict
        print(name.ljust(14) + "".join(f"{g[m]:13.4%}" for m in METRICS) + ("   within" if verdict else "   OUTSIDE") +
              f" {a.tolerance:g}x noise")

    if a.plot:
        try:
            import matplotlib
        except ImportError:
            print("matplotlib not installed; skipping --plot")
        else:
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            k = max(1, a.window // 5)
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            for ax, metric in zip(axes, ("loss", "loss_video", "loss_action")):
                for name, rows in runs:
                    ys = [rows[s][metric] for s in steps]
                    smooth = [sum(ys[max(0, i - k + 1) : i + 1]) / (i + 1 - max(0, i - k + 1)) for i in range(len(ys))]
                    ax.plot(steps, smooth, label=name, linewidth=1.2)
                ax.set_title(f"{metric} (moving mean over {k} steps)")
                ax.set_xlabel("step")
                ax.set_yscale("log")
                ax.grid(alpha=0.3)
                ax.legend()
            fig.tight_layout()
            fig.savefig(a.plot, dpi=120)
            print(f"plot -> {a.plot}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
