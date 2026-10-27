#!/usr/bin/env python
"""Split a training.torch_profile trace into GPU time per stage and kernel kind.

    python scripts/perf/analyze_torch_profile.py <run dir>/torch_profile/rank0.json.gz [--json out.json]

Each kernel is attributed by its CPU launch time to the innermost ``stage/<name>``
range (StepTimer stages) and, inside ``stage/ds_backward``, to activation recompute
when the launch falls in a ``mot_layer`` range. Per stage it reports the kernel time
by kind (summed over streams), the GPU-busy union, and the NCCL time not overlapped
by compute kernels (exposed communication). Values are per profiled step.
"""

import argparse
import bisect
import gzip
import json
import re
from collections import defaultdict

KINDS = [
    ("nccl", re.compile(r"nccl", re.I)),
    ("attention", re.compile(r"flash|fmha|attention|sdpa|cutlass.*mha", re.I)),
    ("conv", re.compile(r"conv|cudnn|implicit_gemm|winograd|wgrad|dgrad", re.I)),
    ("gemm", re.compile(r"gemm|xmma|cublas|cutlass|sm\d+_|ampere_|hopper_|nvjet", re.I)),
    ("copy", re.compile(r"^Memcpy|^Memset|copy_kernel|CatArrayBatched", re.I)),
]


def kind_of(name):
    for kind, pat in KINDS:
        if pat.search(name):
            return kind
    return "elementwise"


def merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def union_len(intervals):
    return sum(e - s for s, e in merge(intervals))


def subtract_len(a, b):
    """Length of the union of ``a`` not covered by the union of ``b``."""
    a, b = merge(a), merge(b)
    covered, j = 0.0, 0
    for s, e in a:
        while j < len(b) and b[j][1] <= s:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            covered += min(e, b[k][1]) - max(s, b[k][0])
            k += 1
    return sum(e - s for s, e in a) - covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json")
    args = ap.parse_args()
    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        events = json.load(f)["traceEvents"]

    stages, layers, launches, kernels = [], [], {}, []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat, name = ev.get("cat", ""), ev.get("name", "")
        if cat == "user_annotation" and name.startswith("stage/"):
            stages.append((ev["ts"], ev["ts"] + ev["dur"], name[6:]))
        elif cat == "user_annotation" and name == "mot_layer":
            layers.append((ev["ts"], ev["ts"] + ev["dur"]))
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in ev.get("args", {}):
            launches[ev["args"]["correlation"]] = ev["ts"]
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append(ev)
    if not stages:
        raise SystemExit("no stage/ ranges: was training.step_timing.enabled set?")
    n_steps = max(1, sum(1 for s in stages if s[2] == "compute_loss"))
    layers.sort()
    layer_starts = [s for s, _ in layers]

    def innermost(t):
        best = None
        for s, e, name in stages:
            if s <= t <= e and (best is None or e - s < best[1] - best[0]):
                best = (s, e, name)
        return best[2] if best else "outside_stages"

    def in_layer(t):
        i = bisect.bisect_right(layer_starts, t) - 1
        return any(layers[j][0] <= t <= layers[j][1] for j in range(max(0, i - 2), i + 1))

    per = defaultdict(lambda: defaultdict(float))
    spans = defaultdict(list)
    comm, compute = defaultdict(list), defaultdict(list)
    for k in kernels:
        t = launches.get(k.get("args", {}).get("correlation"))
        stage = innermost(t) if t is not None else "outside_stages"
        if stage == "ds_backward" and t is not None and in_layer(t):
            stage = "ds_backward.recompute"
        kind = kind_of(k["name"])
        iv = (k["ts"], k["ts"] + k["dur"])
        per[stage][kind] += k["dur"]
        spans[stage].append(iv)
        (comm if kind == "nccl" else compute)[stage].append(iv)

    rows = {}
    for stage, kinds in per.items():
        busy = union_len(spans[stage])
        exposed = subtract_len(comm[stage], compute[stage])
        rows[stage] = {
            "kernel_s": {k: v / 1e6 / n_steps for k, v in sorted(kinds.items())},
            "busy_s": busy / 1e6 / n_steps,
            "nccl_exposed_s": exposed / 1e6 / n_steps,
        }
    walls = defaultdict(float)
    for s, e, name in stages:
        walls[name] += (e - s) / 1e6 / n_steps
    all_iv = [iv for v in spans.values() for iv in v]
    step_span = (max(e for _, e in all_iv) - min(s for s, _ in all_iv)) / 1e6 / n_steps if all_iv else 0.0
    report = {"trace": args.trace, "profiled_steps": n_steps, "stage_cpu_wall_s": dict(walls),
              "gpu_span_per_step_s": step_span, "gpu_busy_per_step_s": union_len(all_iv) / 1e6 / n_steps,
              "stages": rows}

    kinds = ["gemm", "attention", "conv", "elementwise", "copy", "nccl"]
    print(f"profiled steps: {n_steps}; GPU span/step {step_span:.3f} s, busy/step {report['gpu_busy_per_step_s']:.3f} s")
    print(f"{'stage':<26}{'busy':>8}" + "".join(f"{k:>12}" for k in kinds) + f"{'nccl_exp':>10}")
    for stage, r in sorted(rows.items(), key=lambda kv: -kv[1]["busy_s"]):
        print(f"{stage:<26}{r['busy_s']:>8.3f}" + "".join(f"{r['kernel_s'].get(k, 0):>12.3f}" for k in kinds)
              + f"{r['nccl_exposed_s']:>10.3f}")
    print("stage CPU wall (s/step): " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(walls.items(), key=lambda kv: -kv[1])))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
