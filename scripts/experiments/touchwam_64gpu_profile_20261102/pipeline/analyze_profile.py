"""Split a profile run's GPU timelines by stage and modality, and find who each long collective waited for.

    python analyze_profile.py [run_dir] [--first-step 23] [--group-ranks 8]    (stdlib only; one trace in memory at a time)

Per (stage, modality), over the traced ranks: wall, compute (union of non-NCCL kernels), communication not hidden
by compute, GPU idle, and NCCL kernel time by kind. For the first node's ranks (same clock), each collective's
start spread gives the late rank and how idle its GPU was just before. The slowest steps are listed with every
rank's Python GC pause from the step traces. Result: <run_dir>/torch_profile/summary.json.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import glob
import gzip
import json
import os
import re
import statistics as st

STAGES = {'stage/ds_backward': 'backward', 'stage/ds_step': 'optimizer', 'stage/metrics_reduce': 'metrics'}
MIN_WAIT_US = 30e3


def mode(record):
    s = record.get('samples', {})
    if s.get('samples_video_action_tactile'):
        return 'VAT'
    if s.get('samples_video_tactile'):
        return 'VT'
    return 'VA' if s.get('samples_video_action') else 'V'


def union(intervals):
    total, cur = 0.0, None
    for a, b in sorted(intervals):
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            total += cur[1] - cur[0] if cur else 0.0
            cur = [a, b]
    return total + (cur[1] - cur[0] if cur else 0.0)


def kind(name):
    m = re.search(r'ncclDevKernel_([A-Za-z]+)', name)
    return m.group(1) if m else 'other'


def stage_windows(events):
    """{stage: [(start, end), ...]} in call order; the forward runs from the encoders' end to compute_loss's end.

    A kernel belongs to its innermost annotation on the GPU timeline, so a parent's range covers only its own
    kernels (prepare_inputs can be a few ms): the encoders' extent comes from encode_text and encode_video.
    """
    raw = defaultdict(list)
    for e in events:
        if e.get('cat') == 'gpu_user_annotation' and e.get('name', '').startswith('stage/'):
            raw[e['name']].append((e['ts'], e['ts'] + e['dur']))
    out = {STAGES[k]: sorted(v) for k, v in raw.items() if k in STAGES}
    # fsdp2 has plain backward / optimizer_wrapper; under DeepSpeed those wrap ds_backward / ds_step.
    for name, stage in (('stage/backward', 'backward'), ('stage/optimizer_wrapper', 'optimizer')):
        if stage not in out and name in raw:
            out[stage] = sorted(raw[name])
    text, video, loss = sorted(raw['stage/encode_text']), sorted(raw['stage/encode_video']), sorted(raw['stage/compute_loss'])
    out['encode'] = [(min(t[0], v[0]), max(t[1], v[1])) for t, v in zip(text, video)]
    out['forward'] = [(enc[1], c[1]) for enc, c in zip(out['encode'], loss)]
    return out


def load_rank(path):
    with gzip.open(path) as f:
        events = json.load(f)['traceEvents']
    kernels = sorted((e['ts'], e['ts'] + e['dur'], e['name']) for e in events
                     if e.get('cat') == 'kernel' and e.get('ph') == 'X')
    return kernels, stage_windows(events)


def stage_rows(kernels, windows):
    starts = [k[0] for k in kernels]
    longest = max((k[1] - k[0] for k in kernels), default=0)
    rows, collectives = {}, {}
    for stage, spans in windows.items():
        for call, (s, e) in enumerate(spans):
            i = bisect.bisect_left(starts, s - longest)
            inside = [(max(a, s), min(b, e), n) for a, b, n in kernels[i:bisect.bisect_right(starts, e)] if b > s]
            compute = union((a, b) for a, b, n in inside if 'nccl' not in n.lower())
            busy = union((a, b) for a, b, n in inside)
            nccl = Counter()
            for a, b, n in inside:
                if 'nccl' in n.lower():
                    nccl[kind(n)] += b - a
            rows[stage, call] = {'wall': e - s, 'compute': compute, 'comm_exposed': busy - compute,
                                 'idle': e - s - busy, 'nccl': dict(nccl)}
            collectives[stage, call] = [(a, b, kind(n)) for a, b, n in kernels[i:bisect.bisect_right(starts, e)]
                                        if 'nccl' in n.lower() and a >= s and b <= e + 1e5]
    return rows, collectives, kernels


def gpu_busy_fraction(kernels, starts, a0, a1):
    longest = 5e6
    i = bisect.bisect_left(starts, a0 - longest)
    return union((max(a, a0), min(b, a1)) for a, b, n in kernels[i:bisect.bisect_right(starts, a1)]
                 if b > a0 and 'nccl' not in n.lower()) / max(a1 - a0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', nargs='?')
    p.add_argument('--first-step', type=int, default=23)
    p.add_argument('--group-ranks', type=int, default=8)
    args = p.parse_args()
    run = args.run_dir
    if run is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from runtime import output_root
        run = str(sorted(q for q in output_root().iterdir() if q.is_dir())[-1])
    traces = {int(re.search(r'rank(\d+)\.json', q).group(1)): q for q in glob.glob(f'{run}/torch_profile/rank*.json.gz')}
    step_traces = {int(re.search(r'rank(\d+)\.jsonl', q).group(1)): q for q in glob.glob(f'{run}/step_trace_rank*.jsonl')}
    records = {r: {x['step']: x for x in map(json.loads, open(q))} for r, q in step_traces.items()}
    print(f'{run}: traces {sorted(traces)}, step traces {len(records)}')

    table = defaultdict(lambda: defaultdict(list))
    per_rank, group = {}, {}
    for rank in sorted(traces):
        kernels, windows = load_rank(traces[rank])
        rows, collectives, kernels = stage_rows(kernels, windows)
        per_rank[rank] = rows
        for (stage, call), row in rows.items():
            md = mode(records[rank][args.first_step + call])
            for m in (md, 'all'):
                for f in ('wall', 'compute', 'comm_exposed', 'idle'):
                    table[stage, m][f].append(row[f])
                for k, v in row['nccl'].items():
                    table[stage, m]['nccl_' + k].append(v)
                table[stage, m]['n'].append(1)
        if rank < args.group_ranks:
            group[rank] = (collectives, kernels, [k[0] for k in kernels])

    summary = {'stages': {}}
    print('\nstage / modality: mean ms per call over the traced ranks')
    for (stage, m), cols in sorted(table.items()):
        n = len(cols['n'])
        row = {f: sum(v) / n / 1e3 for f, v in cols.items() if f != 'n'}
        summary['stages'][f'{stage}/{m}'] = row
        extra = ' '.join(f'{f[5:]} {v:.0f}' for f, v in sorted(row.items()) if f.startswith('nccl_'))
        print(f"  {stage:9s} {m:4s} n={n:4d} wall {row['wall']:7.1f} compute {row['compute']:7.1f} "
              f"exposed {row['comm_exposed']:7.1f} idle {row['idle']:6.1f} | {extra}")

    late, waits = Counter(), []
    for stage_call in sorted({k for c, _, _ in group.values() for k in c}, key=str):
        lists = {r: g[0].get(stage_call, []) for r, g in group.items()}
        if len(lists) < 2 or len({len(v) for v in lists.values()}) != 1:
            continue
        for i in range(len(next(iter(lists.values())))):
            start = {r: v[i][0] for r, v in lists.items()}
            spread = max(start.values()) - min(start.values())
            if spread < MIN_WAIT_US:
                continue
            r = max(start, key=start.get)
            busy = gpu_busy_fraction(group[r][1], group[r][2], start[r] - spread, start[r])
            late[r] += spread
            waits.append({'stage': stage_call[0], 'step': args.first_step + stage_call[1], 'collective': i,
                          'kind': lists[r][i][2], 'late_rank': r, 'wait_ms': spread / 1e3, 'late_gpu_busy': busy})
    by_stage = defaultdict(float)
    for w in waits:
        by_stage[w['stage']] += w['wait_ms']
    steps = len({w['step'] for w in waits}) or 1
    summary['node0_waits'] = waits
    print(f"\nnode 0 waits over {MIN_WAIT_US / 1e3:.0f} ms: {len(waits)}; ms by late rank "
          f"{ {k: round(v / 1e3) for k, v in sorted(late.items())} }; ms by stage { {k: round(v) for k, v in by_stage.items()} }; "
          f"late GPU idle (<20% busy) in {sum(w['late_gpu_busy'] < .2 for w in waits)} of them")
    for w in sorted(waits, key=lambda w: -w['wait_ms'])[:12]:
        print(f"  step {w['step']} {w['stage']:9s} #{w['collective']:<3d} {w['kind']:12s} late rank {w['late_rank']} "
              f"{w['wait_ms']:6.0f} ms, its GPU {w['late_gpu_busy'] * 100:3.0f}% busy")

    ranks = sorted(records)
    active = [s for s in sorted(records[ranks[0]]) if args.first_step <= s < args.first_step + 20]
    step_rows = []
    for s in active:
        rows = [records[r][s] for r in ranks if s in records[r]]
        gc = [x.get('gc_pause_cpu_s', 0.0) for x in rows]
        step_rows.append({'step': s, 'mode': mode(rows[0]), 'step_s': st.median(x['step_cpu_s'] for x in rows),
                          'gc_max_s': max(gc), 'gc_ranks_over_100ms': sum(g > .1 for g in gc),
                          'gen2': sum(x.get('gc_gen2_count', 0) for x in rows)})
    summary['steps'] = step_rows
    print('\nprofiled steps, slowest first (step time median over ranks; GC from all ranks):')
    for x in sorted(step_rows, key=lambda x: -x['step_s'])[:10]:
        print(f"  step {x['step']} {x['mode']:3s} {x['step_s']:6.2f} s  GC max {x['gc_max_s'] * 1e3:6.0f} ms, "
              f"{x['gc_ranks_over_100ms']} ranks over 100 ms, {x['gen2']} gen2")
    with open(f'{run}/torch_profile/summary.json', 'w') as f:
        json.dump(summary, f)


if __name__ == '__main__':
    main()
