"""Speed test vs the baseline formal run on the same steps (same sample ids, same LR schedule).

    python compare_with_baseline.py [run_dir]     (default: newest run dir of this experiment)
"""
import json
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).parent))
from runtime import BASELINE_RUN, ROOT, SAMPLE_KEYS, output_root, phase, write_json  # noqa: E402

SKIP = 20  # startup, compile and first-batch steps


def read(run, until=None):
    rows = {}
    with (Path(run) / 'metrics.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if until and r['step'] > until:
                break
            rows[r['step']] = r
    return rows


def step_stats(rows, steps):
    s = sorted(1 / rows[k]['steps_per_sec'] for k in steps)
    return {'mean': st.mean(s), 'median': st.median(s), 'p90': s[int(.9 * len(s))], 'max': s[-1],
            'steps_over_15s': sum(x > 15 for x in s), 'steps': len(s)}


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in output_root().iterdir() if p.is_dir())[-1]
    ours = read(run)
    base = read(BASELINE_RUN, until=max(ours))
    common = sorted(set(ours) & set(base))
    same_ids = [k for k in common
                if [{x: s[x] for x in SAMPLE_KEYS} for s in base[k]['sample_ids']] == ours[k]['sample_ids']]
    same_lr = [k for k in common if abs(base[k]['lr'] - ours[k]['lr']) <= 1e-12 + 1e-9 * abs(base[k]['lr'])]
    first_diff = next((k for k in common if k not in same_ids), None)

    windows = []
    size = 50 if common[-1] <= 2000 else 500  # the full run: 48 windows instead of 477
    for lo in range(1, common[-1] + 1, size):
        w = [k for k in common if lo <= k < lo + size]
        entry = {'from': w[0], 'to': w[-1]}
        for m in ('loss_total', 'loss_video', 'loss_action', 'grad_norm'):
            a, b = st.mean(ours[k]['metrics'][m] for k in w), st.mean(base[k]['metrics'][m] for k in w)
            entry[m] = {'baseline': b, 's9': a, 'rel': (a - b) / b if b else None}
        windows.append(entry)

    timed = [k for k in common if k > SKIP]
    ours_s, base_s = step_stats(ours, timed), step_stats(base, timed)
    report = {'baseline': str(BASELINE_RUN), 'run': str(run), 'common_steps': len(common),
              'identical_sample_ids': len(same_ids), 'identical_lr': len(same_lr), 'first_differing_batch': first_diff,
              'windows': windows, 'baseline_step_s': base_s, 's9_step_s': ours_s,
              'speedup_mean': base_s['mean'] / ours_s['mean'], 'speedup_median': base_s['median'] / ours_s['median'],
              'projected_full_run_hours': {'baseline': 23845 * base_s['mean'] / 3600,
                                           's9': 23845 * ours_s['mean'] / 3600}}
    timing = Path(run) / 'step_timing.jsonl'
    if timing.exists():
        records = [json.loads(line) for line in timing.open()]
        late = [r for r in records if r['step'] > SKIP] or records
        stage = lambda r, k: r['stages'].get(k, {}).get('median_of_ranks', float('nan'))
        report['s9_stages_s'] = {k: st.mean(stage(r, k) for r in late) for k in (
            'data_wait_cpu_s', 'encode_text_gpu_s', 'encode_video_gpu_s', 'compute_loss_gpu_s',
            'ds_backward_gpu_s', 'ds_step_gpu_s', 'iteration_cpu_s')}
        report['s9_peak_reserved_gib'] = max(r['stages'].get('peak_reserved_gib', {}).get('rank_max', 0) for r in records)
        report['s9_mfu'] = st.mean(r['mfu'] for r in late if r.get('mfu'))
    cpu = {}
    for path in sorted((ROOT / 'results').glob(f'{phase()}_resources_node*.jsonl')):
        used = [json.loads(line).get('cpu_cores_used') for line in path.open()]
        used = [u for u in used if u]
        if used:
            cpu[path.stem.split('_')[-1]] = {'mean_cores': st.mean(used), 'p90_cores': sorted(used)[int(.9 * len(used))]}
    report['cpu_cores_used'] = cpu
    write_json(ROOT / 'results/baseline_comparison.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'windows'}, indent=2))
    for w in windows:
        print(f"steps {w['from']:>4}-{w['to']:<4} loss {w['loss_total']['rel']:+.2%}  video {w['loss_video']['rel']:+.2%}  "
              f"action {w['loss_action']['rel']:+.2%}")


if __name__ == '__main__':
    main()
