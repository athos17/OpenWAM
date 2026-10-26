"""Read-only terminal dashboard for the s7d formal run, with the 2026-10-22 run on the same steps."""
import argparse
from collections import deque
import json
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path('/mnt/pfs/v89atp/workspace/setup/action_random_64gpu_s7d_20261026')
BASELINE = Path('/mnt/pfs/v89atp/workspace/outputs/action_random_64gpu/2026-10-22_22-50-01')
WINDOW = 100


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s'


def tail(path, size=4 * 1024 * 1024):
    with path.open('rb') as handle:
        handle.seek(0, 2)
        start = max(0, handle.tell() - size)
        handle.seek(start)
        if start:
            handle.readline()
        return handle.read().decode('utf-8', errors='replace')


def load_baseline():
    """step -> (step seconds, loss_total, loss_video, loss_action, sample ids) of the 2026-10-22 run."""
    out = {}
    with (BASELINE / 'metrics.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            m = row['metrics']
            out[row['step']] = (1 / row['steps_per_sec'] if row['steps_per_sec'] > 0 else None,
                                m['loss_total'], m['loss_video'], m['loss_action'], row['sample_ids'])
    return out


def typical(seconds):
    """Drop checkpoint-save steps (over 3x the median) before averaging step time."""
    seconds = [s for s in seconds if s]
    if not seconds:
        return None
    med = statistics.median(seconds)
    kept = [s for s in seconds if s <= 3 * med]
    return sum(kept) / len(kept)


def render(baseline):
    report_path = ROOT / 'results/formal_rank0.json'
    report = json.loads(report_path.read_text())
    latest = report['history'][-1]
    metrics = latest['metrics']
    run_dir = Path(report['output_path'])
    rows = deque(maxlen=WINDOW)
    for line in tail(run_dir / 'metrics.jsonl').splitlines():
        try:
            row = json.loads(line)
            if row['step'] <= latest['step'] and row['steps_per_sec'] > 0:
                rows.append(row)
        except (ValueError, KeyError):
            continue
    avg = typical([1 / row['steps_per_sec'] for row in rows])
    matches = re.findall(r'(\d+)/(\d+)\s*\[', tail(ROOT / 'results/formal_node0.log', 128 * 1024))
    total = int(matches[-1][1]) if matches else None
    step = latest['step']
    progress = f'{step:,} / {total:,} ({step / total:.2%})' if total else f'{step:,}'
    age = time.time() - report_path.stat().st_mtime
    lines = [
        'OpenWAM | FORMAL s7d | 8 x 8 A800 | rank 0 report',
        f'Status: {report["status"]}    Last report: {age:.0f}s ago',
        f'Step: {progress}    Optimizer updates: {latest["optimizer_step"]:,}',
        f'Trainer elapsed (includes initialization): {duration(report["elapsed_seconds"])}',
    ]
    if avg:
        lines.append(f'Last {len(rows)} steps: {avg:.2f} s/step | {report["world_size"] * report["microbatch"] / avg:.1f} samples/s (saves excluded)')
        if total:
            lines.append(f'Estimated remaining: {duration((total - step) * avg)}')
    timing = run_dir / 'step_timing.jsonl'
    if timing.exists():
        recent = [line for line in tail(timing, 256 * 1024).splitlines() if line.strip()]
        if recent:
            t = json.loads(recent[-1])
            lines.append(f'Step timing @ {t["step"]}: MFU {t["mfu"]:.1%} (incl. encoders {t["mfu_incl_encoders"]:.1%}) | '
                         f'peak reserved {t["stages"]["peak_reserved_gib"]["rank_max"]:.1f} GiB')
    lines += [
        '',
        'Rank-0 loss:          total       video      action',
        f'  s7d          {metrics["loss_total"]:11.4f} {metrics["loss_video"]:11.4f} {metrics["loss_action"]:11.4f}',
    ]
    if baseline and rows:
        steps = [row['step'] for row in rows if row['step'] in baseline]
        if steps:
            mine = {row['step']: row for row in rows}
            mean = lambda key: sum(mine[s]['metrics'][key] for s in steps) / len(steps)
            base = lambda i: sum(baseline[s][i] for s in steps) / len(steps)
            rel = lambda a, b: f'{(a - b) / abs(b):+.2%}' if b else 'n/a'
            same = sum(mine[s]['sample_ids'] == baseline[s][4] for s in steps)
            base_avg = typical([baseline[s][0] for s in steps])
            lines += [
                f'Mean of steps {steps[0]}-{steps[-1]} (same batches {same}/{len(steps)}):',
                f'  s7d          {mean("loss_total"):11.4f} {mean("loss_video"):11.4f} {mean("loss_action"):11.4f}',
                f'  2026-10-22   {base(1):11.4f} {base(2):11.4f} {base(3):11.4f}',
                f'  difference   {rel(mean("loss_total"), base(1)):>11s} {rel(mean("loss_video"), base(2)):>11s} {rel(mean("loss_action"), base(3)):>11s}',
            ]
            if avg and base_avg:
                lines.append(f'Speed on the same steps: s7d {avg:.2f} s/step vs 2026-10-22 {base_avg:.2f} s/step -> {base_avg / avg:.3f}x')
    lines += [
        f'LR: {latest["lr"]:.3e}    Modality: {"video only" if latest["video_only"] else "video + action"}',
        f'Rank 0 peak allocated GPU memory: {report["peak_gpu_memory_bytes"] / 2**30:.2f} GiB',
        '',
        f'Output: {report["output_path"]}',
        'Refresh: 10s | Ctrl+C exits viewer only',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--no-baseline', action='store_true', help='skip loading the 2026-10-22 run')
    args = parser.parse_args()
    baseline = None if args.no_baseline else load_baseline()
    try:
        while True:
            try:
                message = render(baseline)
            except (OSError, ValueError, KeyError, IndexError) as error:
                message = f'Waiting for readable training reports: {error}'
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(message, flush=True)
            if args.once:
                return
            time.sleep(10)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
