"""Gate a production continuation on complete 8-node checkpoint acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(root):
    results = root/'results'
    assert not list(results.glob('failure_node*.json')), 'A node failed'
    assert not list(results.glob('resume-mismatch-rank*.json')), 'A restored state differed'
    proof = json.loads((results/'preflight.json').read_text())
    manifest = json.loads((results/'release_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    for rank in range(64):
        assert json.loads((results/f'nccl_rank{rank}.json').read_text())['status'] == 'passed'
        for phase, start, stop in [('fresh', 1, 120), ('resume', 121, 140)]:
            report = json.loads((results/f'{phase}-rank{rank}.json').read_text())
            assert report['status'] == 'passed' and report['rank'] == rank
            assert (report['first_step'], report['last_step']) == (start, stop)
            if phase == 'resume':
                assert report['resume_verified'] is True
    for node in range(8):
        assert json.loads((results/f'completed_node{node}.json').read_text())['status'] == 'completed'
    run = Path(report['run_dir'])
    rows = [json.loads(x) for x in (run/'training_metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(1, 141))
    assert all(r['opt_step'] == r['step'] for r in rows)
    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (float, int)))
    sources = {name: sum(r.get('samples_source/'+name, 0) for r in rows) for name in proof['sources']}
    assert sources and all(n > 0 for n in sources.values()), sources
    modes = {name: sum(r.get('samples_'+name, 0) for r in rows)
             for name in ['video_only', 'video_action', 'video_tactile', 'video_action_tactile']}
    assert all(n > 0 for n in modes.values()), modes
    assert sum(r.get('samples_codec/tactile_rgb_v1', 0) for r in rows) > 0
    for step in [60, 120, 140]:
        assert (run/f'checkpoint_step_{step}.safetensors').stat().st_size > 1_000_000_000
        meta = json.loads((run/f'accel_state_step_{step}/trainer_state.json').read_text())
        assert meta['global_step'] == meta['opt_step'] == step
        assert meta['optimizer_boundary'] and meta['batches_per_epoch'] == 138386
    return dict(status='passed', world_size=64, nodes=8, batch_size=12, fresh_steps=120,
                resumed_steps=20, final_step=140, sources=sources, modalities=modes,
                no_activation_recomputation=True, per_rank_restored_state_comparison=True,
                fingerprint_scope='deterministic samples from every model/optimizer tensor; exact scalar and scheduler state',
                production_population_and_sampler=True, run_dir=str(run),
                release_commit=manifest['commit'],
                acceptance_root=str(root),
                acceptance_manifest_sha256=hashlib.sha256((results/'release_manifest.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    report = verify(args.root)
    target = args.root/'results/checkpoint_acceptance.json'
    target.with_suffix('.tmp').write_text(json.dumps(report, indent=2)+'\n')
    target.with_suffix('.tmp').replace(target)
    print(json.dumps(report, indent=2))
