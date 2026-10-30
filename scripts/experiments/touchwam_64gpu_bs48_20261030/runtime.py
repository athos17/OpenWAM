"""Validate the frozen rebatch release and its separately staged full state."""
import hashlib
import json
from pathlib import Path

ROOT=Path(str(Path(__file__).absolute().parents[1]).replace('/mnt/pfs/ls3r34/','/mnt/pfs/v89atp/',1))
CONFIG_NAME='config'


def write_json(path, value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def require_preflight():
    manifest=json.loads((ROOT/'results/release_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen release changed: {name}')
    proof=json.loads((ROOT/'results/preflight.json').read_text())
    if proof['status']!='passed' or not proof['training_launch_allowed']:
        raise RuntimeError('Rebatch preflight has not passed')
    run=Path(proof['run_dir'])
    if hashlib.sha256((run/'config.yaml').read_bytes()).hexdigest()!=proof['run_config_sha256']:
        raise RuntimeError('Staged run configuration changed')
    if hashlib.sha256((run/'training_data_contract.json').read_bytes()).hexdigest()!=proof['data_contract_sha256']:
        raise RuntimeError('Staged population changed')
    t=proof['config']['training'];a=proof['config']['model']['architecture']
    assert t['batch_size']==48 and t['dataset_num_workers']==8 and t['gradient_accumulation_steps']==1
    assert t['num_epochs']==1 and t['max_steps'] is None and t['stop_after_steps'] is None
    assert t['learning_rate']==5e-5 and t['save_steps']==500 and t['save_full_states_for_resume']
    assert t['use_gradient_checkpointing'] and not t['use_gradient_checkpointing_offload']
    assert t['gradient_checkpointing_layers']==list(range(20))
    assert t['gradient_checkpointing_layers_video_only'] is None and a['mot_checkpoint_mixed_attn']
    assert t['rebatch']['world_size']==64 and t['rebatch']['target_global_batch']==3072
    assert any(run.glob('accel_state_step_*/trainer_state.json'))
    return proof
