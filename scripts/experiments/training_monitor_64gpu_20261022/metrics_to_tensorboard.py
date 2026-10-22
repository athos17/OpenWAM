"""Read OpenWAM JSONL and rank reports; write independent TensorBoard events.

No torch imports, trainer edits, network APIs, or writes to the training run.
Event wall_time is ingestion time; the meaningful history axis is microstep.
"""
import argparse,fcntl,json,math,os,time
from pathlib import Path
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

DEFAULT=Path('/mnt/pfs/v89atp/workspace')

def atomic_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,path)

def read_complete_rows(path,offset):
    rows=[]
    with path.open('rb') as f:
        f.seek(offset)
        while True:
            start=f.tell();line=f.readline()
            if not line:break
            if not line.endswith(b'\n'):
                f.seek(start);break
            rows.append(json.loads(line))
        return rows,f.tell()

def scalar(writer,tag,value,step):
    if not math.isfinite(value):raise ValueError(f'Nonfinite scalar {tag}: {value}')
    writer.add_event(Event(wall_time=time.time(),step=step,summary=Summary(value=[Summary.Value(tag=tag,simple_value=float(value))])))

def consume(writer,rows,state):
    for row in rows:
        step=row['step'];m=row['metrics']
        if step!=state['last_step']+1:raise ValueError(f'Expected step {state["last_step"]+1}, got {step}; do not mix runs.')
        source='egodex' if row['video_only'] else 'robot'
        scalar(writer,f'loss/{source}_video',m['loss_video'],step)
        if not row['video_only']:
            scalar(writer,'loss/robot_action',m['loss_action'],step)
            scalar(writer,'loss/robot_total',m['loss_total'],step)
        elif m['loss_action']!=0:raise ValueError('Video-only row contains action loss')
        scalar(writer,'train/learning_rate',row['lr'],step)
        scalar(writer,'progress/optimizer_updates',row['optimizer_step'],step)
        scalar(writer,'progress/epoch_percent',100*step/state['microsteps_per_epoch'],step)
        if row['steps_per_sec']>0:
            scalar(writer,'speed/seconds_per_microstep',1/row['steps_per_sec'],step)
            scalar(writer,'speed/global_samples_per_second',row['steps_per_sec']*state['world_size']*state['microbatch'],step)
        state['pending_losses'].append(m['loss_total'])
        if row['optimizer_step']>state['optimizer_step']:
            scalar(writer,'loss/optimizer_window_mean_total',sum(state['pending_losses'])/len(state['pending_losses']),step)
            scalar(writer,'train/grad_norm_before_clip',m['grad_norm'],step)
            state['pending_losses']=[]
        state['last_step']=step;state['optimizer_step']=row['optimizer_step']
        state['source_batches'][source]+=1
        state['last_metrics']=m

def rank_snapshot(reports, world_size):
    loaded=[];missing=[]
    for rank in range(world_size):
        p=reports/f'formal_rank{rank}.json'
        try:
            obj=json.loads(p.read_text());loaded.append((obj,max(0,time.time()-p.stat().st_mtime)))
        except (OSError,json.JSONDecodeError):missing.append(rank)
    if not loaded:return {'readable_ranks':0,'missing_ranks':missing}
    steps=[o['history'][-1]['step'] if o['history'] else 0 for o,_ in loaded]
    return {'readable_ranks':len(loaded),'missing_ranks':missing,'min_rank_microstep':min(steps),'max_rank_microstep':max(steps),
            'max_report_age_seconds':max(age for _,age in loaded),'peak_allocated_GiB_max_rank':max(o['peak_gpu_memory_bytes'] for o,_ in loaded)/2**30,
            'note':'Rank reports are read sequentially; a one-step difference can be a sampling race. This is not scheduler status or current GPU utilization.'}

def resource_snapshot(reports):
    nodes=[]
    for path in reports.glob('formal_resources_node*.jsonl'):
        try:
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-16384))
                lines=stream.read().splitlines()
            nodes.append(json.loads(lines[-1]))
        except (OSError,ValueError,IndexError):
            continue
    gpus=[gpu for node in nodes for gpu in node.get('gpus',[])]
    if not gpus:return {}
    return {'resources/gpu_utilization_mean_percent':sum(g['utilization.gpu'] for g in gpus)/len(gpus),
            'resources/gpu_utilization_min_percent':min(g['utilization.gpu'] for g in gpus),
            'resources/gpu_used_GiB_max':max(g['memory.used'] for g in gpus)/1024,
            'resources/gpu_power_mean_watts':sum(g['power.draw'] for g in gpus)/len(gpus),
            'resources/cpu_cores_used_total':sum(n.get('cpu_cores_used',0) for n in nodes),
            'resources/host_memory_GiB_max':max(int(n.get('memory.current',n.get('memory.usage_in_bytes',0))) for n in nodes)/2**30,
            'health/resource_nodes':len(nodes),
            'health/resource_snapshot_max_age_seconds':max(0,time.time()-min(n['time'] for n in nodes))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--monitor-dir',type=Path,default=DEFAULT/'setup/training_monitor_64gpu_20261022')
    p.add_argument('--reports-dir',type=Path,default=DEFAULT/'setup/action_random_64gpu_20261022/results')
    p.add_argument('--poll-seconds',type=float,default=15)
    p.add_argument('--event-dir',type=Path,default=DEFAULT/'setup/training_monitor_20261022/tensorboard/action-random-64gpu')
    p.add_argument('--microsteps-per-epoch',type=int,required=True)
    p.add_argument('--once',action='store_true');args=p.parse_args()
    root=args.monitor_dir;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'bridge.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    log=args.run_dir/'metrics.jsonl';state_path=root/'bridge_state.json'
    if state_path.exists():state=json.loads(state_path.read_text())
    else:state={'schema_version':1,'run_dir':str(args.run_dir),'inode':log.stat().st_ino,'offset':0,'last_step':0,'optimizer_step':0,
                'pending_losses':[],'world_size':64,'microbatch':48,'gradient_accumulation':1,'microsteps_per_epoch':args.microsteps_per_epoch,
                'source_batches':{'egodex':0,'robot':0}}
    assert state['run_dir']==str(args.run_dir),'Use a separate monitor directory for another run'
    writer=EventFileWriter(str(args.event_dir),flush_secs=5)
    try:
        while True:
            st=log.stat()
            if st.st_ino!=state['inode'] or st.st_size<state['offset']:raise RuntimeError('Input log rotated/truncated; refusing to mix histories')
            rows,offset=read_complete_rows(log,state['offset']);consume(writer,rows,state)
            health=rank_snapshot(args.reports_dir,state['world_size'])
            for key,tag in [('readable_ranks','health/readable_ranks'),('max_report_age_seconds','health/max_rank_report_age_seconds'),('peak_allocated_GiB_max_rank','resources/peak_allocated_GiB_max_rank')]:
                if key in health:scalar(writer,tag,health[key],state['last_step'])
            for tag,value in resource_snapshot(args.reports_dir).items():scalar(writer,tag,value,state['last_step'])
            writer.flush();state['offset']=offset;atomic_json(state_path,state)
            status={'captured_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'run_dir':str(args.run_dir),'last_microstep':state['last_step'],
                    'optimizer_updates':state['optimizer_step'],'metrics_age_seconds':max(0,time.time()-log.stat().st_mtime),
                    'new_rows':len(rows),'latest_metrics':state.get('last_metrics'),**health}
            atomic_json(root/'status.json',status)
            if rows:print(json.dumps(status),flush=True)
            if args.once:break
            time.sleep(args.poll_seconds)
    finally:writer.close();lock.close()

if __name__=='__main__':main()
