"""Use the production trainer; add assertions and durable run observations.

smoke/resume: actual batch3072, graph switching and full-state recovery;
formal: unmodified training horizon, scheduler, sampler and checkpoint lifecycle.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import torch
from hydra import compose, initialize_config_dir
from safetensors import safe_open
from openwam.dataloader.registry import build_dataset
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.checkpointing import save_weights, save_full_state
from openwam.train.utils.seeding import seed_everything
from scripts.train import _build_accelerator, _inject_project_seed

R=Path('/mnt/pfs/v89atp/workspace/setup/action_random_64gpu_20261022')

def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def write_json(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

class ExperimentTrainer(OpenWAMTrainer):
    def __init__(self,*args,phase,**kwargs):
        self.phase=phase;self.history=[];self.started=time.time();self.resume_proof=None;self.seen=set()
        manifest='release_manifest.json'
        self.release_hash=hashlib.sha256((R/'results'/manifest).read_bytes()).hexdigest()
        super().__init__(*args,**kwargs)
        assert self.accelerator.num_processes==64
        arch=self.architecture
        for branch in ('action_backbone','proprio_encoder','video_backbone.dit'):
            assert all(p.requires_grad for p in arch.get_submodule(branch).parameters())
        for branch in ('video_backbone.text_encoder','video_backbone.vae'):
            assert not any(p.requires_grad for p in arch.get_submodule(branch).parameters())
        if phase!='resume':
            provenance=arch.initialization_provenance
            assert len(provenance['random_tensor_names'])==824
            # Hash the full fresh ActionDiT, including standalone parameters.
            dig=hashlib.sha256()
            for name,t in arch.action_backbone.state_dict().items():dig.update(name.encode());dig.update(sha(t).encode())
            hashes=[None]*64;torch.distributed.all_gather_object(hashes,dig.hexdigest())
            assert len(set(hashes))==1,hashes
            if self.accelerator.is_main_process:
                with safe_open(provenance['source_weights'],framework='pt',device='cpu') as source:
                    actual=arch.state_dict()
                    for name in provenance['retained_tensor_names']:
                        assert torch.equal(actual[name].cpu(),source.get_tensor(name)),name
                    # All randomized values come from constructor; constants may coincide.
                    assert not torch.equal(actual['action_backbone.action_encoder.weight'],source.get_tensor('action_backbone.action_encoder.weight'))
                write_json(R/'results'/f'{phase}_initialization.json',{
                    'status':'passed','all_retained_tensors_exact':True,'retained_tensors':len(provenance['retained_tensor_names']),
                    'random_tensors':824,'full_action_sha256':hashes[0],'all_64_initial_action_hashes_equal':True,
                    'seed':42,'source_weights':provenance['source_weights']})
            self.accelerator.wait_for_everyone()
        self.before=None
        prepare=self.architecture.prepare_inputs
        def checked_prepare(batch):
            assert len(batch)==int(self.cfg.training.batch_size),len(batch)
            inputs=prepare(batch)
            self.batch_shapes={key:list(inputs[key].shape) for key in ('input_latents','context','actions','proprio')}
            assert all(shape[0]==len(batch) for shape in self.batch_shapes.values()),self.batch_shapes
            return inputs
        self.architecture.prepare_inputs=checked_prepare
        action_loss=self.architecture._compute_action_loss
        def observed_action_loss(noise_pred,target,timestep_ids,scheduler,inputs,device):
            self.action_loss_details={
                'timestep_ids':timestep_ids.tolist(),
                'weights':scheduler.training_weight(timestep_ids).tolist(),
                'valid_cells':(~inputs['action_is_pad']).flatten(1).sum(1).tolist(),
            }
            return action_loss(noise_pred,target,timestep_ids,scheduler,inputs,device)
        self.architecture._compute_action_loss=observed_action_loss

    def tracked_parameters(self):
        arch=self.accelerator.unwrap_model(self.architecture);params=dict(arch.named_parameters())
        names=['video_backbone.dit.patch_embedding.weight','action_backbone.action_encoder.weight',
               'action_backbone.action_decoder.weight','proprio_encoder.weight']
        return {name:sha(params[name]) for name in names}

    def prepare_accelerate(self,*args,**kwargs):
        result=super().prepare_accelerate(*args,**kwargs)
        self.before=self.tracked_parameters()
        return result

    def optimizer_steps(self):
        obj=self.architecture.optimizer
        while not isinstance(obj,torch.optim.Optimizer) and hasattr(obj,'optimizer'):obj=obj.optimizer
        return sorted({int(v['step']) for v in obj.state.values() if 'step' in v})

    def compute_loss(self,batch):
        self.sources=[s['_dataset_name'] for s in batch];self.seen.update(self.sources)
        self.sample_ids=[{k:str(s.get(k)) for k in ('episode_id','episode_start_frame','_dataset_name','_dataset_index')} for s in batch]
        self.video_only=all(s.get('training_modality')=='video_only' for s in batch)
        self.action_loss_details=None
        losses=super().compute_loss(batch)
        assert all(torch.isfinite(v).all() for v in losses.values()),self.sample_ids
        assert float(losses['action'])>=0
        if self.video_only:
            assert float(losses['action'])==0 and self.action_loss_details is None
        else:
            # The BSMNTW action weight is exactly zero at timestep ID 0.
            # A zero weighted loss is valid; verify the action branch ran.
            assert self.action_loss_details is not None
        self.losses={k:float(v.detach()) for k,v in losses.items()}
        return losses

    def log_step(self,**kw):
        # Avoid 64 progress bars; keep durable per-rank records and rank-0 JSON metrics.
        if self.accelerator.is_main_process:super().log_step(**kw)
        step=kw['global_step'];opt=kw['opt_step'];boundary=step%int(self.cfg.training.gradient_accumulation_steps)==0
        record={'step':step,'optimizer_step':opt,'lr':kw['lr'],'sources':self.sources,'sample_ids':self.sample_ids,
                'video_only':self.video_only,'local_losses':self.losses,
                'metrics':{k:float(v) for k,v in kw['metrics'].items()},'steps_per_sec':kw['steps_per_sec'],
                'batch_shapes':self.batch_shapes}
        record['action_loss_details']=self.action_loss_details
        assert all(torch.isfinite(torch.tensor(v)) for v in record['metrics'].values())
        if self.phase!='formal' and boundary:
            arch=self.accelerator.unwrap_model(self.architecture)
            invalid=[name for name,p in arch.named_parameters() if not torch.isfinite(p).all()]
            assert not invalid,invalid
            current=self.tracked_parameters();hashes=[None]*64
            torch.distributed.all_gather_object(hashes,current)
            assert all(h==hashes[0] for h in hashes),'Parameter hashes diverged'
            record.update(parameter_sha256=current,all_parameters_finite=True,tracked_parameters_equal_across_ranks=True,
                          optimizer_state_steps=self.optimizer_steps())
            assert self.optimizer_steps()==[opt],record
        if self.phase=='resume':
            reference=json.loads((R/'results'/f'smoke_rank{self.accelerator.process_index}.json').read_text())
            previous=next(x for x in reference['history'] if x['step']==step)
            assert previous['sample_ids']==record['sample_ids']
            assert previous['lr']==record['lr']
            record['sample_order_matches']=True
        if self.phase=='formal':
            if self.accelerator.is_main_process:
                with (Path(kw['output_path'])/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            self.history=[record]
        else:self.history.append(record)
        self.write_report('running',kw['output_path'])
        if self.accelerator.is_main_process:print('TRAIN_PROGRESS '+json.dumps(record),flush=True)
        if self.phase=='smoke' and step==4:
            save_weights(self.accelerator,self.architecture,kw['output_path'],step,final=False)
            save_full_state(self.accelerator,kw['output_path'],step,opt,kw['epoch'])
            if self.accelerator.is_main_process:(R/'results/resume_run_path.txt').write_text(kw['output_path']+'\n')
            self.accelerator.wait_for_everyone()

    def resume_if_configured(self,*args,**kwargs):
        position=super().resume_if_configured(*args,**kwargs)
        assert position[:2]==(4,4),position
        original=json.loads((R/'results'/f'smoke_rank{self.accelerator.process_index}.json').read_text())
        checkpoint=next(x for x in original['history'] if x['step']==4)
        assert self.tracked_parameters()==checkpoint['parameter_sha256']
        assert self.optimizer_steps()==[4]
        saved_path=Path(args[0])/'pytorch_model'/f'bf16_zero_pp_rank_{self.accelerator.process_index}_mp_rank_00_optim_states.pt'
        saved=torch.load(saved_path,map_location='cpu',weights_only=False,mmap=True)['optimizer_state_dict']
        zero=self.architecture.optimizer
        for live,expected in zip(zero.single_partition_of_fp32_groups,saved['single_partition_of_fp32_groups']):
            pos=torch.linspace(0,expected.numel()-1,256,dtype=torch.int64)
            assert torch.equal(live.detach()[pos.to(live.device)].cpu(),expected[pos])
        for idx,state in saved['base_optimizer_state']['state'].items():
            for key,expected in state.items():
                live=zero.optimizer.state_dict()['state'][idx][key]
                if isinstance(expected,torch.Tensor) and expected.numel()>1:
                    pos=torch.linspace(0,expected.numel()-1,256,dtype=torch.int64)
                    assert torch.equal(live.detach().flatten()[pos.to(live.device)].cpu(),expected.flatten()[pos]),key
                else:assert live==expected
        self.before=self.tracked_parameters()
        self.resume_proof={'restored_parameter_hashes':True,'sampled_fp32_masters_and_adam_moments':True,'action_not_reset':True}
        return position

    def finish_training(self,output_path,global_step,save_steps,keep_last_k,is_main,wandb_run):
        if self.phase=='formal':
            super().finish_training(output_path,global_step,save_steps,keep_last_k,is_main,wandb_run)
        else:
            expected=8
            assert global_step==expected
            changed={k:v!=self.before[k] for k,v in self.tracked_parameters().items()}
            assert all(changed.values()),changed
            sources=[None]*64;torch.distributed.all_gather_object(sources,sorted(self.seen))
            assert set().union(*map(set,sources))==set(self.dataset._names),sources
            if self.phase=='smoke':save_weights(self.accelerator,self.architecture,output_path,global_step,final=True)
        self.accelerator.wait_for_everyone();self.write_report('passed',output_path)

    def write_report(self,status,output_path):
        write_json(R/'results'/f'{self.phase}_rank{self.accelerator.process_index}.json',{
            'status':status,'phase':self.phase,'world_size':64,'microbatch':int(self.cfg.training.batch_size),
            'release_manifest_sha256':self.release_hash,
            'gradient_accumulation':int(self.cfg.training.gradient_accumulation_steps),
            'zero_stage':2,'precision':'bf16','history':self.history,'resume_proof':self.resume_proof,
            'output_path':output_path,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.time()-self.started})

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','resume','formal']);a=p.parse_args()
    torch.set_num_threads(4);logging.basicConfig(level=logging.WARNING)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group('nccl',timeout=timedelta(seconds=1200))
    overrides=[]
    if a.phase!='formal':
        overrides=['training.num_epochs=null','training.max_steps=8','training.lr_scheduler=null',
                   'training.save_steps=0','training.output_path='+str(R/'debug_runs'/a.phase)]
    if a.phase=='resume':
        overrides+=['training.finetune_ckpt_path=null',
                    'training.resume_ckpt_path='+(R/'results/resume_run_path.txt').read_text().strip()]
    with initialize_config_dir(config_dir=str(R/'release/configs'),version_base=None):
        cfg=compose(config_name='train_action_random_64gpu',overrides=overrides)
    assert cfg.training.batch_size==48 and cfg.training.gradient_accumulation_steps==1
    _inject_project_seed(cfg);seed_everything(cfg.project.seed,rank=int(os.environ['RANK']))
    accelerator=_build_accelerator(cfg);dataset=build_dataset(cfg.dataloader)
    trainer=ExperimentTrainer(cfg,accelerator=accelerator,dataset=dataset,phase=a.phase)
    trainer.train();accelerator.wait_for_everyone();torch.distributed.destroy_process_group()

if __name__=='__main__':main()
