"""Production trainer on 32 GPUs with the 8-node recipe; one run per variant (base | s7d).

Checks: exact retained pretrained weights, identical fresh ActionDiT on every rank, per-sample
batch shapes, finite losses, and all three sources seen. Rank 0 appends every step (sample ids,
losses, LR) to <run>/metrics.jsonl so base and s7d can be compared step by step; step timing
and MFU go to <run>/step_timing.jsonl.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import traceback

import torch
from hydra import compose, initialize_config_dir
from safetensors import safe_open
from openwam.dataloader.registry import build_dataset
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.seeding import seed_everything
from scripts.train import _apply_gpu_memory_cap, _build_accelerator, _inject_project_seed

R=Path('/mnt/pfs/v89atp/workspace/setup/accel_cmp_32gpu_20261026')
WORLD=32

def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def write_json(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

class ExperimentTrainer(OpenWAMTrainer):
    def __init__(self,*args,variant,**kwargs):
        self.variant=variant;self.history=[];self.started=time.time();self.seen=set();self.lr_horizon=None
        self.release_hash=hashlib.sha256((R/'results/release_manifest.json').read_bytes()).hexdigest()
        super().__init__(*args,**kwargs)
        assert self.accelerator.num_processes==WORLD
        arch=self.architecture
        for branch in ('action_backbone','proprio_encoder','video_backbone.dit'):
            assert all(p.requires_grad for p in arch.get_submodule(branch).parameters())
        for branch in ('video_backbone.text_encoder','video_backbone.vae'):
            assert not any(p.requires_grad for p in arch.get_submodule(branch).parameters())
        provenance=arch.initialization_provenance
        assert len(provenance['random_tensor_names'])==824
        dig=hashlib.sha256()
        for name,t in arch.action_backbone.state_dict().items():dig.update(name.encode());dig.update(sha(t).encode())
        hashes=[None]*WORLD;torch.distributed.all_gather_object(hashes,dig.hexdigest())
        assert len(set(hashes))==1,hashes
        if self.accelerator.is_main_process:
            with safe_open(provenance['source_weights'],framework='pt',device='cpu') as source:
                actual=arch.state_dict()
                for name in provenance['retained_tensor_names']:
                    assert torch.equal(actual[name].cpu(),source.get_tensor(name)),name
                assert not torch.equal(actual['action_backbone.action_encoder.weight'],source.get_tensor('action_backbone.action_encoder.weight'))
            write_json(R/'results'/f'{variant}_initialization.json',{
                'status':'passed','all_retained_tensors_exact':True,'retained_tensors':len(provenance['retained_tensor_names']),
                'random_tensors':824,'full_action_sha256':hashes[0],'all_initial_action_hashes_equal':True,
                'seed':42,'source_weights':provenance['source_weights']})
        self.accelerator.wait_for_everyone()
        prepare=self.architecture.prepare_inputs
        def checked_prepare(batch):
            assert len(batch)==int(self.cfg.training.batch_size),len(batch)
            inputs=prepare(batch)
            self.batch_shapes={key:list(inputs[key].shape) for key in ('input_latents','context','actions','proprio')}
            assert all(shape[0]==len(batch) for shape in self.batch_shapes.values()),self.batch_shapes
            return inputs
        self.architecture.prepare_inputs=checked_prepare

    def build_lr_scheduler(self,optimizer,total_opt_steps,debug=False):
        # LR follows the 8-node run's schedule; max_steps only stops the run early.
        self.lr_horizon=int(self.cfg.experiment.lr_horizon_opt_steps)
        assert total_opt_steps<=self.lr_horizon,(total_opt_steps,self.lr_horizon)
        return super().build_lr_scheduler(optimizer,self.lr_horizon,debug=debug)

    def compute_loss(self,batch):
        self.sources=[s['_dataset_name'] for s in batch];self.seen.update(self.sources)
        self.sample_ids=[{k:str(s.get(k)) for k in ('episode_id','episode_start_frame','_dataset_name','_dataset_index')} for s in batch]
        self.video_only=all(s.get('training_modality')=='video_only' for s in batch)
        losses=super().compute_loss(batch)
        assert all(torch.isfinite(v).all() for v in losses.values()),self.sample_ids
        if self.video_only:assert float(losses['action'])==0
        self.losses={k:float(v.detach()) for k,v in losses.items()}
        return losses

    def log_step(self,**kw):
        if self.accelerator.is_main_process:super().log_step(**kw)
        record={'step':kw['global_step'],'optimizer_step':kw['opt_step'],'lr':kw['lr'],'sources':self.sources,
                'sample_ids':self.sample_ids,'video_only':self.video_only,'local_losses':self.losses,
                'metrics':{k:float(v) for k,v in kw['metrics'].items()},'steps_per_sec':kw['steps_per_sec'],
                'batch_shapes':self.batch_shapes}
        assert all(torch.isfinite(torch.tensor(v)) for v in record['metrics'].values())
        if self.accelerator.is_main_process:
            with (Path(kw['output_path'])/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('TRAIN_PROGRESS '+json.dumps({k:record[k] for k in ('step','lr','metrics','steps_per_sec')}),flush=True)
        self.history=[record]
        self.write_report('running',kw['output_path'])

    def finish_training(self,output_path,global_step,save_steps,keep_last_k,is_main,wandb_run):
        super().finish_training(output_path,global_step,save_steps,keep_last_k,is_main,wandb_run)
        sources=[None]*WORLD;torch.distributed.all_gather_object(sources,sorted(self.seen))
        assert set().union(*map(set,sources))==set(self.dataset._names),sources
        self.accelerator.wait_for_everyone();self.write_report('passed',output_path)

    def write_report(self,status,output_path):
        write_json(R/'results'/f'{self.variant}_rank{self.accelerator.process_index}.json',{
            'status':status,'variant':self.variant,'world_size':WORLD,'microbatch':int(self.cfg.training.batch_size),
            'release_manifest_sha256':self.release_hash,
            'gradient_accumulation':int(self.cfg.training.gradient_accumulation_steps),
            'lr_horizon_opt_steps':self.lr_horizon,'max_steps':int(self.cfg.training.max_steps),
            'zero_stage':2,'precision':'bf16','history':self.history,
            'output_path':output_path,'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
            'elapsed_seconds':time.time()-self.started})

def main():
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['base','s7d']);a=p.parse_args()
    torch.set_num_threads(4);logging.basicConfig(level=logging.WARNING)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group('nccl',timeout=timedelta(seconds=1200))
    with initialize_config_dir(config_dir=str(R/'release/configs'),version_base=None):
        cfg=compose(config_name=f'train_accel_cmp_{a.variant}')
    assert cfg.training.batch_size==48 and cfg.training.gradient_accumulation_steps==1
    _inject_project_seed(cfg);seed_everything(cfg.project.seed,rank=int(os.environ['RANK']))
    _apply_gpu_memory_cap(cfg)
    accelerator=_build_accelerator(cfg);dataset=build_dataset(cfg.dataloader)
    trainer=ExperimentTrainer(cfg,accelerator=accelerator,dataset=dataset,variant=a.variant)
    trainer.train();accelerator.wait_for_everyone();torch.distributed.destroy_process_group()

if __name__=='__main__':
    try:
        main()
    except BaseException:
        # A failed rank must take its node down: otherwise peers block in NCCL and the
        # job keeps its GPUs (and their memory) until the collective timeout.
        traceback.print_exc()
        os._exit(1)
