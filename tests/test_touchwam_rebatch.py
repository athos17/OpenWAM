import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from openwam.dataloader.modality_sampler import GlobalModalitySampler
from openwam.train.utils.training_utils import build_cosine_scheduler

spec=importlib.util.spec_from_file_location("stage_rebatch", Path(__file__).parents[1]/"scripts/validation/stage_touchwam_rebatch.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    contract=dict(training_layout=dict(world_size=2,microbatch=2,gradient_accumulation=1,
        batching="global_modality_codec_observation_v4"), schema_version=1,
        sampling_algorithm="full_passes_random_partial_v1", seed=42,virtual_windows=160,
        source_order=["v","a","t"],weights=dict(v=.5,a=.3,t=.2),
        sources={"v":dict(windows=80,training_modality="video_only"),
                 "a":dict(windows=48,training_modality="video_action"),
                 "t":dict(windows=32,training_modality="video_tactile",tactile_observation_shape=[16,10])})
    modes=[contract["sources"][k]["training_modality"] for k in contract["source_order"]]
    index=np.array([[i,j] for i,n in enumerate([80,48,32]) for j in range(n)])
    np.random.RandomState(42).shuffle(index)
    dataset=SimpleNamespace(names=["v","a","t"],_seed=42,_index_map=index,
        source_modalities=modes,source_tactile_codecs=[None,None,"tactile_intensity_v1"],
        source_tactile_observation_shapes=[None,None,[16,10]],
        dataset_sample_counts=lambda:dict(v=80,a=48,t=32),set_epoch=lambda epoch:None)
    cfg=OmegaConf.create(dict(training=dict(learning_rate=.001,warmup_ratio=.5,lr_min_ratio=.01)))
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))],lr=.001)
    scheduler=build_cosine_scheduler(optimizer,total_opt_steps=40,cfg=cfg,num_processes=2)
    for _ in range(8): optimizer.step(); scheduler.step()
    meta=dict(epoch=0,global_step=4,opt_step=4,batches_per_epoch=40,optimizer_boundary=True)
    return contract,dataset,cfg,optimizer,scheduler,meta


def test_rebatch_preserves_exact_remaining_windows_and_layout():
    c,ds,cfg,opt,sch,meta=fixture()
    old=list(GlobalModalitySampler(ds,2,2))
    plan=module.make_plan(c,meta,sch.state_dict(),8)
    sampler=GlobalModalitySampler(ds,8,2,rebatch=plan)
    new=list(sampler)
    consumed=set(old[:16]); remaining=set(old[16:])
    assert set(new)==remaining
    assert not consumed.intersection(new)
    assert len(new)==len(sampler)==plan["remaining_steps"]*16
    for i in range(0,len(new),16):
        assert len({ds.source_modalities[ds._index_map[j,0]] for j in new[i:i+16]})==1
    assert list(sampler)==new
    assert plan["consumed_unique_virtual_windows"]==16
    with pytest.raises(ValueError,match="epoch"): sampler.set_epoch(1)


def test_rebatch_scheduler_matches_original_at_equal_sample_counts_and_resumes():
    c,ds,cfg,opt,old,meta=fixture()
    plan=module.make_plan(c,meta,old.state_dict(),8)
    cfg.training.rebatch=plan
    new_opt=torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))],lr=.001)
    new=build_cosine_scheduler(new_opt,total_opt_steps=plan["remaining_steps"],cfg=cfg,num_processes=2)
    assert new.get_last_lr()==pytest.approx(old.get_last_lr(),rel=1e-12)
    for step in range(1,10):
        for _ in range(8): opt.step();old.step()
        for _ in range(2): new_opt.step();new.step()
        assert new.get_last_lr()==pytest.approx(old.get_last_lr(),rel=1e-10)
        if step==2:
            state=copy.deepcopy(new.state_dict()); opt_state=copy.deepcopy(new_opt.state_dict())
            rebuilt_opt=torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))],lr=.001)
            rebuilt=build_cosine_scheduler(rebuilt_opt,total_opt_steps=plan["remaining_steps"],cfg=cfg,num_processes=2)
            rebuilt_opt.load_state_dict(opt_state);rebuilt.load_state_dict(state)
            new_opt,new=rebuilt_opt,rebuilt


@pytest.mark.parametrize("bad",["seed","counts","offset","batch"])
def test_rebatch_rejects_changed_population(bad):
    c,ds,cfg,opt,sch,meta=fixture()
    p=module.make_plan(c,meta,sch.state_dict(),8)
    if bad=="seed": p["seed"]+=1
    if bad=="counts": p["bucket_counts"]["0"]+=1
    if bad=="offset": p["consumed_by_bucket"]["0"]=-1
    if bad=="batch": p["target_global_batch"]+=1
    with pytest.raises(ValueError):GlobalModalitySampler(ds,8,2,rebatch=p)


def test_local_stage_trace_preserves_slow_data_wait(tmp_path):
    from openwam.train.utils.step_timing import StepTimer
    import json
    timer=StepTimer(enabled=True,output_path=str(tmp_path),record_steps=2,summary_every=50)
    for i,value in enumerate([.001,6.0],1):
        timer.last={"data_wait_cpu_s":value};timer.record_step(i,{"samples_video_only":16})
    rows=[json.loads(x) for x in (tmp_path/"step_trace_rank0.jsonl").read_text().splitlines()]
    assert [x["data_wait_cpu_s"] for x in rows]==[.001,6.0]
    timer.record_step(3,{})
    assert len((tmp_path/"step_trace_rank0.jsonl").read_text().splitlines())==2


def test_staging_preserves_source_tensor_and_adam_files(tmp_path):
    import json
    c,ds,cfg,opt,sch,meta=fixture()
    cfg.training.step_timing={}
    cfg.model={"architecture":{"mot_checkpoint_mixed_attn":False}}
    cfg.project={"name":"source"}
    source=tmp_path/"source";source.mkdir()
    state=source/"accel_state_step_4";state.mkdir()
    (state/"pytorch_model").mkdir()
    for rank in range(2):
        torch.save(opt.state_dict(),state/"pytorch_model"/f"rank{rank}_optim_states.pt")
        torch.save({"rng":rank},state/f"random_states_{rank}.pkl")
    torch.save({"weight":torch.ones(2)},state/"pytorch_model/mp_rank_00_model_states.pt")
    torch.save(sch.state_dict(),state/"scheduler.bin")
    (state/"trainer_state.json").write_text(json.dumps(meta))
    (source/"training_data_contract.json").write_text(json.dumps(c))
    (source/"checkpoint_step_4.safetensors").write_bytes(b"fixture")
    for name in ("normalization","tokenizer"):(source/name).mkdir()
    OmegaConf.save(cfg,source/"config.yaml")
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}
    target=tmp_path/"target"
    proof=module.stage(source,target,4,target_batch=8)
    assert proof["initial_lr"]==pytest.approx(sch.get_last_lr())
    for path,value in before.items():assert (source/path).read_bytes()==value
    for f in (state/"pytorch_model").iterdir():
        assert f.samefile(target/"accel_state_step_0/pytorch_model"/f.name)
    assert not (state/"scheduler.bin").samefile(target/"accel_state_step_0/scheduler.bin")
    assert json.loads((target/"accel_state_step_0/trainer_state.json").read_text())["global_step"]==0
    assert OmegaConf.load(target/"config.yaml").model.architecture.mot_checkpoint_mixed_attn
