from pathlib import Path

from hydra import compose, initialize_config_dir


def test_fresh_bs48_configuration_preserves_initialization_and_s9():
    with initialize_config_dir(config_dir=str(Path(__file__).parents[1]/"configs"),version_base=None):
        cfg=compose(config_name="train_touchwam_six_source_64gpu_bs48_20261030")
    t=cfg.training
    assert t.batch_size==48 and t.dataset_num_workers==8
    assert t.learning_rate==1e-4 and t.warmup_ratio==.05
    assert t.resume_ckpt_path is None and t.initialize_touch_from_openwam
    assert "OpenWAM-Alpha-Pretrain-Foundation-Model" in t.finetune_ckpt_path
    assert t.use_gradient_checkpointing and not t.use_gradient_checkpointing_offload
    assert list(t.gradient_checkpointing_layers)==list(range(20))
    assert t.gradient_checkpointing_layers_video_only is None
    assert cfg.model.architecture.mot_checkpoint_mixed_attn
    assert cfg.model.architecture.tactile.observation_encoder.freeze_pretrained
    assert t.num_epochs==1 and t.max_steps is None and t.stop_after_steps is None
    assert t.save_full_states_for_resume and t.save_steps==500


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
