# 64-GPU training dashboard

The independent CPU-only bridge reads the new run's `metrics.jsonl`, 64 rank
reports and eight node resource logs. It writes events into the existing
TensorBoard server's `action-random-64gpu` subdirectory, preserving the old
32-GPU curves. It never modifies training state.

Use the existing TensorBoard virtual environment on dev-cpu. Pass the formal
run directory with `--run-dir` and `--microsteps-per-epoch 23845`; this run has
64 ranks, batch 48 and accumulation 1. The local forwarded URL is
http://127.0.0.1:16006/#scalars .

GPU utilization and power are instantaneous 15-second samples, not sustained
throughput measurements. Loss/action pools AgiBot and SharpaDex; EgoDex video
loss is separate. Event wall time is ingestion time; compare by step.
