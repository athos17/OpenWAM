#!/usr/bin/env python3
"""P3: a training run's deploy safetensors against its resumable DCP state, on CPU.

Builds the architecture the way finetune / deploy does (``build_architecture_from_ckpt_dir``,
which loads the run's latest ``checkpoint_step_*.safetensors``), then compares every tensor of
its state dict with the model state in ``accel_state_step_<N>/dcp`` cast to the same dtype.

  python -m scripts.validation.touchwam_export_check <run_dir> <step>
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("step", type=int)
    args = parser.parse_args()
    import torch
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

    from openwam.train.utils.ckpt_model_loader import build_architecture_from_ckpt_dir

    run = Path(args.run_dir)
    weights = run / f"checkpoint_step_{args.step}.safetensors"
    latest = max(run.glob("checkpoint_step_*.safetensors"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if latest != weights:
        raise SystemExit(f"latest weights are {latest.name}, not step {args.step}")
    _, arch, _ = build_architecture_from_ckpt_dir(str(run), weights_required=True)
    exported = arch.state_dict()
    dcp_dir = run / f"accel_state_step_{args.step}" / "dcp"
    trained = _load_state_dict_from_keys({f"model.{k}" for k in exported}, checkpoint_id=str(dcp_dir))
    missing = sorted(k for k in exported if f"model.{k}" not in trained)
    worst, mismatched = 0.0, []
    for name, tensor in exported.items():
        ref = trained.get(f"model.{name}")
        if ref is None:
            continue
        ref = ref.to(tensor.dtype)
        if ref.shape != tensor.shape:
            mismatched.append(name)
            continue
        diff = (ref.float() - tensor.float()).abs().max().item() if tensor.numel() else 0.0
        worst = max(worst, diff)
        if diff:
            mismatched.append(name)
    report = {"weights": str(weights), "tensors": len(exported), "missing_in_dcp": missing[:20],
              "n_missing": len(missing), "n_mismatched": len(mismatched), "mismatched": mismatched[:20],
              "max_abs_diff": worst, "status": "passed" if not missing and not mismatched else "failed"}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
