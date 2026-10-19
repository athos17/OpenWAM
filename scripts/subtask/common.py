"""Shared helpers for the Marvin pick-place subtask annotation pipeline.

Pipeline (all outputs under ``--out``, default ``outputs/subtask``):
  segment.py   P0  proprio rules -> events + candidate segments (segments_rules.parquet)
  keyframes.py P1  per-segment keyframes (+ per-episode contact sheets)
  label.py     P1  multimodal LLM labels per segment (two models, cached per call)
  vocab.py     P1  canonical object names, prompts, subtasks.parquet
  review.py        HTML review page (timeline + keyframes + labels)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30.0
SOURCE_DIM = 76
HAND_SLICE = {"left": slice(14, 36), "right": slice(36, 58)}
EEF_XYZ = {"left": slice(58, 61), "right": slice(67, 70)}
HEAD_CAMERA = "observation.images.left_head"
WRIST_CAMERA = {"left": "observation.images.left_wrist", "right": "observation.images.right_wrist"}


# Label generations live side by side under llm/<kind>/; v3 = pre-grasp wrist frames + fixed taxonomy.
LABEL_KIND = "segment_v3"
ARBITRATION_KIND = "arbitration_v3"


def label_path(out: Path, model: str, seg, kind: str = LABEL_KIND) -> Path:
    """LLM label cache for one rule segment. A PICK is keyed by its grasp/release frames only, so
    labels survive boundary changes; other segments are keyed by their frame range."""
    ev = json.loads(seg.events) if isinstance(seg.events, str) and seg.events else {}
    key = f"g{ev['grasp']:05d}_r{ev['release']:05d}" if ev else f"{int(seg.start):05d}_{int(seg.end):05d}"
    return out / "llm" / kind / model / f"ep{int(seg.episode_index):03d}_{seg.type}_{key}.json"


def flexion_columns(dataset_dir: Path) -> dict:
    """Per-hand state indices of flexion joints (FE / PIP / DIP / IP), by name."""
    names = json.loads((dataset_dir / "meta" / "info.json").read_text())["features"]["observation.state"]["names"]
    out = {}
    for hand, sl in HAND_SLICE.items():
        out[hand] = [i for i in range(sl.start, sl.stop) if names[i].rsplit("_", 1)[-1] in ("FE", "PIP", "DIP", "IP")]
    return out


def episode_table(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


def load_state(dataset_dir: Path, ep_row) -> np.ndarray:
    path = dataset_dir / "data" / f"chunk-{int(ep_row['data/chunk_index']):03d}" / f"file-{int(ep_row['data/file_index']):03d}.parquet"
    df = pd.read_parquet(path, columns=["observation.state", "episode_index", "frame_index"])
    df = df[df["episode_index"] == int(ep_row["episode_index"])]
    if not (df["frame_index"].to_numpy() == np.arange(len(df))).all():
        raise ValueError(f"episode {ep_row['episode_index']}: non-contiguous frame_index")
    return np.stack(df["observation.state"].values).astype(np.float32)


def video_path(dataset_dir: Path, ep_row, camera: str) -> Path:
    chunk = int(ep_row[f"videos/{camera}/chunk_index"])
    file = int(ep_row[f"videos/{camera}/file_index"])
    return dataset_dir / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"


def read_frames(path: Path, indices, max_width: int | None = None) -> dict:
    """Decode the requested frame indices (one sequential pass) -> {index: RGB uint8 array}."""
    import av
    import cv2

    wanted = sorted({int(i) for i in indices})
    out = {}
    if not wanted:
        return out
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        k = 0
        for n, frame in enumerate(container.decode(stream)):
            if n == wanted[k]:
                img = frame.to_ndarray(format="rgb24")
                if max_width and img.shape[1] > max_width:
                    h = round(img.shape[0] * max_width / img.shape[1])
                    img = cv2.resize(img, (max_width, h), interpolation=cv2.INTER_AREA)
                out[n] = img
                k += 1
                if k == len(wanted):
                    break
    missing = set(wanted) - set(out)
    if missing:
        raise ValueError(f"{path}: frames {sorted(missing)[:5]} beyond end of video")
    return out


def save_jpeg(path: Path, img: np.ndarray, quality: int = 88) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
