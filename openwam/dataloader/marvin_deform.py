"""Converted Marvin 76D -> 62D EEF/hand data with audited deform sidecars."""

import hashlib

import numpy as np

from openwam.dataloader.sharpadex_tianji import SharpaDexTianjiDataset
from openwam.tactile.deform_store import AlignedDeformStore


class MarvinDeformDataset(SharpaDexTianjiDataset):
    DATASET_NAME = "MarvinDeform"
    CONFIG_KEYS = SharpaDexTianjiDataset.CONFIG_KEYS + (
        "tactile_manifest_path",
        "tactile_history",
        "tactile_map_size",
        "tactile_future_frames",
        "tactile_future_stride",
        "tactile_fps",
    )

    def __init__(
        self,
        *args,
        tactile_manifest_path,
        tactile_history=4,
        tactile_map_size=32,
        tactile_future_frames=8,
        tactile_future_stride=3,
        tactile_fps=30,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        values = (tactile_history, tactile_map_size, tactile_future_frames, tactile_future_stride)
        if any(type(v) is not int or v < 1 for v in values) or tactile_fps != self._fps:
            raise ValueError("Invalid tactile dimensions or mismatched dataset fps")
        if tactile_future_frames * tactile_future_stride > self._num_frames - 1:
            raise ValueError("Actions must cover the future tactile horizon")
        self._touch = AlignedDeformStore(
            tactile_manifest_path, self._dataset_dir, size=tactile_map_size, fps=tactile_fps
        )
        self._touch_window = dict(
            history=tactile_history, future_frames=tactile_future_frames, future_stride=tactile_future_stride
        )
        # Filter by actual target validity before sampling, rather than retrying
        # arbitrary windows because an episode has no recorded tactile topics.
        selected = []
        future_offsets = np.arange(1, tactile_future_frames + 1) * tactile_future_stride
        for ep_local, row in self._eps_df.iterrows():
            episode = int(row["episode_index"])
            if str(episode) not in self._touch.episodes:
                continue
            maps, valid, times, present = self._touch.arrays(episode)
            begin, end = int(self._ep_valid_start[ep_local]), int(self._ep_valid_end[ep_local])
            if len(maps) < end:
                raise ValueError(f"Truncated episode {episode} deform cache")
            if not present.any():
                continue
            for raw_index in range(int(self._cum_n_starts[ep_local]), int(self._cum_n_starts[ep_local + 1])):
                offset = begin + (raw_index - int(self._cum_n_starts[ep_local])) * self._window_stride
                rows = offset + future_offsets
                if rows[-1] < end and valid[rows][:, present].all():
                    selected.append(raw_index)
        self._touch_indices = np.asarray(selected, dtype=np.int64)
        if not len(selected):
            raise ValueError("No complete tactile target windows in this split")
        self.training_data_contract = dict(
            schema_version=1,
            dataset_type="marvin_deform",
            tactile_manifest_sha256=self._touch.digest,
            window_indices_sha256=hashlib.sha256(self._touch_indices.tobytes()).hexdigest(),
            selected_windows=len(selected),
            total_windows=self._n_total,
            split=self._split,
        )

    def _resolve_cameras(self, info):
        contract = info.get("openwam_contract", {})
        if (
            info.get("codebase_version") != "v3.0"
            or contract.get("raw_action_dim") != 62
            or contract.get("rotation_6d_layout") != "[R00,R10,R20,R01,R11,R21]"
        ):
            raise ValueError("Expected converted Marvin v3, 76D storage / 62D EEF-hand contract, column rot6d")
        return "observation.images.left_head", "observation.images.left_wrist", "observation.images.right_wrist"

    def __len__(self):
        return len(self._touch_indices) if hasattr(self, "_touch_indices") else super().__len__()

    def __getitem__(self, idx):
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        return self._safe_get(idx)

    def _getitem_impl(self, idx):
        # Retry stays in the filtered index space; the hook below receives the
        # actual successfully selected episode and offset, never the requested idx.
        return super()._getitem_impl(int(self._touch_indices[idx]))

    def _observation_extras(self, idx, row, offset, valid_start, valid_end, win):
        if "timestamp" not in win:
            raise ValueError("Marvin tactile alignment requires dataset timestamps")
        result = self._touch.window(
            int(row["episode_index"]),
            offset,
            valid_start,
            valid_end,
            float(win["timestamp"].iloc[0]),
            **self._touch_window,
        )
        result["window_index"] = idx
        return result

    NEEDED_COLS = (*SharpaDexTianjiDataset.NEEDED_COLS, "timestamp")
