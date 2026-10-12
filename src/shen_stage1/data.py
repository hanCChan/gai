"""Read-only packet access governed by the audited stage-one protocol.

No final-test override is provided. A future evaluation entry point must be
designed after model selection is frozen. Raw HDF5 files are never rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset
from .protocol import DEVELOPMENT_FILES, ROLE_PATHS, development_path


class ShenDevelopmentPackets(Dataset):
    """Lazy HDF5 access; returns IQ [2, 8192], mapped label, and row index.

    This adapter applies no RMS normalization, CFO compensation, augmentation,
    or resampling. Those must be explicit transformations in a future runner.
    Opening each requested row avoids sharing HDF5 handles across processes.
    """

    def __init__(self, project_root: str | Path, protocol_path: str | Path,
                 receiver: str, role: str):
        if role not in ROLE_PATHS:
            raise PermissionError(f"Development adapter cannot access role {role}")
        project = Path(project_root).resolve()
        protocol = json.loads(Path(protocol_path).read_text(encoding="utf8"))
        matching = [x for x in protocol["files"]
                    if x["receiver"] == receiver and x["role"] == role]
        if len(matching) != 1:
            raise ValueError("Protocol must contain exactly one file for this receiver/role")
        item = matching[0]
        relative = item["dataset_relative_path"]
        if relative != ROLE_PATHS[role].format(receiver=receiver):
            raise PermissionError("Protocol path disagrees with the fixed development role")
        self.path = development_path(project / protocol["data_root"], relative)
        if self.path.stat().st_size != item["bytes"]:
            raise ValueError("File size changed since the audit; rerun the audit")
        label_map = {int(k): int(v) for k, v in protocol["label_map"].items()}
        with h5py.File(self.path, "r") as f:
            # Never dereference an external or soft link supplied by a file.
            for name in ("label", "data"):
                if not isinstance(f.get(name, getlink=True), h5py.HardLink):
                    raise ValueError(f"Unsupported HDF5 link: {name}")
            labels = np.asarray(f["label"][:]).reshape(-1)
            if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
                raise ValueError("Labels must be finite integers")
            self.labels = np.asarray([label_map[int(x)] for x in labels], dtype=np.int64)
            if f["data"].shape != (len(labels), 16384):
                raise ValueError("Expected packet data with shape [N, 16384]")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with h5py.File(self.path, "r") as f:
            row = np.asarray(f["data"][index], dtype=np.float32)
        if not np.isfinite(row).all():
            raise ValueError(f"Nonfinite signal at packet {index}")
        return row.reshape(2, 8192), int(self.labels[index]), int(index)
