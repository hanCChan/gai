from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import complex_iq_to_channels, normalize_iq


DOMAIN_FIELDS = ("day", "receiver", "location", "distance", "sf", "config")


@dataclass(frozen=True)
class ManifestRow:
    path: Path
    device: int
    label: int
    split: str
    setup: str
    domains: dict[str, int]


def _to_int(value: str | int | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def load_manifest(
    manifest_path: str | Path,
    root: str | Path | None = None,
    split: str | None = None,
    setup: str | None = None,
    fold: str | None = None,
    max_files: int | None = None,
) -> list[ManifestRow]:
    manifest_path = Path(manifest_path)
    root_path = Path(root) if root is not None else manifest_path.parents[3] if len(manifest_path.parents) >= 4 else Path.cwd()
    fold_key = str(fold) if fold is not None else None
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            if split is not None and item.get("split") != split:
                continue
            if fold_key is not None and str(item.get("fold", "")) != fold_key:
                continue
            if setup is not None and item.get("setup") != setup:
                continue
            data_path = Path(item["path"])
            if not data_path.is_absolute():
                data_path = root_path / data_path
            if not data_path.exists():
                continue
            domains = {field: _to_int(item.get(field), 0) for field in DOMAIN_FIELDS}
            rows.append(
                ManifestRow(
                    path=data_path,
                    device=_to_int(item.get("device")),
                    label=_to_int(item.get("label")),
                    split=item.get("split", ""),
                    setup=item.get("setup", ""),
                    domains=domains,
                )
            )
            if max_files is not None and len(rows) >= max_files:
                break
    return rows


def remap_labels(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    rows = list(rows)
    devices = {device: idx for idx, device in enumerate(sorted({row.device for row in rows}))}
    remapped = []
    for row in rows:
        remapped.append(
            ManifestRow(
                path=row.path,
                device=row.device,
                label=devices[row.device],
                split=row.split,
                setup=row.setup,
                domains=row.domains,
            )
        )
    return remapped


def infer_domain_sizes(rows: Iterable[ManifestRow]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for field in DOMAIN_FIELDS:
        values = [row.domains[field] for row in rows]
        sizes[field] = max(values) + 1 if values else 1
    return sizes


class SigMFIQDataset(Dataset):
    """Random or deterministic windows from SigMF `cf32` files."""

    def __init__(
        self,
        rows: list[ManifestRow],
        window_size: int = 8192,
        samples_per_file: int = 128,
        random_windows: bool = True,
        seed: int = 1234,
        input_norm: str = "iq_rms",
    ) -> None:
        if not rows:
            raise ValueError("No usable manifest rows found.")
        if input_norm not in ("none", "iq_rms"):
            raise ValueError(f"Unknown input_norm={input_norm!r}; expected 'none' or 'iq_rms'")
        self.rows = rows
        self.window_size = window_size
        self.samples_per_file = samples_per_file
        self.random_windows = random_windows
        self.seed = seed
        self.input_norm = input_norm
        self.epoch = 0
        self._memmaps: dict[Path, np.memmap] = {}
        self._lengths = {row.path: row.path.stat().st_size // np.dtype(np.complex64).itemsize for row in rows}

    def __len__(self) -> int:
        return len(self.rows) * self.samples_per_file

    @property
    def num_classes(self) -> int:
        return max(row.label for row in self.rows) + 1

    @property
    def domain_sizes(self) -> dict[str, int]:
        return infer_domain_sizes(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _open(self, path: Path) -> np.memmap:
        mm = self._memmaps.get(path)
        if mm is None:
            mm = np.memmap(path, dtype=np.complex64, mode="r")
            self._memmaps[path] = mm
        return mm

    def _offset(self, row_index: int, sample_index: int, length: int) -> int:
        max_offset = max(0, length - self.window_size)
        if max_offset == 0:
            return 0
        if self.random_windows:
            rng = np.random.default_rng(self.seed + self.epoch * 10_000_019 + row_index * 1_000_003 + sample_index)
            return int(rng.integers(0, max_offset + 1))
        stride = max(1, max_offset // max(1, self.samples_per_file - 1))
        return min(sample_index * stride, max_offset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row_index = index // self.samples_per_file
        sample_index = index % self.samples_per_file
        row = self.rows[row_index]
        length = self._lengths[row.path]
        if length < self.window_size:
            raise ValueError(f"{row.path} has {length} samples, shorter than window_size {self.window_size}")
        offset = self._offset(row_index, sample_index, length)
        iq = np.asarray(self._open(row.path)[offset : offset + self.window_size])
        channels = complex_iq_to_channels(iq)
        if self.input_norm == "iq_rms":
            channels = normalize_iq(channels)
        else:  # "none": raw IQ, no per-window RMS normalization
            channels = channels.astype(np.float32, copy=False)
        domains = torch.tensor([row.domains[field] for field in DOMAIN_FIELDS], dtype=torch.long)
        return {
            "iq": torch.from_numpy(channels.copy()),
            "label": torch.tensor(row.label, dtype=torch.long),
            "device": torch.tensor(row.device, dtype=torch.long),
            "domains": domains,
            "file_path": str(row.path),
            "window_index": torch.tensor(sample_index, dtype=torch.long),
            "sample_offset": torch.tensor(offset, dtype=torch.long),
            "split": row.split,
            "setup": row.setup,
        }
