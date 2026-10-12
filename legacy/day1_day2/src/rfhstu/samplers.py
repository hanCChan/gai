from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler


class DeviceBalancedBatchSampler(Sampler[list[int]]):
    """Sample batches with P devices and K windows per device."""

    def __init__(
        self,
        dataset,
        devices_per_batch: int = 8,
        samples_per_device: int = 2,
        seed: int = 1234,
        drop_last: bool = False,
    ) -> None:
        if devices_per_batch <= 0 or samples_per_device <= 0:
            raise ValueError("devices_per_batch and samples_per_device must be positive.")
        self.dataset = dataset
        self.devices_per_batch = devices_per_batch
        self.samples_per_device = samples_per_device
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            row_index = index // dataset.samples_per_file
            label = int(dataset.rows[row_index].label)
            self.by_label[label].append(index)
        self.labels = sorted(self.by_label)
        if not self.labels:
            raise ValueError("DeviceBalancedBatchSampler requires at least one label.")

    @property
    def batch_size(self) -> int:
        return self.devices_per_batch * self.samples_per_device

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        batches = len(self.dataset) / self.batch_size
        return math.floor(batches) if self.drop_last else math.ceil(batches)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        for _ in range(len(self)):
            if len(self.labels) >= self.devices_per_batch:
                labels = rng.sample(self.labels, self.devices_per_batch)
            else:
                labels = [rng.choice(self.labels) for _ in range(self.devices_per_batch)]
            batch: list[int] = []
            for label in labels:
                candidates = self.by_label[label]
                if len(candidates) >= self.samples_per_device:
                    batch.extend(rng.sample(candidates, self.samples_per_device))
                else:
                    batch.extend(rng.choice(candidates) for _ in range(self.samples_per_device))
            rng.shuffle(batch)
            yield batch
