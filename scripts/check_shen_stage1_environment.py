#!/usr/bin/env python3
"""Verify dependencies, GPU arithmetic and packet access guards without training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import scipy
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shen_stage1.data import ShenDevelopmentPackets, development_path


def check_adapter():
    with tempfile.TemporaryDirectory(prefix="shen-stage1-synthetic-") as tmp:
        root = Path(tmp)
        data_root = root / "data"
        data_path = data_root / "Train/n210_1_day1_wireless_train.h5"
        data_path.parent.mkdir(parents=True)
        # Synthetic test fixture only; these are not research observations.
        rows = np.zeros((2, 16384), dtype=np.float64)
        rows[0, :8192] = 2
        rows[0, 8192:] = 3
        rows[1, :8192] = 4
        rows[1, 8192:] = 5
        with h5py.File(data_path, "w") as f:
            f["data"] = rows
            f["label"] = np.array([[31, 40]], dtype=np.float64)
        protocol_path = root / "protocol.json"
        protocol = {"data_root": "data", "label_map": {str(x): x - 31 for x in range(31, 41)},
                    "files": [{"receiver": "n210_1", "role": "train", "dataset_relative_path":
                               "Train/n210_1_day1_wireless_train.h5", "bytes": data_path.stat().st_size}]}
        protocol_path.write_text(json.dumps(protocol))
        dataset = ShenDevelopmentPackets(root, protocol_path, "n210_1", "train")
        iq, label, index = dataset[0]
        assert iq.shape == (2, 8192) and iq.dtype == np.float32
        assert np.all(iq[0] == 2) and np.all(iq[1] == 3)
        assert label == 0 and index == 0 and dataset[1][1] == 9
        # Real HDF5 handles are never shared between workers.
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)
        batch = next(iter(loader))
        assert batch[0].shape == (2, 2, 8192)
        assert batch[1].tolist() == [0, 9]
        denied = 0
        with patch("h5py.File", side_effect=AssertionError("Must reject before opening a file")):
            for relative in ("Test/n210_1_day3_wireless_test.h5", "Test/rtl_6_day4_wireless_test.h5",
                             "n210_1_test.h5", "../Train/n210_1_day1_wireless_train.h5"):
                try:
                    development_path(data_root, relative)
                except PermissionError:
                    denied += 1
                else:
                    raise AssertionError(f"Protected path unexpectedly allowed: {relative}")
            try:
                ShenDevelopmentPackets(root, protocol_path, "n210_1", "final_test_locked")
            except PermissionError:
                denied += 1
            else:
                raise AssertionError("Final-test role was accepted")
            protocol["files"][0]["dataset_relative_path"] = "Test/n210_1_day3_wireless_test.h5"
            protocol_path.write_text(json.dumps(protocol))
            try:
                ShenDevelopmentPackets(root, protocol_path, "n210_1", "train")
            except PermissionError:
                denied += 1
            else:
                raise AssertionError("Relabeling a final-test path as training bypassed the guard")
        return {"synthetic_iq_layout": "passed", "label_mapping": "passed",
                "two_worker_loader": "passed", "protected_access_cases_denied": denied,
                "real_research_signal_packets_read": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a new empty directory")
    out.mkdir(parents=True, exist_ok=True)
    assert sys.version_info >= (3, 10), "This project requires Python >=3.10"
    result = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
              "system_ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
              "cpu_count": os.cpu_count(), "workspace_disk_free_bytes": shutil.disk_usage(ROOT).free,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("numpy", "h5py", "scipy", "scikit-learn", "torch", "tqdm")},
              "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
              "cuda_available": torch.cuda.is_available(),
              "cuda_architectures": torch.cuda.get_arch_list(), "gpus": [],
              "adapter_checks": check_adapter(), "model_training_started": False}
    protocol_path = ROOT / "inventory/shen_stage1_audit_20261011/protocol.candidate.json"
    if protocol_path.exists():
        result["real_data_headers_only_checks"] = []
        result["candidate_protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        for receiver in ("n210_1", "rtl_6"):
            for role, expected_n in (("train", 8000), ("validation", 2000)):
                dataset = ShenDevelopmentPackets(ROOT, protocol_path, receiver, role)
                assert len(dataset) == expected_n
                assert set(dataset.labels.tolist()) == set(range(10))
                result["real_data_headers_only_checks"].append({"receiver": receiver, "role": role,
                    "packets": len(dataset), "labels": "0..9", "signal_rows_read": 0})
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-enabled PyTorch environment is not ready")
    for gpu in range(torch.cuda.device_count()):
        a = torch.ones((64, 64), device=f"cuda:{gpu}")
        b = a @ a
        torch.cuda.synchronize(gpu)
        assert torch.all(b == 64).item()
        # Exercise cuDNN as well as cuBLAS. Synthetic kernel checks only:
        # no optimizer, parameter updates, real signals, or saved model.
        convolution = torch.nn.Conv2d(1, 8, kernel_size=3).to(f"cuda:{gpu}")
        synthetic = torch.ones((2, 1, 52, 126), device=f"cuda:{gpu}")
        output = convolution(synthetic)
        output.sum().backward()
        torch.cuda.synchronize(gpu)
        assert torch.isfinite(output).all().item()
        assert torch.isfinite(convolution.weight.grad).all().item()
        result["gpus"].append({"index": gpu, "name": torch.cuda.get_device_name(gpu),
                               "capability": list(torch.cuda.get_device_capability(gpu)),
                               "float32_matrix_check": "passed", "synthetic_convolution_forward_backward": "passed"})
        del a, b, synthetic, output, convolution
    torch.cuda.empty_cache()
    dependencies = subprocess.check_output([sys.executable, "-c",
        "import importlib.metadata as m; print('\\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))"], text=True)
    (out / "installed-packages.txt").write_text(dependencies)
    result["installed_packages_sha256"] = hashlib.sha256(dependencies.encode()).hexdigest()
    (out / "environment.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
