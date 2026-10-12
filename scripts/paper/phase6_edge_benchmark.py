#!/usr/bin/env python3
"""Phase 6: Edge deployment metrics — params, FLOPs, latency, memory, voting curve."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rfhstu.cnn_baseline import OSUCNNBaseline
from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from rfhstu.train_utils import load_checkpoint, resolve_device


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(model, device, batch_sizes=(1, 32, 64), steps=50, warmup=10):
    model.eval()
    results = {}
    x = torch.randn(1, 2, 8192, device=device)
    for bs in batch_sizes:
        inp = x.expand(bs, -1, -1)
        with torch.no_grad():
            for _ in range(warmup):
                model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                model(inp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / steps * 1000
        results[bs] = ms
    return results


def peak_memory_mb(model, device, batch_size=32):
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(batch_size, 2, 8192, device=device)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


def build_hybrid(num_classes=24, dim=64, depth=2):
    embedder = RFPatchEmbedder(
        window_size=8192, patch_size=256, patch_embed_type="cnn_stem", cnn_stem_dim=32,
        oob_fusion_type="cross_attn_oob", use_oob_cross_attention=True,
        fft_norm="log_zscore", oob_norm="ratio",
    )
    return DeviceClassifier(
        embedder, num_classes=num_classes, dim=dim, depth=depth,
        use_chirp_embedding=True, oob_num_heads=4,
    )


def voting_curve(checkpoint: Path, manifest: Path, spf_list, py: str, out_dir: Path):
    rows = []
    for spf in spf_list:
        od = out_dir / f"spf_{spf}"
        subprocess.run(
            [
                py, str(ROOT / "scripts/evaluate.py"),
                "--manifest", str(manifest),
                "--checkpoint", str(checkpoint),
                "--eval-samples-per-file", str(spf),
                "--samples-per-file", str(spf),
                "--eval-split", "test", "--train-split", "train", "--val-split", "val",
                "--mode", "classifier", "--file-vote-mode", "mean_logits",
                "--out-dir", str(od), "--device", "cuda",
            ],
            cwd=ROOT,
            check=False,
        )
        metrics_path = od / "metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            rows.append({"eval_samples_per_file": spf, **m})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default="outputs/paper_ready/edge_deployment_summary.csv")
    parser.add_argument("--hybrid-ckpt", default=None)
    parser.add_argument("--cnn-ckpt", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []

    for name, builder in [
        ("CNN-IQ", lambda: OSUCNNBaseline(num_classes=24)),
        ("Hybrid", build_hybrid),
    ]:
        model = builder().to(device)
        nparams = count_params(model)
        lat = benchmark_latency(model, device)
        mem = peak_memory_mb(model, device)
        summary.append({
            "model": name,
            "params": nparams,
            "latency_ms_bs1": lat.get(1, ""),
            "latency_ms_bs32": lat.get(32, ""),
            "latency_ms_bs64": lat.get(64, ""),
            "peak_gpu_mem_mb_bs32": mem,
        })

    # Voting curves if checkpoints provided
    manifest = root / "data/paper/cross_day_day1to5_source_only.csv"
    py = sys.executable
    for label, ckpt_arg in [("hybrid", args.hybrid_ckpt), ("cnn", args.cnn_ckpt)]:
        if ckpt_arg and Path(ckpt_arg).exists():
            vc = voting_curve(
                Path(ckpt_arg), manifest, [32, 64, 128, 256, 512], py,
                out.parent / "figures" / f"voting_{label}",
            )
            vc_out = out.parent / f"voting_curve_{label}.csv"
            if vc:
                with vc_out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=vc[0].keys())
                    w.writeheader()
                    w.writerows(vc)

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
