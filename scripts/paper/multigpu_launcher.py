#!/usr/bin/env python3
""" across multiple GPUs with a fixed-size worker pool.

Each job is one shell command (train+eval wrapped by caller). Jobs whose done-marker
file already exists are skipped.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Job:
    name: str
    cmd: str
    done_marker: Optional[str] = None


def parse_jobs(path: Path) -> List[Job]:
    jobs: List[Job] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            raise ValueError(f"bad job line (need name\\tcmd): {line!r}")
        name, cmd = parts[0], parts[1]
        marker = parts[2] if len(parts) > 2 else None
        jobs.append(Job(name=name, cmd=cmd, done_marker=marker))
    return jobs


def run_one(gpu: str, job: Job, log_dir: Path, dry_run: bool) -> tuple[str, int, float]:
    log_path = log_dir / f"{job.name}.log"
    t0 = time.time()

    if job.done_marker and Path(job.done_marker).is_file():
        msg = f"[skip] {job.name} (marker exists)"
        log_path.write_text(msg + "\n", encoding="utf-8")
        print(msg, flush=True)
        return job.name, 0, time.time() - t0

    header = f"[GPU {gpu}] START {job.name}\n$ {job.cmd}\n"
    print(header, end="", flush=True)
    if dry_run:
        return job.name, 0, 0.0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(header)
        lf.flush()
        proc = subprocess.run(
            job.cmd,
            shell=True,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=os.environ.get("ROOT", None),
        )
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"[GPU {gpu}] {status} {job.name} ({elapsed:.0f}s)", flush=True)
    return job.name, proc.returncode, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Run paper experiment jobs on multiple GPUs")
    ap.add_argument("--jobs-file", required=True, help="TSV: name\\tcmd\\t[done_marker]")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Comma-separated GPU ids")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--max-workers", type=int, default=0, help="Defaults to len(gpus)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        print("No GPUs specified", file=sys.stderr)
        return 2

    jobs = parse_jobs(Path(args.jobs_file))
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    max_workers = args.max_workers or len(gpus)
    max_workers = min(max_workers, len(gpus), len(jobs)) if jobs else 0
    print(f"Launcher: {len(jobs)} jobs, GPUs={gpus}, workers={max_workers}", flush=True)

    if not jobs:
        return 0

    import queue

    gpu_q: queue.Queue[str] = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    def worker(job: Job) -> tuple[str, int, float]:
        gpu = gpu_q.get()
        try:
            return run_one(gpu, job, log_dir, args.dry_run)
        finally:
            gpu_q.put(gpu)

    failed: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(worker, job): job.name for job in jobs}
        for fut in as_completed(futs):
            name, rc, _ = fut.result()
            if rc != 0:
                failed.append(name)

    if failed:
        print(f"FAILED jobs ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        return 1
    print("All jobs finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
