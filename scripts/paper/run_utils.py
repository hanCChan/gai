#!/usr/bin/env python3
"""Utilities for timestamped paper experiment runs with git provenance."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit(root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        )
    except Exception:
        return "unknown"


def run_tag(prefix: str = "paper") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def make_run_dirs(root: Path, phase: str, tag: str | None = None) -> dict[str, Path]:
    tag = tag or run_tag(phase)
    commit = git_commit(root)
    base = root / "outputs" / "paper_runs" / f"{phase}_{tag}_{commit[:8]}"
    runs = base / "runs"
    outputs = base / "outputs"
    logs = base / "logs"
    for d in (runs, outputs, logs):
        d.mkdir(parents=True, exist_ok=True)
    meta = {
        "phase": phase,
        "tag": tag,
        "git_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs_dir": str(runs),
        "outputs_dir": str(outputs),
        "logs_dir": str(logs),
    }
    (base / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"base": base, "runs": runs, "outputs": outputs, "logs": logs, "meta": meta}


def write_run_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["git_commit"] = git_commit(path.parents[3] if len(path.parents) > 3 else Path("."))
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
