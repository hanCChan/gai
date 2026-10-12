#!/usr/bin/env python3
"""Deprecated — use generate_paper_v3_jobs.py instead."""

import sys

if __name__ == "__main__":
    print(
        "ERROR: generate_v3_jobs.py is deprecated.\n"
        "Use: python scripts/paper/generate_paper_v3_jobs.py --step step1 --dry-run",
        file=sys.stderr,
    )
    raise SystemExit(1)
