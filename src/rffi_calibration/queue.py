"""SQLite job queue with transactional claims and a reservation guard (no retries by design).

States: pending -> running -> completed | failed | cancelled ; blocked (never claimable).
A job is claimable only if it is pending, listed in the ACTIVE reservation's allowed_job_ids, and its
predicted GPU seconds fit in the remaining reserved budget. `complete` requires exit code 0, all
artifacts present and an independent verifier pass; otherwise the job is marked failed. Failed jobs are
never re-queued under the same job_id.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

COLUMNS = ['job_id', 'stage', 'method_id', 'dataset_id', 'source_split', 'seed', 'config_sha', 'code_commit',
           'data_manifest_sha', 'support_manifest_sha', 'status', 'gpu_index', 'start_time', 'end_time',
           'gpu_reserved_seconds', 'predicted_gpu_seconds', 'wall_seconds', 'peak_memory_mb', 'exit_code',
           'failure_reason', 'artifact_path']
STATES = ('pending', 'running', 'completed', 'failed', 'cancelled', 'blocked')


class BudgetExceeded(RuntimeError):
    pass


class JobQueue:
    def __init__(self, path):
        self.path = str(path)
        with self._conn() as c:
            c.execute(f'CREATE TABLE IF NOT EXISTS jobs ({", ".join(k + (" PRIMARY KEY" if k == "job_id" else "") for k in COLUMNS)})')

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        return c

    def add(self, **job):
        job = dict(job, status=job.get('status', 'pending'))
        if job['status'] not in ('pending', 'blocked'):
            raise ValueError('new jobs start pending or blocked')
        keys = [k for k in COLUMNS if k in job]
        with self._conn() as c:
            c.execute(f'INSERT INTO jobs ({",".join(keys)}) VALUES ({",".join("?" * len(keys))})',
                      [job[k] for k in keys])

    def _used_seconds(self, c):
        r = c.execute("SELECT COALESCE(SUM(COALESCE(gpu_reserved_seconds, predicted_gpu_seconds, 0)), 0) "
                      "FROM jobs WHERE status IN ('running','completed','failed','cancelled')").fetchone()
        return float(r[0])

    def claim(self, gpu_index, reservation):
        """Atomically claim one allowed pending job, or return None. Raises BudgetExceeded if the next
        allowed job would exceed the reserved GPU hours."""
        if not reservation.get('active'):
            return None
        allowed = list(reservation.get('allowed_job_ids', []))
        if not allowed:
            return None
        cap = float(reservation.get('max_reserved_gpu_hours', 0)) * 3600.
        c = self._conn()
        try:
            c.execute('BEGIN IMMEDIATE')
            q = f"SELECT * FROM jobs WHERE status='pending' AND job_id IN ({','.join('?' * len(allowed))}) ORDER BY job_id LIMIT 1"
            row = c.execute(q, allowed).fetchone()
            if row is None:
                c.execute('COMMIT')
                return None
            need = float(row['predicted_gpu_seconds'] or 0)
            if self._used_seconds(c) + need > cap:
                c.execute('COMMIT')
                raise BudgetExceeded(f'job {row["job_id"]} needs {need:.0f}s; reserved cap {cap:.0f}s')
            n = c.execute("UPDATE jobs SET status='running', gpu_index=?, start_time=? WHERE job_id=? AND status='pending'",
                          (gpu_index, time.time(), row['job_id'])).rowcount
            c.execute('COMMIT')
            return dict(row, status='running', gpu_index=gpu_index) if n == 1 else None
        except Exception:
            if c.in_transaction:
                c.execute('ROLLBACK')
            raise
        finally:
            c.close()

    def _finish(self, job_id, status, **kw):
        sets = ', '.join(f'{k}=?' for k in kw)
        with self._conn() as c:
            n = c.execute(f"UPDATE jobs SET status=?, end_time=?{', ' + sets if sets else ''} "
                          f"WHERE job_id=? AND status='running'", [status, time.time(), *kw.values(), job_id]).rowcount
        if n != 1:
            raise ValueError(f'job {job_id} is not running')

    def complete(self, job_id, exit_code, artifacts_ok, verifier_ok, **kw):
        if exit_code == 0 and artifacts_ok and verifier_ok:
            self._finish(job_id, 'completed', exit_code=exit_code, **kw)
            return 'completed'
        reason = 'exit!=0' if exit_code != 0 else 'missing artifacts' if not artifacts_ok else 'verifier failed'
        self._finish(job_id, 'failed', exit_code=exit_code, failure_reason=reason, **kw)
        return 'failed'

    def fail(self, job_id, reason, **kw):
        self._finish(job_id, 'failed', failure_reason=reason, **kw)

    def cancel(self, job_id, reason='SIGTERM', **kw):
        """Interrupted job: cancelled, never completed."""
        self._finish(job_id, 'cancelled', failure_reason=reason, **kw)

    def get(self, job_id):
        with self._conn() as c:
            r = c.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        return dict(r) if r else None

    def export_jsonl(self, path):
        with self._conn() as c:
            rows = [dict(r) for r in c.execute('SELECT * FROM jobs ORDER BY job_id')]
        Path(path).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return len(rows)
