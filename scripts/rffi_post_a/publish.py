#!/usr/bin/env python3
"""Audit and fast-forward this completed experiment, then verify the remote Git tree/readbacks."""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = 'experiment/rffi-post-stage-a-diagnostics-v1'
REPO = 'hanCChan/gai'
URL = f'https://github.com/{REPO}.git'
PROXY = 'http://star-proxy.oa.com:3128'
TOKEN_FILE = Path('/workspace/user_code/kuozhan/token')
BASE = 'dc805ff2a906e0951e0bfcf371f1b004433567db'
ASKPASS = Path('/tmp/gai-post-a-askpass.py')
EXCLUDED_SUFFIXES = {
    '.npz', '.npy', '.pt', '.pth', '.ckpt', '.h5', '.hdf5', '.dat', '.pkl', '.pickle',
    '.zip', '.7z', '.tar', '.gz', '.sqlite', '.db', '.env', '.pem', '.key', '.p12',
    '.sigmf-data', '.sigmf-meta',
}
SECRET_PATTERNS = [
    re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{40,}\b'),
    re.compile(rb'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----'),
    re.compile(rb'https?://[^/\s:@]+:[^/\s@]+@'),
]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def tree(commit):
    out = {}
    for entry in git('ls-tree', '-rz', commit).split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, typ, sha = meta.decode().split()
        out[name.decode()] = dict(mode=mode, type=typ, sha=sha)
    return out


def audit(token):
    if git('branch', '--show-current').decode().strip() != BRANCH:
        raise ValueError('wrong local branch')
    if git('status', '--porcelain').strip():
        raise ValueError('commit the publication first; do not push a dirty worktree')
    head = git('rev-parse', 'HEAD').decode().strip()
    current = tree(head)
    checked, all_paths = set(), set()
    # Inspect every outgoing historical blob, not only final HEAD.
    commits = git('rev-list', '--reverse', f'{BASE}..{head}').decode().splitlines()
    for commit in commits:
        paths = git('diff-tree', '--no-commit-id', '--name-only', '-rz', commit).split(b'\0')
        for raw_path in paths:
            if not raw_path:
                continue
            path = raw_path.decode()
            all_paths.add(path)
            parts = Path(path).parts
            if path not in ('.gitignore', '.gitattributes', 'AGENTS.md') and not (
                    path.startswith(('src/rffi_post_a/', 'scripts/rffi_post_a/', 'tests/rffi_post_a/',
                                     'configs/rffi_post_a/', 'docs/rffi_post_a/', 'reports/rffi_post_a/'))):
                raise ValueError(f'path outside publication scope: {path}')
            if Path(path).suffix.lower() in EXCLUDED_SUFFIXES or any(
                    p.lower() in ('data', 'outputs', 'credentials', 'token', '.env') for p in parts):
                raise ValueError(f'excluded outgoing path: {path}')
            sha = git('rev-parse', f'{commit}:{path}').decode().strip()
            if sha in checked:
                continue
            blob = git('cat-file', 'blob', sha)
            if token in blob or any(pattern.search(blob) for pattern in SECRET_PATTERNS):
                raise ValueError(f'credential scan rejected blob at {path}')
            if len(blob) > 10 << 20:
                raise ValueError(f'unreviewed large blob: {path}')
            checked.add(sha)
    for rel in ('configs/rffi_post_a/registration.json',
                'configs/rffi_post_a/variance_v1/registration.json',
                'configs/rffi_calibration/registration.json'):
        if json.loads(git('show', f'{head}:{rel}'))['reservation']['active']:
            raise ValueError(f'active numerical reservation: {rel}')
    prereg = json.loads((ROOT / 'configs/rffi_post_a/variance_v1/preregistration.json').read_text())
    for rel, sha in prereg['immutable_inputs'].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != sha:
            raise ValueError(f'RV1 frozen input changed: {rel}')
        if hashlib.sha256(git('show', f'{head}:{rel}')).hexdigest() != sha:
            raise ValueError(f'committed RV1 frozen input differs: {rel}')
    for stage in ('R0', 'RV1'):
        evidence = json.loads((ROOT / f'reports/rffi_post_a/{stage}/evidence_index.json').read_text())
        for entry in evidence['files']:
            rel = entry['path']
            if not entry.get('local_only') and rel in current:
                if hashlib.sha256(git('cat-file', 'blob', current[rel]['sha'])).hexdigest() != entry['sha256']:
                    raise ValueError(f'committed evidence differs: {rel}')
    r0 = json.loads(git('show', f'{head}:reports/rffi_post_a/R0/decision.json'))
    rv1 = json.loads(git('show', f'{head}:reports/rffi_post_a/RV1/decision.json'))
    if r0['decision'] != 'NO_GAIN_IN_FIXED_PILOT' or rv1['decision'] != 'NO_STABLE_TWO_ARM_GAIN':
        raise ValueError('historical decision changed')
    return head, current, dict(outgoing_commits=len(commits), changed_paths=len(all_paths),
                               historical_blobs_scanned=len(checked), credential_scan='passed',
                               excluded_artifact_scan='passed', frozen_inputs_checked=len(prereg['immutable_inputs']))


def api(token, path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': PROXY, 'http': PROXY}))
    request = urllib.request.Request(
        'https://api.github.com' + path,
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'gai-post-a-publication'})
    try:
        with opener.open(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'GitHub HTTP {exc.code} for {path}') from None


def prepare_r0_index():
    report = ROOT / 'reports/rffi_post_a/R0'
    target = report / 'evidence_index.json'
    if target.exists():
        raise ValueError('R0 evidence index already exists; do not overwrite')
    files = []
    for path in sorted(report.rglob('*')):
        if not path.is_file() or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        content = path.read_bytes()
        files.append(dict(path=str(path.relative_to(ROOT)), bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest()))
    checkpoints = {}
    for path in sorted((ROOT / 'outputs/rffi_post_a').glob('R0_*/status.json')):
        status = json.loads(path.read_text())
        checkpoints[status['job']['job_id']] = dict(
            last=status['last_sha256'], native_best=status['native_best_sha256'],
            init=status['init_state_sha256'], all_ids=status['all_ids_chain_sha256'])
    if len(checkpoints) != 6:
        raise ValueError('R0 checkpoint provenance incomplete')
    target.write_text(json.dumps(dict(
        stage='R0', indexed_date=datetime.now().astimezone().date().isoformat(),
        decision='NO_GAIN_IN_FIXED_PILOT', files=files, checkpoint_sha256=checkpoints,
        scope='existing summaries and hashes only; no new numerical evaluation',
        excluded=['credentials', 'scores.npz', 'weights', 'features', 'queue.sqlite']), indent=2) + '\n')
    print(json.dumps(dict(index=str(target.relative_to(ROOT)), curated_files=len(files), checkpoint_provenance=6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', action='store_true')
    ap.add_argument('--record', type=Path)
    ap.add_argument('--prepare-r0-index', action='store_true')
    args = ap.parse_args()
    if args.prepare_r0_index:
        prepare_r0_index()
        return
    token = TOKEN_FILE.read_text().strip()
    if len(token) < 20:
        raise SystemExit('token file is invalid')
    head, local_tree, checks = audit(token.encode())
    user = api(token, '/user')
    repo = api(token, f'/repos/{REPO}')
    if user['login'] != 'hanCChan' or not repo.get('permissions', {}).get('push'):
        raise ValueError('wrong account or missing repository push permission')
    remote_before = api(token, f'/repos/{REPO}/git/ref/heads/{BRANCH}')['object']['sha']
    subprocess.run(['git', 'merge-base', '--is-ancestor', remote_before, head], cwd=ROOT, check=True)
    print(json.dumps(dict(local_head=head, observed_remote_head=remote_before, checks=checks)), flush=True)
    if args.push:
        if not ASKPASS.exists():
            raise ValueError('credential callback missing')
        env = dict(os.environ, GIT_ASKPASS=str(ASKPASS), GIT_TERMINAL_PROMPT='0')
        for key in list(env):
            if key.startswith(('GIT_TRACE', 'GIT_CURL_VERBOSE')):
                del env[key]
        subprocess.run(
            ['git', '-c', f'http.proxy={PROXY}', '-c', 'credential.helper=',
             '-c', f'core.askPass={ASKPASS}', '-c', 'push.followTags=false',
             'push', URL, f'HEAD:refs/heads/{BRANCH}'],
            cwd=ROOT, env=env, check=True, timeout=300)
    remote = api(token, f'/repos/{REPO}/git/ref/heads/{BRANCH}')['object']['sha']
    if remote != head:
        raise ValueError('remote SHA differs from local HEAD; publication not verified')
    remote_commit = api(token, f'/repos/{REPO}/git/commits/{head}')
    remote_tree = api(token, f'/repos/{REPO}/git/trees/{remote_commit["tree"]["sha"]}?recursive=1')
    if remote_tree.get('truncated'):
        raise ValueError('remote tree truncated; verification incomplete')
    observed_tree = {r['path']: {k: r[k] for k in ('mode', 'type', 'sha')}
                     for r in remote_tree['tree'] if r['type'] != 'tree'}
    if observed_tree != local_tree:
        raise ValueError('complete local/remote Git tree differs')
    readbacks = {}
    decisions = {}
    for stage in ('D0', 'R0', 'RV1'):
        for name in ('REPORT.md', 'decision.json'):
            rel = f'reports/rffi_post_a/{stage}/{name}'
            sha = local_tree[rel]['sha']
            remote_blob = api(token, f'/repos/{REPO}/git/blobs/{sha}')
            content = base64.b64decode(remote_blob['content'])
            expected = git('cat-file', 'blob', sha)
            if content != expected:
                raise ValueError(f'remote readback differs: {rel}')
            readbacks[rel] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                                  git_blob_sha=sha)
            if name == 'decision.json':
                decisions[stage] = json.loads(content).get('decision')
    for rel in ('configs/rffi_post_a/registration.json', 'configs/rffi_post_a/variance_v1/registration.json'):
        blob = api(token, f'/repos/{REPO}/git/blobs/{local_tree[rel]["sha"]}')
        if json.loads(base64.b64decode(blob['content']))['reservation']['active']:
            raise ValueError('remote reservation unexpectedly active')
    record = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(), repository=REPO,
                  branch=BRANCH, commit=head, observed_remote_before=remote_before,
                  complete_tree_blobs_verified=len(local_tree),
                  checks=checks, readbacks=readbacks, decisions=decisions, numerical_reservations_active=False,
                  scope='complete committed tree and reports/decisions at this publication commit; record committed subsequently')
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        if args.record.exists():
            raise ValueError('publication record already exists; do not overwrite')
        args.record.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
