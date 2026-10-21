#!/usr/bin/env python3
"""P2D auto-chain runner: the "run one round, decide, generate the next round" driver.

Rounds are cheap (each Shen search fit ~20s on V100, formal similar; 4 GPUs, 8 workers).
The chain runs entirely offline on the sealed Shen protocol; nothing touches outer-RX
labels during training/selection (same guards as queue_p2.py / select_lambda_p2.py).

Round structure per candidate family F (registered in CANDIDATES below):
  1. search   : fold x {0,1} x lambda-grid x seed 20261016 jobs on 4 GPUs (train_p2.py --stage search)
  2. select   : per (fold, arm) lambda* = min held-RX CE (select_lambda_p2.py --tag p2d_<family>)
  3. formal   : fold x seed {20261013..15} at lambda* (train_p2.py --stage formal)
  4. eval     : eval_p2.py with --extra-formal the new outputs; anchors ce/ls/rxnull/rxlogit/ccalign reused
  5. decide   : decide_p2d.py -> decision_p2d.json
  6. continue : GO_CROSS_DATASET -> stop (WiSig needs the user's GPU-time go-ahead);
                ITERATE -> next candidate family in CANDIDATES (chain continues automatically);
                STOP_TAIL -> stop with the diagnosis write-up.

Everything is resumable: each stage skips completed runs (status.json present), so
re-invoking run_p2d_chain.py --families all continues where it stopped.

    python3 run_p2d_chain.py --out-root outputs --report-root reports \
        --sets ../receiver_p1/sets_p1.json [--families rxccp] [--workers 8] [--max-rounds 4] [--dry-run]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (20261013, 20261014, 20261015)
SEARCH_SEED = 20261016
LAMBDA_GRID = (0.3, 1.0, 3.0)

# Candidate families in iteration order. Each entry: family name -> arms (train_p2 methods).
# r1: rxccp (hard worst-source-shift CE) + rxgeo (geometry-only ablation) + rxccp_ls.
#     r1 post-mortem: beats ce (6/6, +1.2/+2.7pp) but not ls/ccalign; the hard inner max
#     collapses onto one receiver per class (b210 chosen for 7/10 classes on fold 0) = the
#     P1 fixed-hard-set failure mode. => r2 softens the max.
# r2: rxccp_soft (LSE relaxation, all RX in play) + rxccp_soft_ls.
#     r2 post-mortem: first P2D arm to beat BOTH ce and ls (5/6, 5/6) but still behind
#     ccalign (-1.24/-1.07pp); worst-RX/Pluto got WORSE (soft max spreads emphasis away
#     from the hard receiver). Confirms: shift-CE family caps at ls-level; the remaining
#     gap tracks margin TAIL risk (pflip/cantelli), which no shift-CE targets.
# r3: rxcant (train the Cantelli ratio of true margins pooled over source RX) + rxcant_ls.
CANDIDATES = {
    'r1': ['rxccp', 'rxgeo', 'rxccp_ls'],
    'r2': ['rxccp_soft', 'rxccp_soft_ls'],
    'r3': ['rxcant', 'rxcant_ls'],
}
CHAIN_STATE = Path(HERE.parent.parent / 'reports' / 'receiver_p2d' / 'chain_state.json')


def sh(cmd, **kw):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    print(r.stdout[-4000:] if r.returncode == 0 else r.stdout[-2000:] + '\nSTDERR:\n' + r.stderr[-4000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(f'stage failed: {cmd}')
    return r


def stage_search(family, arms, out_root, sets, workers):
    jobs_path = HERE / f'jobs_search_p2d_{family}.txt'
    lines = []
    for fold in (0, 1):
        for m in arms:
            for lam in LAMBDA_GRID:
                lines.append(f'--stage search --fold {fold} --method {m} --lam {lam:g} --seed {SEARCH_SEED}')
    jobs_path.write_text('\n'.join(lines) + '\n')
    sh([sys.executable, HERE / 'queue_p2.py', '--jobs', jobs_path, '--out', out_root / 'receiver_p2d' / f'search_{family}',
        '--sets', sets, '--workers', str(workers)])


def stage_select(family, arms, out_root, report_root, sets):
    out = report_root / 'receiver_p2d'
    out.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, HERE / 'select_lambda_p2.py', '--search', out_root / 'receiver_p2d' / f'search_{family}',
        '--p1search', out_root / 'receiver_p1' / 'search', '--out', out, '--methods', ','.join(arms),
        '--tag', f'p2d_{family}', '--rand', 0])
    return out / f'jobs_formal_p2d_{family}.txt'


def stage_formal(family, out_root, report_root, workers, sets):
    sh([sys.executable, HERE / 'queue_p2.py', '--jobs', report_root / 'receiver_p2d' / f'jobs_formal_p2d_{family}.txt',
        '--out', out_root / 'receiver_p2d' / f'formal_{family}',
        '--sets', sets, '--workers', str(workers)])


def stage_eval(family, out_root, report_root, sets):
    ev = report_root / 'receiver_p2d' / f'eval_{family}'
    # select_lambda_p2.py already wrote the exact formal job list: jobs_formal_p2d_<family>.txt
    jobs_file = report_root / 'receiver_p2d' / f'jobs_formal_p2d_{family}.txt'
    cmd = [sys.executable, HERE / 'eval_p2.py',
           '--formal', out_root / 'receiver_p2d' / f'formal_{family}',
           '--p1formal', out_root / 'receiver_p1' / 'formal',
           '--jobs', jobs_file,
           '--sets', sets, '--out', ev,
           '--extra-formal', out_root / 'receiver_p2' / 'formal', out_root / 'receiver_p2b' / 'formal',
           '--extra-methods', 'rxnull,rxlogit,ccalign']
    sh(cmd)
    return ev


def stage_decide(family, report_root):
    dec = report_root / 'receiver_p2d' / f'decision_p2d_{family}.json'
    sh([sys.executable, HERE / 'decide_p2d.py', '--eval', report_root / 'receiver_p2d' / f'eval_{family}',
        '--out', dec])
    return json.loads(dec.read_text())


def run_family(family, arms, out_root, report_root, sets, workers):
    t0 = time.time()
    print(f'=== ROUND {family}: arms={arms} ===', flush=True)
    stage_search(family, arms, out_root, sets, workers)
    stage_select(family, arms, out_root, report_root, sets)
    stage_formal(family, out_root, report_root, workers, sets)
    ev = stage_eval(family, out_root, report_root, sets)
    dec = stage_decide(family, report_root)
    print(f'=== ROUND {family} done in {time.time() - t0:.0f}s: {dec["outcome"]} ===', flush=True)
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default=HERE.parent.parent / 'outputs')
    ap.add_argument('--report-root', default=HERE.parent.parent / 'reports')
    ap.add_argument('--sets', default=str(HERE.parent.parent / 'reports' / 'receiver_p1' / 'sets_p1.json'))
    ap.add_argument('--families', default='r1')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-rounds', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    out_root, report_root = Path(args.out_root), Path(args.report_root)
    todo = args.families.split(',')
    state = dict(rounds=[], started=time.strftime('%Y-%m-%dT%H:%M:%S'))
    rounds_done = 0
    while todo and rounds_done < args.max_rounds:
        fam = todo.pop(0)
        if fam not in CANDIDATES:
            raise SystemExit(f'unknown family {fam}; registered: {list(CANDIDATES)}')
        if args.dry_run:
            print(f'[dry-run] would run family {fam}: {CANDIDATES[fam]}')
            continue
        dec = run_family(fam, CANDIDATES[fam], out_root, report_root, args.sets, args.workers)
        state['rounds'].append(dict(family=fam, outcome=dec['outcome'], main_arm=dec['main_arm'], arms=CANDIDATES[fam]))
        rounds_done += 1
        if dec['outcome'] == 'GO_CROSS_DATASET':
            state['stop_reason'] = 'GO_CROSS_DATASET: winner beats ccalign; WiSig transfer round is next (needs user go-ahead)'
            break
        # ITERATE / STOP_TAIL for this family: auto-continue to the next registered candidate.
        # The chain only stops when the candidate ladder is exhausted (each new family is
        # registered in CANDIDATES only after a post-mortem of the previous round, which is
        # the pre-registration discipline: no arm runs before its mechanism argument exists).
        idx = int(''.join(filter(str.isdigit, fam)) or 0)
        nxt = f'r{idx + 1}'
        if nxt in CANDIDATES:
            print(f'[chain] {dec["outcome"]} on {fam} -> auto-continue with {nxt}: {CANDIDATES[nxt]}', flush=True)
            todo.insert(0, nxt)
        else:
            state['stop_reason'] = f'{dec["outcome"]} on {fam} and no further candidate registered (last={fam})'
            break
    CHAIN_STATE.parent.mkdir(parents=True, exist_ok=True)
    CHAIN_STATE.write_text(json.dumps(state, indent=1))
    print(json.dumps(state, indent=1))


if __name__ == '__main__':
    main()
