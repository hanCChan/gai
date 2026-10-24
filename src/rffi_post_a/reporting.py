"""D0 observation audit (plan §5) from existing Stage A summaries/CSV/curves only, plus independent
replay helpers and a dependency-free SVG line plot (matplotlib is not installed in env-3.12).

Nothing here opens signal arrays, features, weights or per-query score files.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

FIXED_DIFFS = [('H4', 'H1'), ('H4', 'H3'), ('H4', 'H4-pooled'), ('H4', 'H4-AK'), ('H3', 'H1'), ('H5b', 'H1'),
               ('H5a', 'H1'), ('H4-pooled', 'H1'), ('H2L', 'H1')]
CURVE_HEADS = ('H1', 'H1e', 'H2L', 'H3', 'H4', 'H4-pooled', 'H5b')


def read_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['m'] = int(r['m'])
        r['draw'] = int(r['draw'])
        for k in ('acc', 'macro_f1', 'nll', 'brier', 'ece'):
            r[k] = float(r[k])
    return rows


def _mean(v):
    return sum(v) / len(v)


def _sd(v):
    if len(v) < 2:
        return 0.
    mu = _mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))


def per_rx_table(rows, dataset):
    """(split, rx, head, m) -> draw mean / draw SD of accuracy, macro-F1, NLL, per-class recall."""
    g = defaultdict(list)
    for r in rows:
        g[(r.get('split', ''), r['rx'], r['head'], r['m'])].append(r)
    out = []
    for (split, rx, head, m), rs in sorted(g.items()):
        accs = [r['acc'] for r in rs]
        rec = {k: _mean([float(r[k]) for r in rs if r.get(k) not in (None, '')])
               for k in rs[0] if k.startswith('recall_') and any(r.get(k) not in (None, '') for r in rs)}
        out.append(dict(dataset=dataset, split=split, rx=rx, head=head, m=m, n_draws=len(rs), acc_mean=_mean(accs),
                        acc_draw_sd=_sd(accs), macro_f1_mean=_mean([r['macro_f1'] for r in rs]),
                        nll_mean=_mean([r['nll'] for r in rs]), **rec,
                        selected=';'.join(sorted({r.get('selected', '') for r in rs if r.get('selected')}))))
    return out


def rx_equal(table, head, m):
    """Receiver-equal mean (draws already averaged). Physical RX with the same id in two WiSig folds
    cannot occur: selection RX are disjoint across folds."""
    v = [r['acc_mean'] for r in table if r['head'] == head and r['m'] == m]
    return (_mean(v), min(v), len(v)) if v else (None, None, 0)


def fixed_differences(table, budgets=(1, 2, 5, 10)):
    idx = {(r['split'], r['rx'], r['head'], r['m']): r['acc_mean'] for r in table}
    keys = sorted({(r['split'], r['rx']) for r in table})
    out = []
    for a, b in FIXED_DIFFS:
        for m in budgets:
            d = [100 * (idx[(s, rx, a, m)] - idx[(s, rx, b, m)]) for s, rx in keys
                 if (s, rx, a, m) in idx and (s, rx, b, m) in idx]
            if d:
                out.append(dict(a=a, b=b, m=m, n_rx=len(d), mean_pp=_mean(d), min_pp=min(d), max_pp=max(d),
                                n_positive=sum(x > 0 for x in d), per_rx_pp=[round(x, 3) for x in d]))
    return out


def h5c_members(table, budgets=(1, 5)):
    """Fixed H5c grid members against the cross-fitted H5c* (WiSig only has the latter)."""
    out = []
    for m in budgets:
        row = dict(m=m)
        for h in ('H5c[k0=1]', 'H5c[k0=4]', 'H5c[k0=16]', 'H5c*'):
            mean, worst, n = rx_equal(table, h, m)
            row[h] = None if mean is None else 100 * mean
        out.append(row)
    return out


def budget_gains(table, heads=CURVE_HEADS):
    out = []
    for h in heads:
        a = {m: rx_equal(table, h, m)[0] for m in (1, 5, 10)}
        if None in a.values():
            continue
        out.append(dict(head=h, acc_m1=100 * a[1], gain_1_to_5_pp=100 * (a[5] - a[1]), gain_5_to_10_pp=100 * (a[10] - a[5])))
    return out


def replay_aggregate(rows, reference_csv):
    """Independent rows -> RX-equal aggregate, compared with the published Stage A aggregate CSV.
    Does not import the Stage A evaluator."""
    g = defaultdict(list)
    for r in rows:
        g[(r['head'], r['m'], r['rx'])].append(r['acc'])
    per = defaultdict(list)
    for (h, m, rx), v in g.items():
        per[(h, m)].append(_mean(v))
    with open(reference_csv) as f:
        ref = {(r['head'], int(r['m'])): r for r in csv.DictReader(f)}
    worst = 0.
    checked = 0
    for k, v in per.items():
        if k not in ref:
            continue
        worst = max(worst, abs(100 * _mean(v) - float(ref[k]['accuracy_pct'])),
                    abs(100 * min(v) - float(ref[k]['worst_rx_pct'])))
        checked += 1
    return dict(checked=checked, max_abs_pp=worst, ok=checked > 0 and worst < 1e-6)


def curve_audit(curve, status, last_k=5):
    steps = [c['step'] for c in curve]
    ce = [c['val_ce'] for c in curve]
    tail_s, tail_c = steps[-last_k:], ce[-last_k:]
    ms, mc = _mean(tail_s), _mean(tail_c)
    slope = sum((s - ms) * (c - mc) for s, c in zip(tail_s, tail_c)) / sum((s - ms) ** 2 for s in tail_s)
    total_drop = ce[0] - min(ce)
    last_drop = tail_c[0] - tail_c[-1]
    return dict(job=status['job']['job_id'], horizon=status['job']['steps'], eval_every=status['job']['eval_every'],
                best_step=status['best']['step'], best_val_ce=status['best']['ce'], final_val_ce=ce[-1],
                best_is_final=status['best']['step'] == steps[-1], tail_steps=[tail_s[0], tail_s[-1]],
                tail_slope_per_100_steps=100 * slope, tail_drop=last_drop,
                tail_drop_share_of_total=last_drop / total_drop if total_drop > 0 else None,
                n_points=len(curve))


# ------------------------------------------------------------------------------------------ SVG
def svg_lines(series, title, path, xlabel='shots m', ylabel='accuracy %', w=520, h=340):
    """series: {label: [(x, y), ...]}. Minimal, dependency-free line chart."""
    xs = [x for s in series.values() for x, _ in s]
    ys = [y for s in series.values() for _, y in s]
    if not xs:
        return
    x0, x1, y0, y1 = min(xs), max(xs), math.floor(min(ys) - 1), math.ceil(max(ys) + 1)
    L, R, T, B = 60, 130, 30, 45
    px = lambda x: L + (x - x0) / max(x1 - x0, 1e-9) * (w - L - R)
    py = lambda y: h - B - (y - y0) / max(y1 - y0, 1e-9) * (h - T - B)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="11">',
           f'<text x="{w / 2}" y="16" text-anchor="middle" font-size="12">{title}</text>',
           f'<line x1="{L}" y1="{h - B}" x2="{w - R}" y2="{h - B}" stroke="black"/>',
           f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h - B}" stroke="black"/>',
           f'<text x="{(L + w - R) / 2}" y="{h - 8}" text-anchor="middle">{xlabel}</text>',
           f'<text x="14" y="{(T + h - B) / 2}" transform="rotate(-90 14 {(T + h - B) / 2})" text-anchor="middle">{ylabel}</text>']
    for x in sorted(set(xs)):
        out.append(f'<text x="{px(x):.1f}" y="{h - B + 14}" text-anchor="middle">{x:g}</text>')
    for k in range(6):
        y = y0 + k * (y1 - y0) / 5
        out.append(f'<text x="{L - 6}" y="{py(y) + 4:.1f}" text-anchor="end">{y:.0f}</text>')
    for i, (lab, s) in enumerate(series.items()):
        c = colors[i % len(colors)]
        pts = ' '.join(f'{px(x):.1f},{py(y):.1f}' for x, y in sorted(s))
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" points="{pts}"/>')
        out.append(f'<text x="{w - R + 8}" y="{T + 14 * i + 8}" fill="{c}">{lab}</text>')
    out.append('</svg>')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(out))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
