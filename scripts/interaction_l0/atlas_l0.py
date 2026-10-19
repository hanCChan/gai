#!/usr/bin/env python3
"""L0-A: model-free physical atlas of WiSig ManyRx (CPU, numpy only, no model, no fitting).

Questions, all answered from IQ descriptors (physics.py), not from CNN features:
  A1 additivity: how much of CFO / SNR / STF tone-shape variation is TX main, RX main, or TX x RX
     interaction (split-half cross-fitted energies; dates 0/1 separately, then averaged)?
  A2 attribution of the tone-shape interaction: does i_{t,r} resemble i_{t,r'} more when RX r, r'
     are spatially close or share a hardware model (permutation null over RX labels)? does its
     size grow with TX-RX distance?
  A3 internal RX vs source RX: where do internal RX offsets (CFO, tone-shape RX main, SNR) fall
     relative to the 14 source RX, using only source-estimated TX effects as reference?
  A4 association of A3 with the published per-RX development accuracies (10 RX, exploratory).
Reads source RX (dates 0/1) and internal development RX (dates 0-2) through receiver_r01.data,
whose asserts block final RX, date 3 and the equalized version.

    python atlas_l0.py --root /home/keyan/code-y-r0r1 --out OUT_ATLAS
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from link_core import complete_grid, cell_means, crossfit_energies, effects, spearman
from physics import to_complex, stf_descriptors, grid_distance

PERMUTATIONS = 2000
MIN_PERIODICITY = 0.5   # fixed: packets whose window is not inside the L-STF are reported, then dropped


def descriptors(data):
    d = stf_descriptors(to_complex(data['x']))
    feats = dict(cfo_khz=d['cfo_hz'][:, None] / 1e3, snr_db=d['snr_db'][:, None], tone_shape=d['tone_shape'])
    return feats, d['periodicity']


def shares(f, data, keep):
    tx, rx, day, pos = data['y'], data['rx'], data['day'], data['position']
    rows = []
    for d in np.unique(day[keep]):
        m = keep & (day == d)
        txs, rxs = complete_grid(tx, rx, m)
        A, B = (cell_means(f, tx, rx, m, txs, rxs, h, pos) for h in (0, 1))
        rows.append(crossfit_energies(A, B))
    e = {k: float(np.mean([r[k] for r in rows])) for k in ('tx', 'rx', 'link')}
    tot = sum(max(v, 0) for v in e.values())
    return {**e, **{f'{k}_share': max(e[k], 0) / tot for k in ('tx', 'rx', 'link')}}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root / 'src')]
    from receiver_r01.data import wisig
    import pandas as pd
    reg = json.loads((root / 'configs/receiver_r01/registration.json').read_text())
    hw = {h['rx']: h['hardware'] for h in json.loads((root / 'configs/wisig_w01/registration.json').read_text())['hardware_and_coverage']}
    names, tnames = reg['wisig']['rx_names'], reg['wisig']['tx_names']
    roles = reg['wisig']['roles']
    pools = dict(source=wisig(reg, roles['source'], [0, 1], [0, 200]),
                 internal=wisig(reg, roles['internal_rx'], [0, 1, 2], [0, 200]))
    desc = {k: descriptors(v) for k, v in pools.items()}
    result = dict(min_periodicity=MIN_PERIODICITY, permutations=PERMUTATIONS)
    for k, (f, per) in desc.items():
        result[f'{k}_packets'] = int(len(per)); result[f'{k}_low_periodicity_fraction'] = float((per < MIN_PERIODICITY).mean())

    # A1 additivity on source RX (dates 0/1) and, descriptively, on internal RX (dates 0/1)
    a1 = []
    for pool in ('source', 'internal'):
        data = pools[pool]; f, per = desc[pool]; keep = (per >= MIN_PERIODICITY) & (data['day'] < 2)
        for name, v in f.items():
            a1.append(dict(pool=pool, descriptor=name, **shares(v, data, keep)))
    pd.DataFrame(a1).to_csv(out / 'A1_additivity.csv', index=False)

    # A2 tone-shape interaction attribution on source RX: pairwise RX similarity per TX
    src = pools['source']; f, per = desc['source']; keep = (per >= MIN_PERIODICITY) & (src['day'] < 2)
    pairs = []
    for d in (0, 1):
        m = keep & (src['day'] == d); txs, rxs = complete_grid(src['y'], src['rx'], m)
        A, B = (cell_means(f['tone_shape'], src['y'], src['rx'], m, txs, rxs, h, src['position']) for h in (0, 1))
        iA, iB = effects(A)[3], effects(B)[3]
        for ti, t in enumerate(txs):
            for (j, r), (l, s) in [((j, r), (l, s)) for j, r in enumerate(rxs) for l, s in enumerate(rxs) if j < l]:
                sim = 0.5 * (iA[ti, j] @ iB[ti, l] + iB[ti, j] @ iA[ti, l])   # noise-free in expectation
                pairs.append(dict(date=d, tx=int(t), rx_a=int(r), rx_b=int(s), similarity=float(sim),
                                  rx_distance_m=grid_distance(names[r], names[s]), same_hardware=hw[names[r]] == hw[names[s]],
                                  link_distance_gap_m=abs(grid_distance(tnames[t], names[r]) - grid_distance(tnames[t], names[s]))))
        for ti, t in enumerate(txs):
            for j, r in enumerate(rxs):
                pairs.append(dict(date=d, tx=int(t), rx_a=int(r), rx_b=-1, similarity=float(iA[ti, j] @ iB[ti, j]),
                                  tx_rx_distance_m=grid_distance(tnames[t], names[r])))
    P = pd.DataFrame(pairs); P.to_csv(out / 'A2_pairs.csv', index=False)
    cross = P[P.rx_b >= 0]; self_ = P[P.rx_b < 0]
    rxs_all = sorted(set(cross.rx_a) | set(cross.rx_b)); rng = np.random.default_rng(20261019)

    def stat(frame, perm=None):
        ra, rb = frame.rx_a.to_numpy(), frame.rx_b.to_numpy()
        if perm is not None: ra, rb = perm[ra], perm[rb]
        dist = np.array([grid_distance(names[x], names[y]) for x, y in zip(ra, rb)])
        same = np.array([hw[names[x]] == hw[names[y]] for x, y in zip(ra, rb)])
        sim = frame.similarity.to_numpy()
        return spearman(dist, sim), float(sim[same].mean() - sim[~same].mean())
    obs = stat(cross); null = []
    lookup = np.arange(max(rxs_all) + 1)
    for _ in range(PERMUTATIONS):
        perm = lookup.copy(); perm[rxs_all] = rng.permutation(rxs_all); null.append(stat(cross, perm))
    null = np.array(null)
    result['A2'] = dict(rho_similarity_vs_rx_distance=obs[0], p_distance_one_sided=float((null[:, 0] <= obs[0]).mean()),
                        same_hardware_similarity_gap=obs[1], p_hardware_one_sided=float((null[:, 1] >= obs[1]).mean()),
                        mean_cross_rx_similarity=float(cross.similarity.mean()), mean_self_energy=float(self_.similarity.mean()),
                        rho_link_energy_vs_tx_rx_distance=spearman(self_.tx_rx_distance_m, self_.similarity),
                        note='RX labels permuted jointly (Mantel-style); pairs share RX so they are not independent')

    # A3 internal RX relative to source RX: offsets against source-estimated TX effects
    def rx_offsets(pool):
        data = pools[pool]; f, per = desc[pool]; rows = []
        for r in np.unique(data['rx']):
            for d in np.unique(data['day'][data['rx'] == r]):
                m = (data['rx'] == r) & (data['day'] == d) & (per >= MIN_PERIODICITY)
                if not m.any(): continue
                row = dict(pool=pool, rx=int(r), receiver=names[r], hardware=hw[names[r]], date=int(d), packets=int(m.sum()))
                for name, v in f.items():
                    tx_mean = {t: v[m & (data['y'] == t)].mean(0) for t in np.unique(data['y'][m])}
                    ref = {t: ref_tx[name][d if d < 2 else 1][t] for t in tx_mean if t in ref_tx[name][d if d < 2 else 1]}
                    off = np.mean([tx_mean[t] - ref[t] for t in ref], axis=0)
                    row[f'{name}_offset'] = float(off[0]) if off.size == 1 else float(np.linalg.norm(off))
                    if name == 'snr_db': row['snr_db_median'] = float(np.median(v[m, 0]))
                rows.append(row)
        return rows
    ref_tx = {}
    for name, v in desc['source'][0].items():
        ref_tx[name] = {}
        for d in (0, 1):
            m = keep & (src['day'] == d)
            # grand + TX effect over the 14 source RX; internal date 2 is referenced to source date 1
            ref_tx[name][d] = {t: v[m & (src['y'] == t)].mean(0) for t in np.unique(src['y'][m])}
    O = pd.DataFrame(rx_offsets('source') + rx_offsets('internal')); O.to_csv(out / 'A3_rx_offsets.csv', index=False)
    s = O[(O.pool == 'source')]
    for col in ('cfo_khz_offset', 'tone_shape_offset', 'snr_db_median'):
        mu, sd = s[col].mean(), s[col].std(ddof=1)
        O[f'{col}_z_vs_source'] = (O[col] - mu) / sd
    O.to_csv(out / 'A3_rx_offsets.csv', index=False)
    result['A3_source_cfo_rx_offset_sd_khz'] = float(s.cfo_khz_offset.std(ddof=1))

    # A4 exploratory association with published development accuracy (mean over all R0/R1 official runs)
    acc_path = root / 'reports/receiver_r01/2026-10-13/per_RX_date.csv'
    if acc_path.exists():
        acc = pd.read_csv(acc_path)
        acc = acc[(acc.dataset == 'wisig') & (acc.backbone == 'official') & acc.role.isin(['internal_seen_date', 'internal_unseen_date'])]
        acc = acc.assign(date=np.where(acc.role == 'internal_unseen_date', 2, acc.date)).groupby(['rx', 'date']).accuracy.mean().rename('mean_acc').reset_index()
        J = O[O.pool == 'internal'].merge(acc, on=['rx', 'date'])
        J.to_csv(out / 'A4_internal_join.csv', index=False)
        R = J.groupby(['rx', 'receiver', 'hardware']).mean(numeric_only=True).reset_index()
        result['A4'] = {f'rho_acc_vs_abs_{c}': spearman(R[c].abs(), R.mean_acc) for c in
                        ('cfo_khz_offset_z_vs_source', 'tone_shape_offset_z_vs_source', 'snr_db_median_z_vs_source')}
        result['A4']['acc_by_hardware'] = R.groupby('hardware').mean_acc.mean().to_dict()
        result['A4']['note'] = '10 internal RX, development data already viewed: exploratory only'
    (out / 'atlas.json').write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == '__main__':
    main()
