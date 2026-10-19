"""CPU smoke test of run_l0.wisig_physics / attribution with a CFO-only stand-in classifier."""
import numpy as np
from physics import to_cache, stf_descriptors, to_complex
from smoke_atlas import packet_batch
import run_l0


def pool(rxs, days, tx_cfo, rx_cfo, tx_tone, rx_tone, link, rng, n=20):
    xs, ys, rr, dd, pp = [], [], [], [], []
    for r in rxs:
        for d in days:
            for t in range(10):
                xs.append(to_cache(packet_batch(n, tx_tone[t] + rx_tone[r] + link[t, r], tx_cfo[t] + rx_cfo[r], 25.0, rng)))
                ys += [t] * n; rr += [r] * n; dd += [d] * n; pp += list(range(n))
    return dict(x=np.concatenate(xs), y=np.array(ys), rx=np.array(rr), day=np.array(dd), position=np.array(pp))


def main():
    rng = np.random.default_rng(0)
    tx_cfo = np.linspace(-40e3, 40e3, 10); rx_cfo = rng.normal(scale=3e3, size=20); rx_cfo[14:] += 12e3  # internal RX shifted
    tx_tone = rng.normal(scale=1.5, size=(10, 12)); rx_tone = rng.normal(scale=1.0, size=(20, 12)); link = rng.normal(size=(10, 20, 12))
    pools = dict(source=pool(range(14), (0, 1), tx_cfo, rx_cfo, tx_tone, rx_tone, link, rng),
                 internal=pool(range(14, 20), (2,), tx_cfo, rx_cfo, tx_tone, rx_tone, link, rng))
    phys = run_l0.wisig_physics(pools)
    print('sigma_b_hz', round(phys['sigma_b_hz']))
    import pandas as pd
    V = pd.DataFrame(phys['validation']); E = run_l0.ENGINEERING
    print(V[['intervention', 'requested', 'error_median_abs', 'error_rms', 'tone_leakage_rms', 'input_rel_change_mean']].round(4).to_string(index=False))
    assert (V[V.requested == 'cfo_hz'].error_median_abs < E['cfo_error_hz_max']).all()
    assert (V[V.requested == 'tone_log_power'].error_rms < E['tone_error_rms_max']).all()
    assert (V[V.requested == 'snr_db'].error_median_abs.fillna(0) < E['snr_error_db_max']).all() and (V.nonfinite == 0).all()
    assert 1.5e3 < phys['sigma_b_hz'] < 6e3
    R = phys['ref'][1]
    cfo_model = lambda x: R['txs'][np.abs(stf_descriptors(to_complex(x))['cfo_hz'][:, None] - R['cfo'][None]).argmin(1)]
    src = pools['source']; base = cfo_model(src['x'])
    for name, (xp, target) in phys['interventions'].items():
        pr = cfo_model(xp); flip = (pr != base).mean()
        moved = (pr == target).mean() if target is not None else np.nan
        print(f'{name:15s} flip {flip:.2f} moved {moved:.2f}')
        if name == 'cfo_tx_swap': assert moved > 0.9
        if name.startswith('tone') or name == 'snr_link_swap': assert flip < 0.05, name
    rows = run_l0.attribution(cfo_model(pools['internal']['x']), pools['internal'], phys['ref'], rng)
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    cfo = next(r for r in rows if r['cue'] == 'cfo')
    assert cfo['cnn_errors'] > 50 and cfo['agree_given_cnn_wrong'] > 0.99 and cfo['agree_given_cnn_wrong_null'] < 0.6
    print('PASS run_l0 physics smoke')


if __name__ == '__main__':
    main()
