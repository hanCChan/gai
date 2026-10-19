"""End-to-end CPU smoke test of atlas_l0.py on a fake root with synthetic WiSig-format caches.

Planted truth: CFO = TX offset + RX offset (purely additive); STF tone shape = TX + RX + link
(TX x RX) components; SNR depends on RX. The atlas must report CFO link share ~0 and a clearly
non-zero tone-shape link share. Uses the real registration (names, roles, hardware) from gai.
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from physics import to_cache, TONE_BINS, PERIOD, FS

GAI = Path(__file__).resolve().parents[2]  # repository root


def packet_batch(n, tone_db, cfo, snr_db, rng, L=256):
    phase = np.exp(2j * np.pi * rng.uniform(size=12))  # fixed STF phases within a call
    g = 10 ** (tone_db / 20)
    t = np.arange(L + PERIOD)
    out = []
    for _ in range(n):
        off = rng.integers(0, PERIOD)
        s = (g * phase * np.exp(2j * np.pi * np.outer(t, TONE_BINS) / PERIOD)).sum(1)[off:off + L]
        s = s * np.exp(2j * np.pi * cfo * np.arange(L) / FS)
        p = np.mean(np.abs(s) ** 2)
        out.append(s + (rng.normal(size=L) + 1j * rng.normal(size=L)) * np.sqrt(p / 10 ** (snr_db / 10) / 2))
    return np.array(out)


def main(hardware_structured=False):
    tmp = Path(tempfile.mkdtemp()); root = tmp / 'root'; cache = tmp / 'cache'
    for sub in ('configs/receiver_r01', 'configs/wisig_w01'):
        (root / sub).mkdir(parents=True)
    shutil.copytree(GAI / 'src/receiver_r01', root / 'src/receiver_r01')
    reg = json.loads((GAI / 'configs/receiver_r01/registration.json').read_text()); reg['wisig_cache'] = str(cache)
    (root / 'configs/receiver_r01/registration.json').write_text(json.dumps(reg))
    shutil.copy(GAI / 'configs/wisig_w01/registration.json', root / 'configs/wisig_w01/registration.json')
    rng = np.random.default_rng(0); cache.mkdir()
    tx_cfo = rng.normal(scale=20e3, size=10); rx_cfo = rng.normal(scale=3e3, size=32)
    tx_tone = rng.normal(scale=1.5, size=(10, 12)); rx_tone = rng.normal(scale=1.0, size=(32, 12))
    link_tone = rng.normal(scale=1.0, size=(10, 32, 12)); rx_snr = rng.uniform(15, 30, size=32)
    if hardware_structured:  # positive control: TX x RX-hardware-model interaction (e.g. RX nonlinearity)
        hw = {h['rx']: h['hardware'] for h in json.loads((GAI / 'configs/wisig_w01/registration.json').read_text())['hardware_and_coverage']}
        models = sorted(set(hw.values())); per_model = rng.normal(scale=1.5, size=(10, len(models), 12))
        for r, name in enumerate(reg['wisig']['rx_names']):
            link_tone[:, r] = 0.3 * link_tone[:, r] + per_model[:, models.index(hw[name])]
    roles = reg['wisig']['roles']
    for r in roles['source'] + roles['internal_rx']:
        for d in (0, 1, 2):
            xs, ys, ps = [], [], []
            for t in range(10):
                z = packet_batch(40, tx_tone[t] + rx_tone[r] + link_tone[t, r], tx_cfo[t] + rx_cfo[r], rx_snr[r], rng)
                xs.append(to_cache(z)); ys += [t] * 40; ps += list(range(0, 200, 5))
            np.savez(cache / f'rx{r:02d}_day{d}.npz', x=np.concatenate(xs), y=np.array(ys), position=np.array(ps))
    out = tmp / 'atlas'
    subprocess.run([sys.executable, str(Path(__file__).with_name('atlas_l0.py')), '--root', str(root), '--out', str(out)],
                   check=True, capture_output=True)
    import pandas as pd
    a1 = pd.read_csv(out / 'A1_additivity.csv'); src = a1[a1.pool == 'source'].set_index('descriptor')
    print(src[['tx_share', 'rx_share', 'link_share']].round(3).to_string())
    assert src.loc['cfo_khz', 'link_share'] < 0.02, 'planted CFO is additive'
    assert src.loc['tone_shape', 'link_share'] > 0.15, 'planted tone shape has a link component'
    assert src.loc['snr_db', 'rx_share'] > 0.5, 'planted SNR is an RX property'
    res = json.loads((out / 'atlas.json').read_text())
    assert res['source_low_periodicity_fraction'] < 0.01
    if hardware_structured:
        assert res['A2']['p_hardware_one_sided'] < 0.05, res['A2']
    else:
        assert res['A2']['p_hardware_one_sided'] > 0.05 and res['A2']['p_distance_one_sided'] > 0.05, res['A2']
    print('A2', {k: round(v, 3) if isinstance(v, float) else v for k, v in res['A2'].items() if k != 'note'})
    shutil.rmtree(tmp); print('PASS atlas smoke, hardware_structured =', hardware_structured)


if __name__ == '__main__':
    main(False); main(True)
