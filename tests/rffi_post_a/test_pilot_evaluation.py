"""R0 evaluation path end to end on a random OfficialCNN and synthetic WiSig-shaped tables (no real
data): heads, manifests built as in Stage A, rows reproducible from the saved scores."""
import copy
import json

import numpy as np
import torch

from conftest import ROOT, load_by_path

DIAG = json.loads((ROOT / 'configs/rffi_post_a/diagnostic_matrix.json').read_text())
PILOT = json.loads((ROOT / 'configs/rffi_post_a/pilot_matrix.json').read_text())
REG = json.loads((ROOT / 'configs/rffi_post_a/registration.json').read_text())


def test_run_pilot_evaluate_fit_end_to_end_on_synthetic_wisig(tmp_path, monkeypatch):
    rp = load_by_path('run_pilot_eval', 'scripts/rffi_post_a/run_pilot.py')
    from rffi_calibration import registry as regmod
    from rffi_calibration.datasets import wisig as wmod
    from rffi_calibration.encoders import build_wisig_official
    from rffi_calibration.sampling import RecordTable, build_manifest, manifest_sha
    rng = np.random.default_rng(0)

    def fake_load(reg, guard, rxs, dates, positions, numeric=False, purpose=''):
        tx, rx, date, pos = [], [], [], []
        for r in rxs:
            for d in dates:
                for t in range(10):
                    tx += [t] * 30
                    rx += [r] * 30
                    date += [d] * 30
                    pos += list(range(t * 200, t * 200 + 30))
        rx_a, d_a = np.array(rx), np.array(date)
        tab = RecordTable('wisig', np.array(tx), rx_a, d_a, np.array(pos), 'cache_position',
                          capture=np.array([f'rx{a:02d}_day{b}' for a, b in zip(rx_a, d_a)]))
        x = (rng.standard_normal((len(tx), 1, 256, 2)) + np.array(tx)[:, None, None, None] * .3).astype(np.float32)
        return tab, x

    monkeypatch.setattr(wmod, 'load', fake_load)
    job = PILOT['jobs'][2]                                                       # w0 P_RX
    fits = tmp_path / 'fits'
    (fits / job['job_id']).mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(dict(model=build_wisig_official(classes=10).model.state_dict()), fits / job['job_id'] / 'last.pt')
    cfg = tmp_path / 'cfg'
    cfg.mkdir()
    for f in ('datasets.json', 'roles.json', 'registration.json', 'pilot_matrix.json'):
        (cfg / f).write_bytes((ROOT / 'configs/rffi_post_a' / f).read_bytes())
    sq = json.loads((ROOT / 'configs/rffi_calibration/registration.json').read_text())['support_query']['wisig']
    d = copy.deepcopy(DIAG)
    w0 = next(t for t in d['tasks'] if t['task'] == 'A_wisig_w0_T0_s20261013')
    tab, _ = fake_load(None, None, w0['held_rx'], [0, 1], (0, 200))
    for r in w0['held_rx']:
        m = tab.rx == r
        sub = RecordTable('wisig', tab.tx[m], tab.rx[m], tab.date[m], tab.position[m], 'cache_position',
                          capture=tab.capture[m])
        man = build_manifest(sub, r, 10, sq['rule'], q_per_class=sq['q_per_class'], n_draws=20)
        w0['manifest_sha256_prefix'][str(r)] = manifest_sha(man)[:16]
    (cfg / 'diagnostic_matrix.json').write_text(json.dumps(d))
    monkeypatch.setattr(rp, 'CONFIG', cfg)
    monkeypatch.setattr(rp, 'FITS', fits)
    monkeypatch.setattr(rp, 'OUT', tmp_path / 'R0')
    reg = regmod.Registry(cfg)
    res = dict(REG['requests_not_active']['R0'], active=True)
    rows, logits, meta = rp.evaluate_fit(reg, res, job, 'last.pt', 'cpu', PILOT)
    assert {r['head'] for r in rows} == {'H0', 'H1e', 'H1', 'H2L', 'H5a', 'H5b', 'H5c[k0=1]'}
    assert {str(r['rx']) for r in rows} == {'1', '7', '16', '30'} and not meta['skipped']
    vr = load_by_path('verify_eval', 'scripts/rffi_post_a/verify_results.py')
    chk = vr.check_rows_against_scores(rows, logits, lambda r: f'{r["rx"]}|{r["head"]}|{r["m"]}|{r["draw"]}',
                                       lambda r: logits[f'{r["rx"]}|y_query'])
    assert chk['ok'] and chk['checked'] == len(rows)
