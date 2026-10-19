"""CPU smoke test of summarize_l0.py on fabricated outputs with run_l0.py's schema."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROLES = ('source', 'internal_seen_date', 'internal_unseen_date')


def fabricate(out, mechanism, rng):
    runs, er, se, at = [], [], [], []
    for m_i, method in enumerate(['ce', 'groupdro', 'mldg_rx', 'mldg_rx_date', 'pcgrad', 'swad']):
        for s_i, seed in enumerate([20261013, 20261014, 20261015]):
            rid = f'{method}_{seed}'; link = rng.uniform(0.2, 0.6)
            acc = 0.6 - (0.5 * link if mechanism else 0) + 0.05 * s_i + rng.normal(scale=0.02)
            ident = dict(run_id=rid, group='wisig/official/r01', dataset='wisig', backbone='official', fold=0, method=method, seed=seed)
            runs.append(dict(**ident, feat_link_ratio=link, feat_separability=rng.uniform(1, 2), feat_loro_proto=rng.uniform(.5, .9),
                             feat_rx_share=rng.uniform(0, .1), acc_source=0.9 + rng.normal(scale=0.005), acc_internal_seen_date=acc + .02,
                             acc_internal_unseen_date=acc))
            def rec(kind, k, gain, draw=None):
                r = dict(**ident, kind=kind, k=k, draw=draw)
                for h in ('frozen', 'refit'):
                    r[f'{h}_acc_source'] = 0.9 - (0.005 if kind != 'none' else 0)
                    r[f'{h}_acc_internal_seen_date'] = acc + gain
                    r[f'{h}_acc_internal_unseen_date'] = acc + gain
                    r[f'{h}_mean_logit_norm'] = 10.0; r[f'{h}_mean_max_prob'] = 0.8
                return r
            er.append(rec('none', 0, 0))
            for k in (4, 8, 16):
                er.append(rec('link', k, (0.04 * link / 0.4 if mechanism else 0) + rng.normal(scale=.003)))
                er.append(rec('rx', k, rng.normal(scale=.003)))
                er.append(rec('cond', k, (0.035 if mechanism else 0) + rng.normal(scale=.003)))
                for d in range(5): er.append(rec('random', k, rng.normal(scale=.002), d))
            for name, flip, moved in [('cfo_rx_+1sd', .3, np.nan), ('cfo_rx_-1sd', .3, np.nan), ('cfo_rx_+2sd', .6, np.nan),
                                      ('cfo_rx_-2sd', .6, np.nan), ('cfo_tx_swap', .9, .7), ('tone_link_swap', .02, np.nan),
                                      ('tone_rx_swap', .02, np.nan), ('tone_tx_swap', .05, .02), ('snr_link_swap', .01, np.nan)]:
                se.append(dict(**ident, intervention=name, flip_rate=flip + rng.normal(scale=.01), moved_to_target=moved, acc_drop_pp=10 * flip))
            for cue, ag, nu in (('cfo', .6, .15), ('tone', .1, .1)):
                at.append(dict(**ident, role='internal_unseen_date', cue=cue, cue_rule_accuracy=.3, agree_given_cnn_wrong=ag,
                               agree_given_cnn_wrong_null=nu, agree_given_cnn_right=.5, cnn_errors=100, excluded_low_periodicity=0))
    for n, rows in (('run_metrics', runs), ('erasure', er), ('sensitivity', se), ('attribution', at)):
        pd.DataFrame(rows).to_csv(out / f'{n}.csv', index=False)


def main():
    rng = np.random.default_rng(0)
    for mechanism, want in ((True, 'SUPPORTED'), (False, 'UNSUPPORTED_FOR_CURRENT_REPRESENTATION')):
        out = Path(tempfile.mkdtemp()); fabricate(out, mechanism, rng)
        subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_l0.py')), '--l0', str(out)], check=True, capture_output=True)
        s = json.loads((out / 'summary.json').read_text())
        print(mechanism, s['interaction_shortcut_verdict'], 'rho', round(s['statistical']['within_seed_partial_rho_link_ratio|acc_source+separability'], 2),
              'spec', round(s['intervention']['frozen_median_link_specific_pp'], 2), 'H_CFO', s['physical']['H_CFO_rx_offset'],
              'H_tone', s['physical']['H_tone_link_sensitivity'])
        assert s['interaction_shortcut_verdict'] == want
        assert s['physical']['H_CFO_rx_offset'] == 'SUPPORTED' and s['physical']['H_tone_link_sensitivity'] == 'UNSUPPORTED_FOR_CURRENT_REPRESENTATION'
    print('PASS summarize smoke')


if __name__ == '__main__':
    main()
