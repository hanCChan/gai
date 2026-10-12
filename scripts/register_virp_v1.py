#!/usr/bin/env python3
"""Freeze v1 before accessing any new signal-derived results."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='configs/virp/v1_registration.json')
    args = parser.parse_args()
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    prior_path = ROOT / 'configs/shen_p2a8/g1_registration.json'
    prior = json.loads(prior_path.read_text())
    files = [x for x in prior['files'] if x['role'] == 'train']
    assert len(files) == 10 and {x['receiver'] for x in files} == set(prior['train_receivers'])
    code_paths = ['scripts/register_virp_v1.py', 'scripts/audit_virp_v0.py',
                  'src/virp/spectral.py', 'src/shen_p2a/representations.py',
                  'src/shen_stage1/representations.py', 'src/shen_stage1/frequency.py',
                  'docs/virp/V1_PREREGISTRATION.md', 'tests/test_virp_v0.py']
    user_text = ROOT / 'docs/virp/USER_RULES_20261012.txt'
    reg = {
        'schema_version': 1, 'experiment': 'VIRP-v1', 'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'client_timezone': 'Asia/Shanghai', 'user_rules_sha256': sha(user_text),
        'previous_registration_sha256': sha(prior_path),
        'immutable_code_sha256': {p: sha(ROOT / p) for p in code_paths},
        'terminology': 'frequency-response-inspired spectral augmentation; capture-chain style, not measured causal RX hardware',
        'source_receivers': prior['train_receivers'], 'source_files': files,
        'source_receiver_folds': prior['source_receiver_folds'],
        'source_samples_per_TX_RX': 200, 'sample_selection_seed': 20261012,
        'v0_output': 'outputs/virp/v1_v0_20261012',
        'v0': {
            'seed': 20261012, 'samples': 'all G1 selected raw float64 IQ rows, 200/TX/RX, 20000 packets; raw SHA rechecked',
            'fold_fit': 'each fold: fit only five training RX; the other five are source-held-out audits only',
            'n_fft': 8192, 'basis': 'cos/sin of harmonics 1,2,3 of normalized frequency; six nonconstant periodic bases',
            'profile': 'class-balanced mean ln(abs(FFT(packet_RMS_normalized_IQ))/sqrt(8192)+1e-12)',
            'response': 'orthogonal projection of donor-source profile difference onto six bases; remove global mean; shrink whole curve only if max(abs(dB))>3; exp(log_response)',
            'variants_per_packet': 4, 'variant_rule': 'each of four other RX in its fitting fold, strength exactly 1; no per-packet cherry-picking',
            'sigma': 'scalar population SD of unperturbed Full CIS across only five fitting RX, recomputed from raw; log10(power ratio) units',
            'cis': 'historical unchanged boxcar128/hop64, crop .3, 52x126, RMS normalization, epsilon1e-12 complex STFT; complex128',
            'delta': 'RMS((C(Tx)-C(x))/(sigma+1e-12)); packet_delta=median across four registered donor variants',
            'hard_gates_each_fold': {
                'median_packet_delta_min': .10, 'packet_fraction_delta_ge_005_min': .80,
                'all_realizations_delta_q95_max': 1., 'nonfinite_count_max': 0,
                'max_abs_response_db': 3., 'all_realizations_iq_correlation_min': .98,
                'negative_control_delta_max': 1e-6,
                'median_packet_core_delta_min': .05,
                'low_energy_residual_share_q95_max': .50,
                'boundary_residual_share_q95_max': .50,
                'extreme_cis_fraction_increase_max': .001,
                'coherence_change_absolute_q95_max': .05,
                'profile_half_split_stable_pair_fraction_min': .80,
                'profile_loo_tx_relative_change_q95_max': .25,
                'profile_smooth_explained_fraction_median_min': .50,
                'heldout_spectral_improvement_fraction_min': .40,
                'natural_cis_scale_ratio_min': .10, 'natural_cis_scale_ratio_max': 10.
            },
            'negative_controls': ['FFT/IFFT with H=1', 'constant full-band +3 dB gain through identical path'],
            'artifact_mask': 'union of clean denominator bottom1% within packet, abs(clean_CIS)>10 (100dB), first/last2 ratio frames; no changing epsilon',
            'coherence': 'lag1024, diagnostic inherited conditional LoRa period; never proof of preamble or compensate CFO',
            'profile_stability': 'first/last100 rows per TX by original row order; 80% ordered pairs cosine>=.8 and half norm ratio in [.5,2]; no resampling',
            'tx_dominance': 'leave each TX out of class-balanced profiles; projected style relative L2 change across pair/TX, q95<=.25',
            'smoothness': 'six-basis projection energy divided by demeaned raw profile-difference energy, median>=.5',
            'heldout_spectrum': 'responses frozen before accessing other five source RX profiles; for each heldout RX best fitted-source+its registered response versus best clean fitted-source, >=10% L2 improvement for >=2/5 RX',
            'natural_cis': 'same-TX mean CIS centroid distances / fitting sigma: all cross-source-RX pairs versus same-RX synthetic mean displacement; ratio of medians [.1,10]; unpaired packets, not exact matched-signal ground truth',
            'packet_level_export': True, 'no_classifier_training': True,
            'early_stop': 'complete registered V0 audits and document failed gates; no V1/V2/V3 if any hard gate fails',
        },
        'v1': {
            'run_count': 10, 'methods': ['B0', 'B1', 'B2', 'B3', 'B4'], 'seed': 20261012,
            'architecture': 'historical FactorizedAuthorCNN(rank16), shared for all methods',
            'epochs': 20, 'optimizer': 'Adam', 'lr': .0003, 'weight_decay': .0001,
            'tuning': 'zero hyperparameter sweeps for ALL methods; fixed scientifically defined recipes; no unequal search opportunity',
            'sample_budget': '10000 unique fitting packets/fold, each once/epoch; 50 base packets/batch, 5 views/base for all methods',
            'sampler': 'one sample per TX x RX cell/batch; independent seeded permutations without replacement per epoch',
            'B0': 'five copies clean per base sample, ordinary CE (equal computation/exposure budget)',
            'B1': 'clean + four independently AWGN augmented IQ views, SNR uniform[10,40]dB, complex circular noise; before unchanged CIS; .5*(CE_clean+mean CE_aug)',
            'B2': 'clean + four donor-specific frozen spectral augmentations, .5*(CE_clean + mean CE_aug)',
            'B3': 'same views as B2; .5*(CE_clean + mean CE_aug) + .25 * (logsumexp(5*group risks)-ln(40))/5',
            'B3_groups': 'TX x donor-index v, 40 groups, each represented by five RX samples in each balanced batch',
            'B4': 'no new augmentation; standard multiplicative-weight Group DRO, groups TX x source_RX=50, eta .01, weights normalized each batch; shared regularization',
            'selection': 'best mean per-RX held-out Macro-F1 checkpoint, tie earliest epoch; strongest baseline=max mean F1 of B0/B1/B4, tie B0,B1,B4',
            'gates': {
                'B2_minus_B1_mean_macro_f1_min': .01, 'B2_minus_B1_each_fold_min': 0,
                'B3_minus_best_baseline_mean_macro_f1_min': .02, 'B3_minus_best_baseline_each_fold_min': .005,
                'B3_minus_B2_pooled_worst_TX_recall_min': .05, 'B3_minus_B2_mean_macro_f1_min': -.005,
                'severe_TX_RX_recall_loss_max': .10, 'severe_TX_RX_min_support': 50,
                'all_models_source_probe_accuracy_min': .90
            },
            'metrics': 'per-RX Macro-F1, Worst-RX, per-RX worst-TX, pooled worst-TX, TX/RX recall differences and severe-degradation fraction',
            'no_reruns_or_seed_deletion': True
        },
        'v2': {
            'seeds': [20261012, 20261013, 20261014],
            'source_confirmation': 'repeat same five-method two-fold protocol at other two seeds; retain initial seed; no recipe change',
            'go_rule': 'same V1 gates on means across seeds; B3 gain over fixed strongest seed1 baseline positive for all seeds; no severe TX/RX loss at any seed',
            'development_epochs': 'freeze ceil(mean source-CV best epochs across folds and seeds), separately for every method; no dev selection',
        },
        'v3': {
            'shen_development': prior['development_receivers'], 'shen_source_packets': 20000,
            'source_fit': 'all ten source RX only, 200 packets/TX/RX, same method budgets; three seeds; development evaluation after frozen fit',
            'single_source_definition': 'same six bases and ±3dB bound; fixed coefficients independent N(0,.05^2) natural-log amplitude, shrink only; RNG seed+epoch+sample identity; four v per sample',
            'single_source_no_shen_empirical_prior': True,
            'OSU_direction_separation': 'freeze direction A source-only; target readonly audit B cannot tune anything; fixed model evaluation C only if included',
            'OSU_metadata': 'cf32 little endian, sample rate string 1 MS/s, 25 files, finite nonzero signals, byte counts divisible by8',
            'OSU_windows': 'historical aligned-to-file 8192 sample blocks, lag1024 coherence>=.8, at least20/class, deterministic linspace max200/class; no strict-preamble claim',
            'OSU_bad_target': 'stop direction, preserve failures; any new frontend protocol separately registered',
            'OSU_epochs': 20, 'OSU_seeds': [20261012, 20261013, 20261014],
            'OSU_target_audit_may_not_inform_models': True, 'OSU_no_research_history_blind_claim': True,
            'OSU_single_source_V0': 'same numerical/representation/IQ gates on direction source only before training; no inter-RX style gate in single-source setting',
        },
        'forbidden': ['read four Shen dev RX signals before V3', 'read six historical locked RX signals in v1', 'use OSU target to tune',
                      'change ±3dB, bases, CFO, STFT, risk weights, thresholds after results', 'start frequency-band scores/Gates/CrossAttention'],
        'stop_policy': 'Any V0 or V1 hard failure stops this candidate. New hypotheses require independent registration and disclosure of this negative result.',
        'literature_gate': 'source mechanism novelty not established; no submission novelty claim without primary fulltext comparison and fair strong baselines',
    }
    with target.open('x') as f:
        json.dump(reg, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    seal = {'registration_sha256': sha(target), 'registered_utc': reg['registered_utc'], 'new_signal_results_read_before_freeze': False}
    with target.with_suffix('.seal.json').open('x') as f:
        json.dump(seal, f, indent=2)
        f.write('\n')
    print(json.dumps(seal))


if __name__ == '__main__':
    main()
