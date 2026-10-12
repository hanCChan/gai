#!/usr/bin/env python3
"""Freeze E0/E1/E2 before generating new fixed-epoch probabilities."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    prior_path=ROOT/'configs/shen_p2a8/g1_registration.json'
    prior=json.loads(prior_path.read_text())
    pipeline='outputs/shen_p2a8/g1_shen_v2'
    runs=[f'CV{fold}_{method}' for fold in (0,1) for method in ('full_cis','strict_cis')]
    fingerprints={}
    for run in runs:
        folder=ROOT/pipeline/'runs'/run
        for name in ('final.pt','config.json','history.csv','status.json'):
            p=folder/name
            if p.exists():fingerprints[str(p.relative_to(ROOT))]=sha(p)
        p=ROOT/pipeline/'logs'/(run+'.log')
        if p.exists():fingerprints[str(p.relative_to(ROOT))]=sha(p)
    cache_fingerprints={}
    for rx in prior['train_receivers']:
        folder=ROOT/'outputs/shen_p2a8/g1_cache_v1'/('shen_'+rx+'_train')
        for name in ('manifest.json','rows.npy','labels.npy','full_cis.npy','strict_cis.npy'):
            p=folder/name;cache_fingerprints[str(p.relative_to(ROOT))]=sha(p)
    code=['scripts/register_eisr_v1.py','scripts/run_eisr_v1.py','src/eisr/diagnostics.py',
          'scripts/train_shen_p2a8_g1.py','scripts/run_shen_p2a8_g1.py',
          'src/shen_p2a8/g1_data.py','src/shen_p2a/models.py','src/shen_evidence/models.py',
          'tests/test_eisr.py','docs/eisr/V1_PREREGISTRATION.md']
    reg=dict(schema_version=1,experiment='EISR-v1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        client_timezone='Asia/Shanghai',scope=['E0','E1','E2'],no_new_CNN_training=True,no_GPU=True,
        plan_sha256=sha(ROOT/'docs/eisr/USER_PLAN_20261012.txt'),user_rules_sha256=sha(ROOT/'docs/eisr/USER_RULES_20261012.txt'),
        prior_registration_sha256=sha(prior_path),source_receivers=prior['train_receivers'],source_receiver_folds=prior['source_receiver_folds'],
        pipeline=pipeline,runs=runs,asset_sha256=fingerprints,cache_sha256=cache_fingerprints,
        immutable_code_sha256={p:sha(ROOT/p) for p in code},output='outputs/eisr/v1_20261012',
        E0=dict(checkpoint='final.pt only; epoch must equal20; full config must match archived config',
            provenance='20 contiguous epochs in matching history and JSON logs; no validation-driven early stopping/LR/reset; archived source SHA checked; original training code compared to git history',
            provenance_limit='checkpoint stores no optimizer state or training-entrypoint hash; code/log/metadata corroboration, not cryptographic execution attestation',
            scaler='use original corresponding model source scaler unchanged; verify fit keys and pooled source cache stats; never refit using validation RX',
            inference='CPU eval/inference_mode, float32 model/maps, float64 softmax of original float32 logits, batch64, threads2, four independent jobs',
            samples='original G1 2000 rows per source RX, same rows/labels for Full and Strict; 200 per TX/RX',
            no_best_checkpoint_or_old_predictions=True),
        E1=dict(hard='P(Full top1 correct OR Strict top1 correct); an Accuracy ceiling only for hard selection',
            soft='per-packet scalar probability convex combination feasibility by analytic linear inequalities in lambda in[0,1]; not logits/feature/MoE ceiling',
            tie_rule='numpy/torch argmax chooses smallest index; > against lower class indices, >= against higher',
            unit='equal RX weighting; ten source RX, two shared expert pairs; no packet independence significance',
            gate='mean_RX(hard_accuracy - max(full_accuracy,strict_accuracy)) >=0.03',
            hard_gain_min=.03,soft_cannot_rescue_failed_gate=True,
            report='per_RX per_TX four correctness states, Full rescue/Full damage counts, hard/soft Accuracy oracles; no oracle Macro-F1 upper bound'),
        E2=dict(design='within each five unseen RX, four selector-training RX and one test RX; ten evaluations; never pool predictions across expert folds for selector fitting',
            selector='StandardScaler + LogisticRegression(C1,l2,lbfgs,max_iter1000,tol1e-8,random_state20261012); single fixed model, no tuning',
            features=['S_confidence','F_confidence','S_margin','F_margin','S_entropy','F_entropy','confidence_F_minus_S','margin_F_minus_S','entropy_F_minus_S','prediction_disagreement','probability_total_variation','Jensen_Shannon_divergence'],
            labels='fit only expert-disagreement cases with exactly one correct; target1=Full correct,0=Strict correct; no true-class margins in features',
            weights='equal total weight per training RX among informative packets; no class weighting',
            apply='if experts agree, Strict; otherwise Full iff logistic probability>=0.5; all thresholds fixed',
            fallback='if no informative cases use Strict; if only one training outcome constant training outcome; record degeneracy, no new model',
            baselines=['probability_average_50_50','temperature_calibrated_average','source_weight_selected_fusion','maximum_confidence_selection'],
            calibration='fit each expert temperature separately by equal-RX NLL on four training RX only; bounded log T in ln(.25)..ln(4), bounded scipy minimize_scalar xatol1e-6 max_iter100',
            fixed_weights=[i/20 for i in range(21)],weight_choice='maximize equal-RX training Macro-F1; ties closest to0.5 then smaller Full weight',
            baseline_choice='maximize equal-RX training Macro-F1 across four preregistered candidates; exact ties use listed order; freeze before test access',
            confidence_choice='Full only if raw max probability strictly larger, otherwise Strict',
            gate=dict(mean_macro_f1_gain_min=.01,each_RX_macro_f1_gain_min=0,worst_TX_recall_gain_min=.05,severe_RX_TX_recall_loss_max=.10,min_support_for_severe=50),
            worst_TX='min_y(mean_equal_RX(recall_RX_y)); also report each RX worst class',
            fit_only_if_E1_passed=True,fail_stops_this_candidate=True),
        prohibited=['new CNN training','Shen four dev RX for selector development','six historical locked RX','OSU target for design','HSTU/CrossAttn','restarting VIRP','changing checkpoints/features/thresholds after feedback'],
        evidence_boundary='internal staged mechanism screening on historically developed source data, not independent external blind confirmation; E1 label use conditions entry into E2',
        further_stage_authorized=False)
    target=ROOT/'configs/eisr/v1_registration.json';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:json.dump(reg,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    seal=dict(registration_sha256=sha(target),registered_utc=reg['registered_utc'],new_fixed_epoch_predictions_generated_before_freeze=False)
    with target.with_suffix('.seal.json').open('x') as f:json.dump(seal,f,indent=2);f.write('\n')
    print(json.dumps(seal))


if __name__=='__main__':main()
