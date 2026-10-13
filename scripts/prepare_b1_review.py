#!/usr/bin/env python3
"""Generate a concrete B1 review matrix; this script never trains anything."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def build():
    r=json.loads((ROOT/'configs/d05_b0/registration.json').read_text())
    methods=['ce','group_dro_rx_tx','grl','supcon'];pilots=[];formal=[];splits=[]
    # Source-pool fixed choices, not chosen from B0 performance.
    rng=np.random.default_rng(20261015)
    for fold in range(2):
        source=[f'shen_{rx}_train' for rx in r['folds'][1-fold]];outer=[f'shen_{rx}_train' for rx in r['folds'][fold]]
        index=int(rng.integers(0,5));internal_val=[source[index]];internal_train=[key for key in source if key not in internal_val]
        splits.append(dict(fold=fold,inner_train=internal_train,inner_validation=internal_val,outer_evaluation=outer))
        for method in methods:
            common=dict(method=method,fold=fold,representation='strict_cis',backbone='FactorizedAuthorCNN(rank=16)',base_inference_parameters=658314,optimizer='Adam',weight_decay=0.0,clip_norm=5.0,augmentation='none',source_data_protocol='frozen G1 200 packets per RXxTX',scaler='fit only current fitting RX',sampler='group-balanced without replacement, 2 samples per RXxTX group per batch, all packets once/epoch',loss={'ce':'CE','group_dro_rx_tx':'multiplicative-weight Group DRO on RXxTX group means; eta=0.01, q initialized uniform','grl':'TX CE + RX CE, GRL coefficient0.1; RX head512->128->number_of_fitting_RX','supcon':'CE + 0.1 standard supervised contrastive loss on normalized512-D feature, tau0.1; all same-TX positives, no custom pair weights'}[method])
            for lr in (.0001,.0003):
                pilots.append(dict(common,run_id=f'B1_search_CV{fold}_{method}_lr{lr}',seed=20261015,epochs=10,batch_size=80,train_keys=internal_train,validation_keys=internal_val,outer_keys_never_read=outer,lr=lr,checkpoint='fixed10 only',updates=1000,groups=40,sampler_packets=8000,rx_classifier_classes=4 if method=='grl' else None))
            for seed in (20261013,20261014):
                formal.append(dict(common,run_id=f'B1_final_CV{fold}_{method}_seed{seed}',seed=seed,epochs=20,batch_size=100,train_keys=source,evaluation_keys=outer,lr='frozen_from_same-method_same-fold_internal_search',checkpoint='fixed20 only',updates=2000,groups=50,sampler_packets=10000,rx_classifier_classes=5 if method=='grl' else None))
    assert len(pilots)==len(formal)==16
    for job in pilots:
        assert not set(job['train_keys'])&set(job['validation_keys'])
        assert not (set(job['train_keys'])|set(job['validation_keys']))&set(job['outer_keys_never_read'])
    for job in formal:assert not set(job['train_keys'])&set(job['evaluation_keys'])
    return dict(status='REVIEW_ONLY_NOT_AUTHORIZED',training_authorized=False,methods=methods,search_jobs=pilots,formal_jobs=formal,source_splits=splits,search_selection=dict(metric='internal validation macro_F1 at fixed10, max; exact tie -> lower learning rate',loss_parameters='fixed as above, no additional loss-weight search',seed='one fixed search seed, not selected from results',freeze_before_final_fit=True),new_training_jobs_requested=32,formal_model_count=16,total_model_epochs=480,requested_reserved_gpu_hour_cap=2.0,estimated_gpu_hours=[.7,2.0],calibration='first already-counted pilot; if full matrix exceeds2hours then stop and revise budget, do not silently shorten matrix',next_layer_arcface='important pending comparator; excluded from first4-method matrix, needs separate registration/budget',b2_started=False,new_algorithm_implemented=False,current_implementation='configuration generation and split/cost validation only; full four-objective training runner must be prepared and tested in the separately authorized B1 execution before fitting')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'configs/d05_b0/b1_review_matrix.json');a=p.parse_args()
    if a.output.exists():raise ValueError('Do not overwrite a review matrix')
    m=build();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({k:v for k,v in m.items() if k not in ('search_jobs','formal_jobs','source_splits')}))
if __name__=='__main__':main()
