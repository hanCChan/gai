#!/usr/bin/env python3
"""Predeclare finite, symmetric candidate budgets before reading new outcomes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'configs/shen_p2a';OUT.mkdir(exist_ok=True)
protocol='configs/shen_evidence/protocol.candidate.json'
base={'stage':'P2-A','protocol':protocol,'protocol_sha256_expected':hashlib.sha256((ROOT/protocol).read_bytes()).hexdigest(),
      'seed':20261012,'batch_size':64,'workers':2,'max_epochs':30,'min_epochs':0,'patience':7,'min_delta':.0001,
      'early_stopping_enabled':True,'lr':.0003,'weight_decay':0.,'sample_rate_hz':1000000.,'bandwidth_hz':125000.,
      'spreading_factor_assumed':7,'guard_hz':12500.,'coordinate':'frequency','normalization':'independent','window':'rect'}
rows=[]
def register(name,gpu,**kw):
    cfg=dict(base,**kw);path=OUT/f'{name}.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
    rows.append({'run_id':name,'gpu':gpu,'config':str(path.relative_to(ROOT)),'stage':'P2-A','family':cfg['family'],
                 'receiver':cfg['receiver'],'method':cfg['method'],'comparison_group':cfg['comparison_group']})
for rx in ('n210_1','rtl_6'):
    for representation in ('spec','cis'):
        for lr in (.001,.0003):
            method=f'{representation}_light_train_scalar';name=f'A1_{rx}_{method}_lr{lr:g}'
            register(name,0,family='representation',receiver=rx,representation=representation,model='factorized',rank=16,
                     map_normalization='train_scalar',method=method,comparison_group='A1_matched_representation',lr=lr)
    for lr in (.001,.0003):
        register(f'A2_{rx}_cis_light_raw_lr{lr:g}',0,family='representation',receiver=rx,representation='cis',model='factorized',rank=16,
                 map_normalization='none',method='cis_light_raw',comparison_group='A2_capacity_raw_cis',lr=lr)
for model in ('factorized','author'):
    for lr in (.001,.0003):
        register(f'A2_rtl_6_cis_{model}_wd1e4_lr{lr:g}',0,family='representation',receiver='rtl_6',representation='cis',model=model,rank=16,
                 map_normalization='none',method=f'cis_{"light" if model=="factorized" else "author"}_raw_wd1e4',
                 comparison_group='A2_regularization_raw_cis',lr=lr,weight_decay=.0001)
for rx in ('n210_1','rtl_6'):
    for band in ('ib','near','mid','far','oob'):
        register(f'A4_{rx}_{band}_freq_independent',1,family='band',receiver=rx,band=band,
                 method=f'{band}_freq_independent',comparison_group='A4_frequency_rings')
for guard in (0.,25000.):
    register(f'A4_rtl_6_oob_guard{int(guard)}',1,family='band',receiver='rtl_6',band='oob',guard_hz=guard,
             method=f'oob_guard_{int(guard)}',comparison_group='A4_guard_sensitivity')
for norm,window in [('ib_relative','rect'),('independent','projection_hann'),('ib_relative','projection_hann')]:
    register(f'A4_rtl_6_oob_{norm}_{window}',1,family='band',receiver='rtl_6',band='oob',normalization=norm,window=window,
             method=f'oob_{norm}_{window}',comparison_group='A4_scale_window_sensitivity')
fusions=[('single_ib','none','single_ib'),('concat','ib_control','capacity_ib_ib_concat'),
         ('scalar_gate','ib_control','capacity_ib_ib_gate'),('concat','oob','ib_oob_concat'),('scalar_gate','oob','ib_oob_gate')]
for rx in ('n210_1','rtl_6'):
    for fusion,band,method in fusions:
        register(f'A5_{rx}_{method}',2,family='fusion',receiver=rx,fusion=fusion,aux_band=band,method=method,
                 comparison_group='A5_information_complementarity',coordinate='time')
for band in ('near','mid','far'):
    register(f'A4_rtl_6_ib_{band}_gate',2,family='fusion',receiver='rtl_6',fusion='scalar_gate',aux_band=band,
             method=f'ib_{band}_gate',comparison_group='A4_ring_complementarity',coordinate='time')
for band in ('full','ib'):
    for lr in (.001,.0003):
        for wd in (0.,.0001):
            register(f'A6_rtl_6_{band}_lr{lr:g}_wd{wd:g}',3,family='stability',receiver='rtl_6',band=band,
                     method=f'{band}_stability',comparison_group='A6_stability',coordinate='time',lr=lr,weight_decay=wd,
                     early_stopping_enabled=False,max_epochs=30)
(OUT/'screening_matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'registration.json').write_text(json.dumps({'frozen_baseline':'eb4ce2c0db783f9241777f8b94328500a1014113',
 'screening_seed':20261012,'confirmatory_seeds':[20261013,20261014],
 'gpu_candidate_counts':{str(g):sum(r['gpu']==g for r in rows) for g in range(4)},
 'A1':'Spec/CIS paired STFT geometry, same rank16 author encoder, train-only scalar normalization, equal LR budgets',
 'A2':'Raw CIS capacity control reuses both P1 author LR candidates; light raw uses the same two; RTL WD candidates symmetric',
 'A4':'IB/near/mid/far/allOOB frequency RI use same PacketCNN; original FFT projection precedes formal windowing',
 'A5':'P1-exact strict-IB main, same fixed lr=.0003 and budget; exact IB+IB architecture/parameter controls',
 'A6':'P1-exact Full/IB inputs and network; identical 2LR x 2WD budget; train all30, log counterfactual patience7 stopping',
 'early_stopping_changes_are_training_policy_not_physical_information':True,
 'P2_B_or_P3_automatic_start':False,'final_signal_access':False},indent=2)+'\n')
print(json.dumps({'candidates':len(rows),'per_gpu':{g:sum(r['gpu']==g for r in rows) for g in range(4)}},indent=2))
