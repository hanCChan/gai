#!/usr/bin/env python3
"""Post-diagnosis symmetric strict-IB STFT follow-up, registered before training."""
import json,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import build_cache
if __name__=='__main__':
    out=ROOT/'configs/shen_p2a_a7';out.mkdir(exist_ok=False);jobs=[]
    for i,rx in enumerate(('n210_1','rtl_6')):
        for j,repr in enumerate(('spec','cis')):
            for lr in (.001,.0003):
                source=ROOT/f'configs/shen_p2a/A1_{rx}_{repr}_light_train_scalar_lr{lr:g}.json';cfg=json.loads(source.read_text())
                cfg.update(strict_ib_stft=True,comparison_group='A7_strict_IB_STFT_followup',method=f'{repr}_strict_ib',post_diagnosis_followup=True)
                name=f'A7_{rx}_{repr}_strict_ib_lr{lr:g}';path=out/f'{name}.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
                jobs.append({'stage':'P2-A','run_id':name,'gpu':2*i+j,'config':str(path),'receiver':rx,'family':'representation','method':cfg['method'],'comparison_group':cfg['comparison_group']})
    (out/'matrix.json').write_text(json.dumps(jobs,indent=2)+'\n')
    (out/'registration.json').write_text(json.dumps({'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary52_matrix_unchanged':True,'inference':'exploratory follow-up, motivated by observed RTL34 changes near -101.6kHz; not original preregistration',
        'original_IB_before_STFT':True,'map_crop_includes_generated_window_leakage_but_no_original_OOB_information':True,'same_rank16_encoder_budget_and_two_LR_as_full_STFT_A1':True,'screening_runs':8,'three_seed_additional_runs':8,'seed_choice':'select each receiver/representation on seed20261012 entire Day2 Macro-F1, tie .001; freeze and confirm seeds20261013/14','P2_B_or_P3_start':False},indent=2)+'\n')
    for rx in ('n210_1','rtl_6'):
        for role in ('train','validation'):print(build_cache(ROOT,ROOT/'configs/shen_evidence/protocol.candidate.json',rx,role,strict_ib=True),flush=True)
