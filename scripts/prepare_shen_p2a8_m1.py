#!/usr/bin/env python3
"""Freeze 42-cell M1 matrix; reuse 12 checked runs, build/verify new transforms."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a.cache import write_json
from shen_p2a8.features import build,feature_folder,removed_mask,project,scaler,CounterfactualMaps
from shen_p2a.representations import paired_maps
from shen_stage1.data import ShenDevelopmentPackets
from report_shen_p2a import check_run

if __name__=='__main__':
    if (ROOT/'docs/shen_p2a8/USER_SUPPLIED_GENERALIZATION_UPDATE.md').exists():
        raise PermissionError('Latest user plan supersedes the 30-notch-run preparation; execute G1 instead')
    out=ROOT/'configs/shen_p2a8';regfile=out/'registration.json'
    assert not regfile.exists(),'Registration is immutable'
    assert json.loads((ROOT/'outputs/shen_p2a8/m0_v1/status.json').read_text())['status']=='completed'
    centres=np.array([k*7812.5 for k in range(-24,25) if abs(k)>=10 and abs(k-13)>=3 and abs(k+13)>=3])
    centre=float(np.random.default_rng(20261012).choice(centres))
    reg=dict(stage='P2-A8/M1',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exploratory=True,
        physical_frequency_status='assumed nominal passband; paper settings known, local provenance unverified',
        notch=dict(width_hz=15625.,bins=128,negative_reference_hz=-101562.5,negative_low_hz=-109375.,negative_high_hz=-93750.,
            positive_rule='exact reflected negative FFT-bin set',random_seed=20261012,random_candidate_centres_hz=centres.tolist(),random_centre_hz=centre,random_low_hz=centre-7812.5,random_high_hz=centre+7812.5),
        seeds=[20261012,20261013,20261014],lr=.0003,weight_decay=0.,max_epochs=30,patience=7,
        common_scaler='equal pooled Full and Strict Day1 Train maps per RX, mean and second moment',
        selection='historical Day2 best Macro-F1 checkpoint and patience7, exploratory only',new_trainings=30,reused_trainings=12,
        P2_B_or_old_P3_started=False,M2_requires_mechanism_review=True)
    reg['mask_sha256']={v:hashlib.sha256(removed_mask(v,reg).tobytes()).hexdigest() for v in ('full','strict_ib','notch_negative','notch_positive','notch_random')}
    assert all(removed_mask(v,reg).sum()==128 for v in ('notch_negative','notch_positive','notch_random'))
    write_json(regfile,reg)
    jobs=[];reuse=[];checks=[];protocol=ROOT/'configs/shen_evidence/protocol.candidate.json'
    keys=('lr','weight_decay','max_epochs','batch_size','workers','patience','min_delta','early_stopping_enabled','model','rank','representation','family')
    for rx in ('rtl_6','n210_1'):
        base=json.loads((ROOT/f'configs/shen_p2a/A1_{rx}_spec_light_train_scalar_lr0.0003.json').read_text())
        for variant,pipeline,method in (('full','p2a_preregistered_v2','spec_light_train_scalar'),('strict_ib','p2a_a7_followup_v2','spec_strict_ib')):
            history=ROOT/'outputs/shen_p2a'/pipeline
            choice=next(c for c in json.loads((history/'selection.frozen.json').read_text())['choices'] if c['receiver']==rx and c['method']==method)
            paths=[Path(choice['screening_run'])]+[d for d in (history/'confirmation').iterdir() if d.is_dir() and json.loads((d/'config.json').read_text())['receiver']==rx and json.loads((d/'config.json').read_text())['method']==method]
            assert len(paths)==3
            for path in paths:
                cfg,metrics,*_=check_run(path)
                assert all(cfg[k]==base[k] for k in keys),'Existing run is not matched'
                reuse.append(dict(receiver=rx,feature_variant=variant,scaler_mode='independent',seed=cfg['seed'],run=str(path),config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),best_sha256=hashlib.sha256((path/'best.pt').read_bytes()).hexdigest()))
        for variant,mode in [('full','common'),('strict_ib','common'),('notch_negative','independent'),('notch_positive','independent'),('notch_random','independent')]:
            if variant.startswith('notch'):
                for role in ('train','validation'):build(ROOT,rx,role,variant,reg)
            for seed in reg['seeds']:
                name=f'M1_{rx}_{variant}_{mode}_seed{seed}'
                cfg=dict(base,mechanism_stage='M1',feature_variant=variant,scaler_mode=mode,seed=seed,method=f'{variant}_{mode}',comparison_group='M1_controlled_spectral_counterfactual',post_diagnosis_followup=True)
                cfg['registration_sha256']=hashlib.sha256(regfile.read_bytes()).hexdigest()
                write_json(out/f'{name}.json',cfg)
                jobs.append(dict(stage='P2-A',run_id=name,receiver=rx,family='representation',method=cfg['method'],config=str(out/f'{name}.json'),gpu=len(jobs)%4))
        ds=ShenDevelopmentPackets(ROOT,protocol,rx,'train',True,np.float64)
        iq=np.stack([ds[int(np.flatnonzero(ds.labels==c)[0])][0] for c in range(10)]);complex_iq=iq[:,0]+1j*iq[:,1];ds.close()
        for variant in ('full','strict_ib','notch_negative','notch_positive','notch_random'):
            for role in ('train','validation'):
                source=ShenDevelopmentPackets(ROOT,protocol,rx,role,True,np.float64)
                ix=[int(np.flatnonzero(source.labels==c)[0]) for c in range(10)]
                raw=np.stack([source[i][0] for i in ix]);z=raw[:,0]+1j*raw[:,1]
                expected=paired_maps(z if variant=='full' else project(z,variant,reg))[0]
                actual=np.load(feature_folder(ROOT,rx,role,variant)/'spec.npy',mmap_mode='r')[ix]
                np.testing.assert_array_equal(expected,actual);source.close()
            mask=removed_mask(variant,reg);z=project(complex_iq,variant,reg);fft=np.fft.fft(z,axis=-1)
            residual=float(np.max(np.abs(fft[:,mask]))/max(np.max(np.abs(fft)),1e-20)) if mask.any() else 0.
            assert residual<1e-12
            nuisance=np.random.default_rng(123).normal(size=complex_iq.shape)+1j*np.random.default_rng(124).normal(size=complex_iq.shape)
            perturb=np.fft.ifft(np.fft.fft(nuisance,axis=-1)*mask,axis=-1)
            np.testing.assert_allclose(project(complex_iq+perturb,variant,reg),z,atol=1e-12,rtol=1e-12)
            checks.append(dict(receiver=rx,variant=variant,ten_class_maps_exact=True,removed_bin_relative_residual=residual,excluded_information_invariance=True))
        a=scaler(ROOT,rx,'full','common');b=scaler(ROOT,rx,'strict_ib','common');assert a==b
    assert len(jobs)==30 and len(reuse)==12
    write_json(out/'matrix.json',jobs);write_json(out/'reuse_manifest.json',reuse)
    write_json(ROOT/'outputs/shen_p2a8/m1_preflight.json',dict(passed=True,checks=checks,reused_runs_exact_recipe_and_predictions_checked=12,new_runs=30))
    print(json.dumps(dict(new_runs=len(jobs),reused_runs=len(reuse),random_notch_centre_hz=centre,preflight=True)),flush=True)
