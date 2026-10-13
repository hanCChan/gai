#!/usr/bin/env python3
"""Registered source-only mechanism screen. No optimizer, training or WiSig loading."""
import csv, datetime, hashlib, json, math, sys, time
from pathlib import Path
import h5py
import numpy as np
from scipy import signal, stats
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from b1_baselines.model import FeatureCNN
from shen_stage1.representations import rms_normalize_complex

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()

def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)

def stft(z,dtype=np.complex128,epsilon=0.):
    x=rms_normalize_complex(z).astype(dtype)
    _,_,s=signal.stft(x,window='boxcar',nperseg=128,noverlap=64,nfft=128,
                     return_onesided=False,padded=False,boundary=None,axis=-1)
    return np.fft.fftshift(s,axes=-2)[:,38:90,:]+epsilon

def cis(s):
    return np.log10(np.abs(s[:,:,1:]/s[:,:,:-1])**2)

def strict(z):
    mask=np.abs(np.fft.fftfreq(8192,d=1e-6))<=62500
    return np.fft.ifft(np.fft.fft(z,axis=-1)*mask,axis=-1)

def residual_proxy(z):
    x=rms_normalize_complex(z);a=x[:,:-1024];b=x[:,1024:]
    cross=np.sum(b*a.conj(),axis=1)
    phase=np.exp(1j*np.angle(cross))
    coherence=np.abs(cross)/np.sqrt(np.sum(abs(a)**2,axis=1)*np.sum(abs(b)**2,axis=1))
    residual=(b-phase[:,None]*a)/np.sqrt(2)
    _,_,s=signal.stft(residual,window='boxcar',nperseg=128,noverlap=64,nfft=128,
                     return_onesided=False,padded=False,boundary=None,axis=-1)
    proxy=(abs(np.fft.fftshift(s,axes=-2)[:,38:90])**2).mean(axis=-1)
    return proxy,coherence,(abs(residual)**2).mean(axis=-1)

def circular(z):
    x=rms_normalize_complex(z);x=x-x.mean(axis=1,keepdims=True)
    c=(x*x).mean(axis=1)/(abs(x)**2).mean(axis=1)
    if np.any(abs(c)>1+1e-12): raise ValueError('Invalid circularity')
    beta=c/(1+np.sqrt(np.maximum(0,1-abs(c)**2)))
    return z-beta[:,None]*z.conj(),abs(c)

def controls():
    rng=np.random.default_rng(20261013);results={}
    s=rng.normal(size=(4,52,127))+1j*rng.normal(size=(4,52,127))
    h=np.exp(rng.normal(size=(4,52,1))+1j*rng.normal(size=(4,52,1)))
    error=float(abs(cis(s*h)-cis(s)).max())
    results['static_multiplicative_cancellation_max_abs']=error
    assert error<1e-12
    # Local high-SNR log-power delta-method check; not a claim of finite exact ratio variance.
    nu=1e-6;count=200000;variance=[]
    for den in [1.,.1]:
        e1=np.sqrt(nu/2)*(rng.normal(size=count)+1j*rng.normal(size=count))
        e2=np.sqrt(nu/2)*(rng.normal(size=count)+1j*rng.normal(size=count))
        v=np.log10(abs((1+e2)/(den+e1))**2)-np.log10(1/den**2)
        predicted=2*nu/(np.log(10)**2)*(1+1/den**2)
        measured=float(v.var());assert .96<measured/predicted<1.04
        variance.append(dict(den=den,predicted=predicted,measured=measured))
    results['independent_awgn_log_ratio_variance']=variance
    results['low_denominator_variance_ratio']=variance[1]['measured']/variance[0]['measured']
    # Constant quotient contains no TX information. It cannot improve identifiability.
    results['constant_ratio_negative_control']=float(abs(cis(np.ones((1,52,127)))).max())
    assert results['constant_ratio_negative_control']==0
    x=(rng.normal(size=(64,8192))+1j*rng.normal(size=(64,8192)))/np.sqrt(2)
    y=x+.15j*x.conj();fixed,_=circular(y)
    _,before=circular(y);_,after=circular(fixed)
    results['proper_awgn_mirror_positive_control']=dict(before=float(before.mean()),after=float(after.max()))
    assert after.max()<1e-12
    # Same received waveform: distortion could be solely TX or solely RX.
    results['tx_rx_nonidentifiability_exact']=bool(np.array_equal(x+.15j*x.conj(),x+.15j*x.conj()))
    # A repeated deterministic improper waveform violates the proper-input assumption.
    tone=np.tile(np.exp(2j*np.pi*np.arange(8192)/8192),(1,1));_,ct=circular(tone)
    results['balanced_complex_tone_circularity']=float(ct[0])
    results['passed']=True
    return results

def main():
    started=time.perf_counter();r=json.loads((ROOT/'configs/m0_signal/registration.json').read_text())
    out=ROOT/'outputs/m0_signal/2026-10-13';out.mkdir(parents=True,exist_ok=False)
    seal_files=['configs/m0_signal/registration.json','docs/m0_signal/EXECUTION_PLAN.md',
                'docs/m0_signal/LITERATURE_AND_HYPOTHESES.md','scripts/run_m0_m2_signal.py',
                'src/b1_baselines/model.py','src/shen_p2a/models.py','src/shen_stage1/representations.py']
    dump(out/'pre_execution_seal.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         files={p:sha(ROOT/p) for p in seal_files},before_signal_analysis=True))
    dump(out/'synthetic_controls.json',controls())
    hist=Path(r['history_root']);broot=Path(r['b1_root']);models=[];access=[]
    br=json.loads((broot/'configs/b1_baselines/registration.json').read_text())
    torch.set_num_threads(4);device=torch.device('cuda:0')
    assert torch.cuda.is_available()
    for job in br['matrix']['formal_jobs']:
        if job['method']!='ce': continue
        folder=broot/'outputs/b1_baselines/formal/runs'/job['run_id']
        cfg=json.loads((folder/'config.json').read_text())
        saved=json.loads((folder/'status.json').read_text())
        assert sha(folder/'final.pt')==saved['checkpoint_sha256']
        ck=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==20 and ck['optimizer_steps']==2000
        m=FeatureCNN().to(device);m.load_state_dict(ck['model']);m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        models.append(dict(job=job,folder=folder,cfg=cfg,model=m))
        access.append(dict(kind='frozen_B1_CE_checkpoint',path=str(folder/'final.pt'),sha256=saved['checkpoint_sha256']))
    assert len(models)==4
    packet=[];changes=[];predictions=[];numeric=[];identity=[];input_hashes=[]
    perturb_rng=np.random.default_rng(r['seed'])
    for item in r['source_files']:
        rx=item['receiver'];assert rx in r['source_receivers']
        assert item['relative']==rx+'_train.h5'
        raw=Path(r['raw_root'])/item['relative'];assert raw.resolve()==raw
        digest=sha(raw);assert digest==item['expected_sha256']
        cache=hist/'outputs/shen_p2a8/g1_cache_v1'/item['key']
        rows=np.load(cache/'rows.npy',allow_pickle=False);labels=np.load(cache/'labels.npy',allow_pickle=False)
        assert len(rows)==2000 and np.array_equal(np.bincount(labels,minlength=10),np.full(10,200))
        for name in ['rows.npy','labels.npy','strict_cis.npy','manifest.json']:
            p=cache/name;rel=str(p.relative_to(hist));expected=br['feature_assets'][rel]['sha256']
            assert sha(p)==expected
        old=np.load(cache/'strict_cis.npy',mmap_mode='r',allow_pickle=False)
        active=[m for m in models if item['key'] in m['job']['evaluation_keys']];assert len(active)==2
        input_hashes.append(dict(receiver=rx,raw_sha256=digest,raw_bytes=raw.stat().st_size,
                            rows_sha256=sha(cache/'rows.npy'),labels_sha256=sha(cache/'labels.npy')))
        access.append(dict(kind='source_raw_hdf5_read_only',path=str(raw),sha256=digest))
        rx_maps={};rx_scores={};rx_features={};rx_packet=[]
        with h5py.File(raw,'r') as hf:
            assert all(isinstance(hf.get(k,getlink=True),h5py.HardLink) for k in ['data','label'])
            assert hf['data'].shape[1]==16384
            yraw=np.asarray(hf['label'][:]).reshape(-1).astype(int)-31
            np.testing.assert_array_equal(yraw[rows],labels)
            for begin in range(0,2000,50):
                ids=rows[begin:begin+50];stored=np.asarray(hf['data'][ids],dtype=np.float64)
                z=stored[:,:8192]+1j*stored[:,8192:];zs=strict(z)
                s=stft(zs,epsilon=1e-12);base=cis(s)
                np.testing.assert_array_equal(base[:,None].astype(np.float32),old[begin:begin+len(ids)])
                proxy,coh,respower=residual_proxy(zs)
                d=abs(s[:,:,:-1])**2;n=abs(s[:,:,1:])**2
                nu=proxy[:,:,None]
                regularized=np.log10((n+nu)/(d+nu))
                fixed,mirror=circular(z);compensated=cis(stft(strict(fixed),epsilon=1e-12))
                variants={'identity':base,'repeat_proxy_regularization':regularized,'circularization_diagnostic':compensated}
                c0=cis(stft(zs));c32=cis(stft(zs,dtype=np.complex64,epsilon=1e-12))
                for j in range(len(ids)):
                    p=begin+j;tx=int(labels[p]);den=d[j];rel=den/max(float(den.mean()),1e-30)
                    halves=[]
                    for part in np.array_split(z[j],2):
                        _,v=circular(part[None]);halves.append(float(v[0]))
                    metric=dict(receiver=rx,tx=tx+31,row=int(ids[j]),position=p,
                        low_rel_1e4=float((rel<=1e-4).mean()),low_rel_1e2=float((rel<=.01).mean()),
                        proxy_low_reliability_fraction=float((den<=10*proxy[j,:,None]).mean()),
                        repeat_coherence=float(coh[j]),repeat_residual_power=float(respower[j]),
                        mirror_circularity=float(mirror[j]),mirror_half1=halves[0],mirror_half2=halves[1],
                        cis_extreme_10_fraction=float((abs(base[j])>10).mean()),
                        regularization_rms_delta=float(np.sqrt(((regularized[j]-base[j])**2).mean())),
                        circularization_rms_delta=float(np.sqrt(((compensated[j]-base[j])**2).mean())))
                    packet.append(metric);rx_packet.append(metric)
                numeric.append(dict(receiver=rx,begin=begin,epsilon_max_abs=float(abs(base-c0).max()),
                                    float32_max_abs=float(abs(base-c32).max()),all_finite=bool(np.isfinite(c32).all())))
                for variant,value in variants.items():
                    assert np.isfinite(value).all()
                    rx_maps.setdefault(variant,[]).append(value.astype(np.float32))
                    for model in active:
                        cfg=model['cfg'];tag=(variant,cfg['run_id'])
                        x=((value[:,None]-cfg['scaler']['mean'])/cfg['scaler']['std']).astype(np.float32)
                        with torch.inference_mode():
                            logits,features=model['model'].forward_with_features(torch.from_numpy(x).to(device))
                        rx_scores.setdefault(tag,[]).append(logits.cpu().numpy())
                        rx_features.setdefault(tag,[]).append(features.cpu().numpy())
                chosen=np.array([p for p in range(begin,begin+len(ids)) if np.sum(labels[:p]==labels[p])<20])-begin
                if len(chosen):
                    zz=z[chosen];clean=base[chosen]
                    for db in r['awgn_db']:
                        for rep in range(r['awgn_replicates']):
                            power=(abs(zz)**2).mean(axis=1,keepdims=True)/10**(db/10)
                            noise=np.sqrt(power/2)*(perturb_rng.normal(size=zz.shape)+1j*perturb_rng.normal(size=zz.shape))
                            changed=cis(stft(strict(zz+noise),epsilon=1e-12));delta=(changed-clean)**2
                            for jj,local in enumerate(chosen):
                                low=d[local]<=np.quantile(d[local],.01)
                                total=float(delta[jj].sum())
                                changes.append(dict(receiver=rx,tx=int(labels[begin+local])+31,row=int(ids[local]),
                                    added_snr_db=db,replicate=rep,rms_delta=float(np.sqrt(delta[jj].mean())),
                                    low_den_1pct_change_share=float(delta[jj][low].sum()/max(total,1e-30))))
        for model in active:
            job=model['job'];run=job['run_id'];reference=None;ref_feat=None
            with np.load(model['folder']/'predictions'/(rx+'.npz'),allow_pickle=False) as archived:
                np.testing.assert_array_equal(archived['rows'],rows);np.testing.assert_array_equal(archived['labels'],labels)
                archived_logits=archived['logits'].copy()
            for variant in variants:
                logits=np.concatenate(rx_scores[variant,run]);features=np.concatenate(rx_features[variant,run])
                if variant=='identity':
                    np.testing.assert_allclose(logits,archived_logits,rtol=1e-4,atol=1e-4)
                    reference=logits.argmax(1);ref_feat=features
                pred=logits.argmax(1);margin=logits[np.arange(2000),labels]-np.max(np.where(np.eye(10)[labels],-np.inf,logits),axis=1)
                centers=np.stack([features[labels==c].mean(0) for c in range(10)])
                within=float(np.mean((features-centers[labels])**2));between=float(np.mean((centers-centers.mean(0))**2))
                identity.append(dict(run=run,fold=job['fold'],seed=job['seed'],receiver=rx,variant=variant,
                    accuracy=float((pred==labels).mean()),macro_margin=float(margin.mean()),
                    correct_to_wrong=int(((reference==labels)&(pred!=labels)).sum()),
                    wrong_to_correct=int(((reference!=labels)&(pred==labels)).sum()),
                    within_tx_variance=within,between_tx_variance=between,separability_ratio=between/max(within,1e-30)))
                for p in range(2000):
                    predictions.append(dict(run=run,fold=job['fold'],seed=job['seed'],receiver=rx,
                        tx=int(labels[p])+31,row=int(rows[p]),position=p,variant=variant,
                        prediction=int(pred[p])+31,correct=int(pred[p]==labels[p]),margin=float(margin[p])))
        print('Completed source raw / diagnostics / frozen inference:',rx,flush=True)
    table(out/'packet_diagnostics.csv',packet);table(out/'synthetic_awgn_deltas.csv',changes)
    table(out/'frozen_predictions.csv',predictions);table(out/'frozen_RX_control_metrics.csv',identity)
    table(out/'numerical_controls.csv',numeric);table(out/'input_hashes.csv',input_hashes)
    dump(out/'access_log.json',access)
    dump(out/'execution.json',dict(status='completed',source_receivers=10,packets=20000,training_jobs=0,
        optimizer_steps=0,frozen_ce_models=4,unseen_source_predictions_per_variant=40000,
        variants=3,added_awgn_trials=len(changes),seconds=time.perf_counter()-started,
        device=str(device),gpu_name=torch.cuda.get_device_name(0),
        interpretation='noise proxies and circularity are observational; no natural RX-causal identification',
        WiSig_loaded=False,forbidden_inputs_opened=False))

if __name__=='__main__':main()
