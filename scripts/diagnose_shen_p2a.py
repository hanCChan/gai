#!/usr/bin/env python3
"""Development-only descriptive diagnostics; training-fitted clusters and no causal claims."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import cache_dir,write_json
from shen_evidence.models import AuthorCISCNN
from shen_stage1.data import ShenDevelopmentPackets

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def describe(root,out):
    stats=[];drifts=[];clusters=[];profiles=[];torch.set_num_threads(2)
    for rx in ('n210_1','rtl_6'):
        all_data={}
        for role in ('train','same_day_diagnostic','validation'):
            folder=cache_dir(root,rx,role);label=np.load(folder/'labels.npy');v=np.load(folder/'diagnostic_vectors.npy')
            diag=rows(folder/'packet_diagnostics.csv');all_data[role]=(label,v,diag)
            for c in range(10):
                selected=[r for r in diag if int(r['label'])==c]
                for metric in selected[0]:
                    if metric in ('receiver','role','row','label','raw_label'):continue
                    values=np.array([float(r[metric]) for r in selected if r[metric]!=''])
                    if values.size:stats.append({'receiver':rx,'role':role,'raw_label':c+31,'metric':metric,'n':len(values),'mean':float(values.mean()),'std':float(values.std()),'p05':float(np.quantile(values,.05)),'median':float(np.median(values)),'p95':float(np.quantile(values,.95))})
                for repr in ('spec','cis'):
                    a=np.load(folder/f'{repr}.npy',mmap_mode='r')[label==c,0].mean(0)
                    if rx=='rtl_6' and c in (0,3):np.save(out/f'{rx}_{role}_device{c+31}_{repr}_mean.npy',a)
                    for k,power in enumerate(a.mean(1)):profiles.append({'receiver':rx,'role':role,'raw_label':c+31,'representation':repr,'frequency_bin':k,'frequency_hz_assumed':float((k+38-64)*1e6/128),'mean_log_power_or_ratio':float(power)})
        for repr,sl in [('spec',slice(0,104)),('cis',slice(104,208))]:
            labels,train,_=all_data['train'];scaler=StandardScaler().fit(train[:,sl]);ztrain=scaler.transform(train[:,sl])
            centers=np.array([ztrain[labels==c].mean(0) for c in range(10)])
            within=np.array([np.mean(np.linalg.norm(ztrain[labels==c]-centers[c],axis=1)) for c in range(10)])
            km=KMeans(n_clusters=10,n_init=10,random_state=20261012).fit(ztrain)
            for role,(yl,v,_) in all_data.items():
                z=scaler.transform(v[:,sl]);cluster=km.predict(z)
                for c in range(10):
                    center=z[yl==c].mean(0);dist=np.linalg.norm(centers-center,axis=1)
                    drifts.append({'receiver':rx,'representation':repr,'role':role,'raw_label':c+31,'nearest_train_centroid_device':int(dist.argmin()+31),'own_centroid_distance':float(dist[c]),'device31_centroid_distance':float(dist[0]),'own_distance_over_train_within':float(dist[c]/max(within[c],1e-12))})
                    for k in range(10):clusters.append({'receiver':rx,'representation':repr,'role':role,'raw_label':c+31,'train_fitted_cluster':k,'count':int(((yl==c)&(cluster==k)).sum())})
            np.savez(out/f'{rx}_{repr}_train_fitted_cluster.npz',mean=scaler.mean_,scale=scaler.scale_,cluster_centers=km.cluster_centers_)
    write_csv(out/'per_device_feature_quality.csv',stats);write_csv(out/'centroid_shifts.csv',drifts);write_csv(out/'train_fitted_error_clusters.csv',clusters);write_csv(out/'frequency_profiles.csv',profiles)

@torch.no_grad()
def p1_recheck(root,out):
    info=json.loads((root/'configs/shen_p2a/p1_artifacts.json').read_text());result=[];prediction_rows=[]
    for choice in info['selection']['choices']:
        if choice['kind']!='cis' or choice['receiver']!='rtl_6':continue
        runs=[Path(choice['screening_run'])]+[Path(info['confirmation_root'])/f'E04_rtl_6_seed{s}' for s in (20261013,20261014)]
        for run in runs:
            checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False);model=AuthorCISCNN();model.load_state_dict(checkpoint['model']);model.eval()
            seed=checkpoint['config']['seed']
            for role in ('train','same_day_diagnostic','validation'):
                folder=cache_dir(root,'rtl_6',role);x=np.load(folder/'cis.npy',mmap_mode='r');label=np.load(folder/'labels.npy');pred=[]
                for start in range(0,len(label),64):pred.extend(model(torch.from_numpy(np.array(x[start:start+64],copy=True))).argmax(1).tolist())
                pred=np.array(pred);cm=np.bincount(label*10+pred,minlength=100).reshape(10,10)
                np.savetxt(out/f'P1_CIS_rtl6_seed{seed}_{role}_confusion.csv',cm,delimiter=',',fmt='%d')
                if role=='validation':
                    saved=rows(run/'predictions.csv');assert len(saved)==len(label)
                    expected=np.array([int(r['prediction']) for r in sorted(saved,key=lambda r:int(r['row']))]);np.testing.assert_array_equal(pred,expected)
                    diag=rows(folder/'packet_diagnostics.csv')
                    for i in range(len(label)):prediction_rows.append({'seed':seed,'row':i,'raw_label':int(label[i]+31),'raw_prediction':int(pred[i]+31),'correct':bool(label[i]==pred[i]),**{k:v for k,v in diag[i].items() if k not in ('receiver','role','row','label','raw_label')}})
                result.append({'seed':seed,'role':role,'accuracy':float((pred==label).mean()),'device34_correct':int(cm[3,3]),'device34_count':int(cm[3].sum()),'device34_to31':int(cm[3,0]),'Day2_saved_predictions_exact':role=='validation'})
                print(json.dumps(result[-1]),flush=True)
    write_csv(out/'P1_CIS_rechecked.csv',result);write_csv(out/'P1_Day2_errors_with_quality.csv',prediction_rows)
    # Separate original Full/IB training history; P1 gradients were not recorded.
    history=[]
    for choice in info['selection']['choices']:
        if choice['receiver']!='rtl_6' or choice['kind'] not in ('full_iq','strict_ib'):continue
        runs=[Path(choice['screening_run'])]+[Path(info['confirmation_root'])/f'{choice["config"]["experiment_id"]}_rtl_6_seed{s}' for s in (20261013,20261014)]
        for run in runs:
            m=json.loads((run/'metrics.json').read_text());cfg=json.loads((run/'config.json').read_text());curve=rows(run/'history.csv')
            history.append({'method':choice['kind'],'seed':cfg['seed'],'lr':cfg['lr'],'best_epoch':m['best_epoch'],'epochs_completed':m['epochs_completed'],'accuracy':m['packet_accuracy'],'macro_f1':m['macro_f1'],'final_train_accuracy':curve[-1]['train_accuracy'],'gradients_available':False})
    write_csv(out/'P1_RTL_optimization_recheck.csv',history)

def spectrum_audit(root,out):
    records=[];separability=[];protocol=root/'configs/shen_evidence/protocol.candidate.json';n=8192;f=np.fft.fftshift(np.fft.fftfreq(n,d=1e-6))
    bands={'ib':np.abs(f)<=62500.,'near':(np.abs(f)>75000.)&(np.abs(f)<=125000.),'mid':(np.abs(f)>125000.)&(np.abs(f)<=250000.),'far':np.abs(f)>250000.,'oob':np.abs(f)>75000.}
    for rx in ('n210_1','rtl_6'):
        features={}
        for role in ('train','same_day_diagnostic','validation'):
            ds=ShenDevelopmentPackets(root,protocol,rx,role,signal_dtype=np.float64);features[role]={b:[] for b in bands}
            for start in range(0,len(ds),128):
                batch=[ds[i] for i in range(start,min(len(ds),start+128))];x=np.stack([r[0][0]+1j*r[0][1] for r in batch]);p=np.abs(np.fft.fftshift(np.fft.fft(x,norm='ortho'),axes=-1))**2
                hann=np.hanning(n);ph=np.abs(np.fft.fftshift(np.fft.fft(x*hann,norm='ortho'),axes=-1))**2
                for b,mask in bands.items():
                    # 32 within-band ordered bins, descriptive log-energy profiles only.
                    groups=np.array_split(np.flatnonzero(mask),32);v=np.stack([np.log10(p[:,g].mean(1)+1e-30) for g in groups],axis=1);features[role][b].append(v)
                for j,(_,label,row) in enumerate(batch):
                    for b,mask in bands.items():records.append({'receiver':rx,'role':role,'row':row,'raw_label':label+31,'band':b,'rect_energy_fraction':float(p[j,mask].sum()/p[j].sum()),'raw_hann_before_fft_energy_fraction_NONSTRICT':float(ph[j,mask].sum()/ph[j].sum())})
            for b in bands:features[role][b]=np.concatenate(features[role][b])
            ds.close()
        for b in bands:
            train=features['train'][b];train_labels=np.load(cache_dir(root,rx,'train')/'labels.npy');sc=StandardScaler().fit(train);z=sc.transform(train);centers=np.array([z[train_labels==c].mean(0) for c in range(10)])
            for role in features:
                labels=np.load(cache_dir(root,rx,role)/'labels.npy');r=sc.transform(features[role][b])
                for c in range(10):
                    local=r[labels==c];center=local.mean(0);within=np.linalg.norm(z[train_labels==c]-centers[c],axis=1).mean();other=np.delete(centers,c,axis=0)
                    separability.append({'receiver':rx,'band':b,'role':role,'raw_label':c+31,'own_shift_over_train_within':float(np.linalg.norm(center-centers[c])/max(within,1e-12)),'nearest_other_train_distance_over_within':float(np.linalg.norm(other-center,axis=1).min()/max(within,1e-12))})
    write_csv(out/'packet_ring_energy_window_probe.csv',records);write_csv(out/'ring_descriptive_separability.csv',separability)
    # Pure in-band off-grid tones demonstrate finite-record leakage; this is a probe, not its attribution in real data.
    synthetic=[];t=np.arange(n)/1e6
    for freq in (30012.3,62012.3):
        x=np.exp(2j*np.pi*freq*t)
        for window in ('rect','hann'):
            y=x if window=='rect' else x*np.hanning(n);p=np.abs(np.fft.fftshift(np.fft.fft(y)))**2
            for b,mask in bands.items():synthetic.append({'tone_hz':freq,'window':window,'band':b,'energy_fraction':float(p[mask].sum()/p.sum())})
    write_csv(out/'synthetic_inband_tone_leakage.csv',synthetic)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',{'status':'running'})
    try:
        describe(ROOT,out);spectrum_audit(ROOT,out);p1_recheck(ROOT,out)
        write_json(out/'status.json',{'status':'completed','final_signal_files_opened':0,'train_fitted_clusters_only':True,'causality_established':False,'unknown_CFO_SNR_units':True,'denominator_threshold_column_mapping':{'1e6':1e-6,'1e4':1e-4,'1e2':1e-2}})
    except BaseException as exc:write_json(out/'status.json',{'status':'failed','error':repr(exc)});raise
