#!/usr/bin/env python3
"""Audit only the six permitted drift sources, without training or packet pairing across RX."""
import csv, hashlib, json, sys
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_stage1.data import ShenDevelopmentPackets
from shen_stage1.frequency import periodic_phase_offset
from shen_p2a.cache import write_json,metadata_array
from report_shen_p2a import check_run,write_csv

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def run(out):
    out.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/'configs/shen_evidence/protocol.candidate.json'
    p=json.loads(protocol.read_text());freq=np.fft.fftfreq(8192,d=1e-6)
    bands={'ib':np.abs(freq)<=62500,'near':(np.abs(freq)>62500)&(np.abs(freq)<=125000),
           'mid':(np.abs(freq)>125000)&(np.abs(freq)<=250000),'far':np.abs(freq)>250000,
           'neg101':(freq>=-109375)&(freq<-93750),'pos101':(freq>93750)&(freq<=109375)}
    records=[];headers=[];cohorts={};spectra={}
    for rx in ('rtl_6','n210_1'):
        for role in ('train','same_day_diagnostic','validation'):
            ds=ShenDevelopmentPackets(ROOT,protocol,rx,role,True,np.float64)
            source=next(x for x in p['files'] if x['receiver']==rx and x['role']==role)
            actual=sha(ds.path)
            assert actual==source['sha256'],'Source changed'
            with h5py.File(ds.path,'r') as f:
                cfo=metadata_array(f,'CFO',len(ds));snr=metadata_array(f,'SNR',len(ds))
                headers.append(dict(receiver=rx,role=role,path=source['dataset_relative_path'],sha256=actual,
                    attributes={k:str(v) for k,v in f.attrs.items()},datasets={k:{'shape':list(f[k].shape),'dtype':str(f[k].dtype),'attributes':{a:str(b) for a,b in f[k].attrs.items()}} for k in f.keys()}))
            psd=np.lib.format.open_memmap(out/f'{rx}_{role}_psd128.npy',mode='w+',dtype='float32',shape=(len(ds),128))
            rows_for_source=[]
            for start in range(0,len(ds),128):
                batch=[ds[i] for i in range(start,min(start+128,len(ds)))]
                iq=np.stack([x[0][0]+1j*x[0][1] for x in batch]);assert np.isfinite(iq).all()
                power=np.abs(iq)**2; fftpower=np.abs(np.fft.fft(iq,axis=-1))**2
                fractions=fftpower/fftpower.sum(1,keepdims=True)
                binned=np.fft.fftshift(fractions,axes=-1).reshape(len(iq),128,64).sum(-1)
                psd[start:start+len(iq)]=binned
                phase=periodic_phase_offset(iq,1024,1e6)
                for j,(_,label,row) in enumerate(batch):
                    components=np.concatenate((iq[j].real,iq[j].imag));maximum=np.abs(components).max()
                    r=dict(receiver=rx,role=role,row=row,raw_label=label+31,raw_rms=float(np.sqrt(power[j].mean())),
                        peak_to_rms=float(np.sqrt(power[j].max()/power[j].mean())),zero_fraction=float((components==0).mean()),
                        repeated_extreme_fraction_proxy=float((np.abs(components)==maximum).mean()),
                        first256_energy_ratio=float(power[j,:256].mean()/power[j].mean()),last256_energy_ratio=float(power[j,-256:].mean()/power[j].mean()),
                        max_adjacent_power_change_ratio=float(np.abs(np.diff(power[j])).max()/power[j].mean()),
                        far_psd_q20_proxy=float(np.quantile(fftpower[j,bands['far']],.2)/fftpower[j].mean()),
                        spectral_centroid_assumed_hz=float((fractions[j]*freq).sum()),
                        spectral_peak_assumed_hz=float(freq[fftpower[j].argmax()]),
                        periodic_offset_assumed_hz=float(phase['all_pairs_offset_hz_under_assumed_fs'][j]),periodic_coherence=float(phase['coherence'][j]),
                        CFO_raw_unknown_units=float(cfo[row]),SNR_raw_unknown_units=float(snr[row]),
                        **{k+'_energy_fraction':float(fractions[j,v].sum()) for k,v in bands.items()})
                    rows_for_source.append(r)
            psd.flush();records.extend(rows_for_source)
            for tx in range(31,41):
                ix=ds.labels==tx-31;cohorts[rx,role,tx]=[r for r in rows_for_source if r['raw_label']==tx]
                spectra[rx,role,tx]=np.asarray(psd[ix]).copy()
            ds.close();print(rx,role,'audited',len(rows_for_source),flush=True)
    write_json(out/'hdf5_headers_and_hashes.json',headers);write_csv(out/'per_packet_raw_audit.csv',records)
    fields=[k for k in records[0] if k not in ('receiver','role','row','raw_label')]
    summaries=[];drifts=[];separability=[]
    for key,rows in cohorts.items():
        for field in fields:
            x=np.asarray([r[field] for r in rows]);finite=x[np.isfinite(x)]
            q=dict(zip(('q05','median','q95'),map(float,np.quantile(finite,[.05,.5,.95])))) if len(finite) else dict(q05=None,median=None,q95=None)
            summaries.append(dict(receiver=key[0],role=key[1],raw_label=key[2],feature=field,packets=len(x),nonfinite=int((~np.isfinite(x)).sum()),**q))
    for rx in ('rtl_6','n210_1'):
        for tx in range(31,41):
            for field in fields:
                a=np.asarray([r[field] for r in cohorts[rx,'train',tx]]);b=np.asarray([r[field] for r in cohorts[rx,'validation',tx]])
                a=a[np.isfinite(a)];b=b[np.isfinite(b)]
                if len(a) and len(b):drifts.append(dict(receiver=rx,raw_label=tx,feature=field,wasserstein_train_sd=float(wasserstein_distance(a,b)/max(a.std(),1e-12)),median_delta=float(np.median(b)-np.median(a))))
        for role in ('train','same_day_diagnostic','validation'):
            for band in bands:
                groups=[np.array([r[band+'_energy_fraction'] for r in cohorts[rx,role,tx]]) for tx in range(31,41)]
                separability.append(dict(receiver=rx,role=role,band=band,between_class_mean_variance_over_within=float(np.var([x.mean() for x in groups])/max(np.mean([x.var() for x in groups]),1e-12))))
    write_csv(out/'per_device_distribution_quantiles.csv',summaries);write_csv(out/'train_to_day2_distribution_shift.csv',drifts);write_csv(out/'band_class_separability.csv',separability)
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    x=(np.arange(128)+.5)*1e6/128-500000
    for i,rx in enumerate(('rtl_6','n210_1')):
        for j,tx in enumerate((32,33,34)):
            ax=axes[i,j]
            for role,color in zip(('train','same_day_diagnostic','validation'),('C0','C2','C1')):
                data=10*np.log10(np.maximum(spectra[rx,role,tx],1e-20));q=np.quantile(data,[.05,.5,.95],axis=0)
                ax.plot(x/1000,q[1],label=role,color=color);ax.fill_between(x/1000,q[0],q[2],alpha=.12,color=color)
            ax.axvline(-101.5625,c='k',ls=':');ax.axvline(101.5625,c='k',ls=':');ax.set_xlim(-200,200);ax.set_title(f'{rx} TX{tx}');ax.set_xlabel('Assumed kHz');ax.set_ylabel('Relative FFT power dB/bin')
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'TX32_33_34_spectrum_distributions.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    names=list(bands)
    for i,rx in enumerate(('rtl_6','n210_1')):
        matrix=[[next(r['wasserstein_train_sd'] for r in drifts if r['receiver']==rx and r['raw_label']==tx and r['feature']==band+'_energy_fraction') for band in names] for tx in range(31,41)]
        im=axes[i,0].imshow(matrix,aspect='auto');fig.colorbar(im,ax=axes[i,0]);axes[i,0].set_xticks(range(6),names);axes[i,0].set_yticks(range(10),range(31,41));axes[i,0].set_title(rx+' Train→Day2 Wasserstein / Train SD')
        for role in ('train','same_day_diagnostic','validation'):
            axes[i,1].plot(names,[next(r['between_class_mean_variance_over_within'] for r in separability if r['receiver']==rx and r['role']==role and r['band']==b) for b in names],marker='o',label=role)
        axes[i,1].set_yscale('log');axes[i,1].set_title(rx+' energy-fraction class separation');axes[i,1].legend()
    fig.tight_layout();fig.savefig(out/'all_TX_band_shift_and_separability.png',dpi=180);plt.close(fig)
    primary=ROOT/'outputs/shen_p2a/p2a_preregistered_v2';a7=ROOT/'outputs/shen_p2a/p2a_a7_followup_v2'
    fig,axes=plt.subplots(2,3,figsize=(13,8));error_rows=[]
    for i,rx in enumerate(('rtl_6','n210_1')):
        cms=[]
        for pipeline,method in ((primary,'spec_light_train_scalar'),(a7,'spec_strict_ib')):
            choice=next(c for c in json.loads((pipeline/'selection.frozen.json').read_text())['choices'] if c['receiver']==rx and c['method']==method)
            paths=[Path(choice['screening_run'])]+[d for d in (pipeline/'confirmation').iterdir() if d.is_dir() and json.loads((d/'config.json').read_text())['receiver']==rx and json.loads((d/'config.json').read_text())['method']==method]
            assert len(paths)==3
            cm=np.zeros((10,10),dtype=int)
            for d in paths:
                cfg,m,rows,y,pred=check_run(d);cm+=np.bincount(y*10+pred,minlength=100).reshape(10,10)
                for tx in (32,33,34):
                    for dest in range(31,41):error_rows.append(dict(receiver=rx,method=method,seed=cfg['seed'],raw_label=tx,predicted_raw_label=dest,packets=int(((y==tx-31)&(pred==dest-31)).sum())))
            cms.append(cm)
        for j,cm in enumerate((cms[0],cms[1],cms[1]-cms[0])):
            im=axes[i,j].imshow(cm,cmap='coolwarm' if j==2 else 'Blues',vmin=-600 if j==2 else 0,vmax=600);fig.colorbar(im,ax=axes[i,j]);axes[i,j].set_title(rx+' '+('Full','Strict IB','Strict−Full')[j]);axes[i,j].set_xticks(range(10),range(31,41),rotation=90);axes[i,j].set_yticks(range(10),range(31,41));axes[i,j].set_xlabel('Predicted TX');axes[i,j].set_ylabel('True TX')
    fig.tight_layout();fig.savefig(out/'Full_vs_StrictIB_confusion_difference.png',dpi=180);plt.close(fig);write_csv(out/'TX32_33_34_error_destinations.csv',error_rows)
    write_json(out/'status.json',dict(status='completed',raw_packets_audited=len(records),source_files=6,source_hashes_match=True,locked_files_opened=False,physical_metadata_locally_verified=False,cross_receiver_packet_pairing=False))

if __name__=='__main__':run(Path(sys.argv[1]))
