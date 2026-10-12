#!/usr/bin/env python3
"""Envelope/peak probes in stored units; these do not establish physical ADC rails."""
import sys,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from shen_stage1.data import ShenDevelopmentPackets
from report_shen_p2a import write_csv
if __name__=='__main__':
    out=ROOT/'outputs/shen_p2a/diagnostics_v1';target=out/'packet_waveform_quality.csv'
    if target.exists():raise FileExistsError('Preserve quality probes')
    records=[];summary=[]
    for rx in ('n210_1','rtl_6'):
        for role in ('train','same_day_diagnostic','validation'):
            ds=ShenDevelopmentPackets(ROOT,ROOT/'configs/shen_evidence/protocol.candidate.json',rx,role,cache_handles=True,signal_dtype=np.float64)
            for start in range(0,len(ds),128):
                batch=[ds[i] for i in range(start,min(start+128,len(ds)))];x=np.stack([r[0][0]+1j*r[0][1] for r in batch]);amplitude=np.abs(x);rms=np.sqrt((amplitude**2).mean(1));peak=amplitude.max(1)
                for j,(_,label,row) in enumerate(batch):records.append({'receiver':rx,'role':role,'row':row,'raw_label':label+31,'complex_crest_factor':float(peak[j]/rms[j]),'envelope_cv':float(amplitude[j].std()/amplitude[j].mean()),'iq_dc_over_rms':float(np.abs(x[j].mean())/rms[j]),'component_abs_max':float(max(np.abs(x[j].real).max(),np.abs(x[j].imag).max())),'fraction_within_point1percent_of_packet_peak':float((amplitude[j]>=.999*peak[j]).mean())})
            ds.close()
    write_csv(target,records)
    for rx in ('n210_1','rtl_6'):
        for role in ('train','same_day_diagnostic','validation'):
            for c in range(31,41):
                chosen=[r for r in records if r['receiver']==rx and r['role']==role and r['raw_label']==c]
                for metric in ('complex_crest_factor','envelope_cv','iq_dc_over_rms','component_abs_max','fraction_within_point1percent_of_packet_peak'):
                    x=np.array([r[metric] for r in chosen]);summary.append(dict(receiver=rx,role=role,raw_label=c,metric=metric,mean=float(x.mean()),p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95))))
    write_csv(out/'per_device_waveform_quality.csv',summary);print(json.dumps({'packets':len(records),'final_files_opened':0}))
