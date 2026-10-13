#!/usr/bin/env python3
"""Independent arithmetic, provenance, pairing and archive verification; no inference."""
import csv, hashlib, json, zipfile
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/m0_signal/2026-10-13'
PUB=ROOT/'reports/m0_signal/2026-10-13'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def main():
    seal=json.loads((OUT/'pre_execution_seal.json').read_text())
    for p,d in seal['files'].items():assert sha(ROOT/p)==d,p
    reg=json.loads((ROOT/'configs/m0_signal/registration.json').read_text())
    packets=read('packet_diagnostics.csv');pred=read('frozen_predictions.csv');awgn=read('synthetic_awgn_deltas.csv')
    assert len(packets)==20000 and len(pred)==120000 and len(awgn)==8000
    keys={(r['receiver'],r['row']) for r in packets};assert len(keys)==20000
    count=defaultdict(int)
    for row in packets:
        assert row['receiver'] in reg['source_receivers'];count[row['receiver'],row['tx']]+=1
    assert len(count)==100 and set(count.values())=={200}
    slots={};scores=defaultdict(lambda:[0,0]);seed_groups=defaultdict(set)
    for row in pred:
        assert (row['receiver'],row['row']) in keys
        correct=int(row['prediction']==row['tx']);assert correct==int(row['correct'])
        k=(row['run'],row['receiver'],row['row']);slots.setdefault(k,{})[row['variant']]=correct
        seed_groups[row['receiver'],row['row']].add(row['seed'])
        scores[row['variant']][0]+=correct;scores[row['variant']][1]+=1
    assert len(slots)==40000 and all(len(v)==3 for v in slots.values())
    assert all(len(v)==2 for v in seed_groups.values())
    with (PUB/'control_summary.csv').open() as f:summary=list(csv.DictReader(f))
    for row in summary:
        variant=row['variant'];correct,n=scores[variant];assert n==40000
        assert abs(correct/n-float(row['accuracy']))<1e-12
        harm=sum(v['identity']==1 and v[variant]==0 for v in slots.values())
        gain=sum(v['identity']==0 and v[variant]==1 for v in slots.values())
        assert harm==int(row['correct_to_wrong']) and gain==int(row['wrong_to_correct'])
        assert scores[variant][0]==scores['identity'][0]+gain-harm
    trials={(r['receiver'],r['row'],r['added_snr_db'],r['replicate']) for r in awgn}
    assert len(trials)==8000
    counts=defaultdict(int)
    for row in awgn:counts[row['receiver'],row['tx'],row['added_snr_db'],row['replicate']]+=1
    assert len(counts)==400 and set(counts.values())=={20}
    entries=json.loads((PUB/'evidence_members.json').read_text())
    with zipfile.ZipFile(PUB/'evidence.zip') as z:
        assert z.testzip() is None
        assert sorted(z.namelist())==sorted(e['member'] for e in entries)
        for e in entries:
            body=z.read(e['member']);assert len(body)==e['bytes'] and hashlib.sha256(body).hexdigest()==e['sha256']
            assert not Path(e['member']).suffix in ['.pt','.h5','.npz','.npy','.pkl','.pdf']
    numeric=read('numerical_controls.csv')
    report=dict(passed=True,source_cells=100,source_packets=20000,paired_model_packet_slots=40000,
        derived_predictions=120000,awgn_trials=8000,archive_members=len(entries),archive_sha256=sha(PUB/'evidence.zip'),
        sealed_inputs_code_unchanged=True,new_inference=False,new_training=False,
        epsilon_max_abs=max(float(r['epsilon_max_abs']) for r in numeric),
        complex64_max_abs=max(float(r['float32_max_abs']) for r in numeric))
    (PUB/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
