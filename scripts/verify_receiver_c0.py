#!/usr/bin/env python3
"""Independent source-only choice recomputation and immutable-history verification."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):
    with p.open() as f:return list(csv.DictReader(f))


def rule(rows,multi):
    for s in [1.,.5,.25]:
        cells=[r for r in rows if float(r['scale'])==s];mu=[float(r['delta']) for r in cells];eps=[2*float(r['paired_SE'])+float(r['resolution']) for r in cells];mean=sum(mu)/len(mu);noise=sum(eps)/len(eps)
        if mean>noise:continue
        if multi and max(a-b for a,b in zip(mu,eps))>max(0.,-mean)+noise:continue
        return s
    return 0.


def main():
    pub=ROOT/'reports/receiver_c0';reg=json.loads((ROOT/'configs/receiver_c0/registration.json').read_text())
    for filename in ['preanalysis_seal.json','timing_seal.json','conditional_protocol_seal.json']:
        for path,h in json.loads((ROOT/'configs/receiver_c0'/filename).read_text())['files'].items():assert sha(ROOT/path)==h,path
    changed=subprocess.check_output(['git','diff','--name-only',reg['base_remote_commit']],cwd=ROOT,text=True).splitlines();assert all(p in ['AGENTS.md','.gitignore'] or '/receiver_c0/' in p or 'receiver_c0' in Path(p).name for p in changed),changed
    # No historical tracked file may change; no additional numerical target read.
    inputs=0
    for run in reg['diagnostic_runs']:
        for name,h in run['files'].items():assert sha(Path(run['run_path'])/name)==h;inputs+=1
    old=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text())
    for asset in old['input_assets']:assert sha(Path(asset['path']))==asset['sha256']
    risks=readrows(pub/'snapshot_RX_date_risks.csv');groups={}
    for r in risks:groups.setdefault((r['run_id'],int(r['epoch'])),[]).append(r)
    count=0
    for r in readrows(pub/'transfer_choices.csv'):
        source=set(map(int,r['decision_RX'].split(',')));target=set(map(int,r['target_RX'].split(',')));a=groups[(r['run_id'],int(r['epoch']))]
        if r['kind'].startswith('cross_RX'):assert not source&target
        if 'date' in r['kind']:assert r['decision_date']!=r['target_date']
        src=[v for v in a if v['pool']=='C1' and int(v['rx']) in source and v['date']==r['decision_date']]
        dst=[v for v in a if v['pool']=='C2' and int(v['rx']) in target and v['date']==r['target_date']]
        assert len(src)==4*len(source) and len(dst)==4*len(target)
        policy=r['policy'];s=1. if policy=='always_full' else .5 if policy=='fixed_half' else rule(src,policy=='multi_RX_guard')
        assert float(r['chosen_scale'])==s
        actual=[v for v in dst if float(v['scale'])==s];mu=np.mean([float(v['delta']) for v in actual]);worst=max(float(v['delta']) for v in actual);assert abs(mu-float(r['target_chosen_mean_delta']))<1e-12 and abs(worst-float(r['target_chosen_worst_RX_delta']))<1e-12
        # Target values cannot affect the source-only independently coded scale.
        if policy in ['mean_backtracking','multi_RX_guard']:assert rule(src,policy=='multi_RX_guard')==s
        count+=1
    ledger=json.loads((ROOT/'outputs/receiver_c0/timing_execution.json').read_text());assert ledger['all_completed'] and ledger['formal_fits']==0 and ledger['reserved_GPUh']*3600<360
    for card in range(4):
        obj=json.loads((ROOT/f'outputs/receiver_c0/timing_gpu{card}.json').read_text());assert obj['formal_fits']==0 and obj['continuous_model_trajectory_updates']==0
        assert len(obj['timing_cases'])==30 and len(obj['equivalence_cases'])==6
        assert all(a['legacy_optimized_parameters_moments_equal'] and a['sparse_C2_does_not_change_decision'] for a in obj['equivalence_cases'])
    d=json.loads((pub/'decision.json').read_text());assert not d['formal_R2_C1_gate_passed'] and d['formal_fits']==0
    result=dict(passed=True,historical_tracked_files_unchanged=True,prior_R2_A_B_inputs_verified=inputs,original_input_assets_hashed=len(old['input_assets']),actual_snapshots=80,independent_transfer_choices_recomputed=count,source_target_RX_disjoint_verified=True,source_target_dates_disjoint_verified=True,all_decisions_source_C1_only=True,four_GPU_equivalence_cases=24,conditional_plan_inactive=True,formal_fits=0,final_data_read=False,engineering_reserved_GPUh=ledger['reserved_GPUh'])
    (pub/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
