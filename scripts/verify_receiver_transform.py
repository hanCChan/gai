#!/usr/bin/env python3
"""Recompute published distance decisions and inspect all launch outcomes."""
import csv
import json
import math
from pathlib import Path
from receiver_transform_common import dump, sha
ROOT=Path(__file__).resolve().parents[1]

def main():
    report=ROOT/'reports/receiver_transform/2026-10-13';out=ROOT/'outputs/receiver_transform'
    dec=json.loads((report/'decision.json').read_text())
    train=list(csv.DictReader((report/'training_jobs.csv').open()))
    rows=list(csv.DictReader((report/'held_TX_RX_pair_distances.csv').open()))
    assert len(train)==20 and len(rows)==3200
    seal=json.loads((out/'pre_execution_seal.json').read_text())
    for path,digest in seal['files_sha256'].items():assert sha(ROOT/path)==digest,path
    for seed in [20261013,20261014]:
        part=[x for x in train if int(x['seed'])==seed]
        assert len({x['initial_sha256'] for x in part})==1
        assert len({x['schedule_sha256'] for x in part})==1
    for s in dec['strata']:
        for field in ['mean_squared_distance','gaussian_w2_squared']:
            avg={m:sum(float(x[field]) for x in rows if int(x['fold'])==s['fold'] and int(x['seed'])==s['seed'] and x['method']==m)/200 for m in ['identity','translation','coral','lowrank']}
            best=min(avg[m] for m in ['identity','translation','coral'])
            assert math.isclose(1-avg['lowrank']/best,s['criteria'][field]['relative_improvement'],abs_tol=1e-12)
    zero_hours=0.;zero_jobs=0
    for name in ['zero_step_launch_failure','zero_step_missing_reseal_failure']:
        e=json.loads((out/name/'execution.json').read_text())
        zero_hours+=e['reserved_gpu_hours'];zero_jobs+=len(e['jobs'])
        for p in (out/name/'runs').iterdir():
            status=json.loads((p/'status.json').read_text());assert status['optimizer_steps']==0
    for p in (out/'runs').iterdir():
        cfg=json.loads((p/'config.json').read_text());status=json.loads((p/'status.json').read_text())
        assert not set(cfg['fitted_tx_ids'])&set(cfg['held_tx_ids'])
        assert cfg['outer_rx_opened']==[] and cfg['fitting_held_tx_access'] is False
        assert status['state']=='completed' and status['optimizer_steps']==1200
        assert sha(p/'final.pt')==status['checkpoint_sha256']
        assert sha(p/'features.npy')==status['features_sha256']
    result=dict(passed=True,completed_encoder_jobs=20,distance_rows=3200,strict_TX_exclusion=True,
        common_initialization_within_seed=True,common_batch_schedule_within_seed=True,
        decision_recomputed_from_public_csv=True,zero_step_launch_aborts=zero_jobs,
        zero_step_launch_gpu_hours=zero_hours,reserved_gpu_hours_including_aborts=dec['reserved_gpu_hours']+zero_hours,
        outer_receiver_evaluation=False,classification_poc_jobs=0)
    dump(report/'verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
