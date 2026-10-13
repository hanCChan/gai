#!/usr/bin/env python3
"""Package complete derived evidence; never archive signals, states or weights."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from wisig_w01_common import sha,dump


def main():
    out=ROOT/'outputs/receiver_r2';pub=ROOT/'reports/receiver_r2/r2_a';reg=json.loads((ROOT/'configs/receiver_r2/registration.json').read_text());entries=[]
    for j in reg['jobs']:
        run=out/j['run_id']
        for p in sorted(run.iterdir()):
            if p.suffix in ['.json','.csv']:entries.append((p,'jobs/'+j['run_id']+'/'+p.name))
        log=out/(j['run_id']+'.log');entries.append((log,'logs/'+log.name))
    for name in ['execution.json','measured_budget.json']:
        entries.append((out/name,'execution/'+name))
    for p in out.glob('*.log'):
        if '_seed' not in p.name:entries.append((p,'administration/'+p.name))
    for p in pub.glob('*.csv'):entries.append((p,'tables/'+p.name))
    for p in (ROOT/'reports/receiver_r2/preflight').glob('*.json'):entries.append((p,'preflight/'+p.name))
    members=[]
    with zipfile.ZipFile(pub/'evidence.zip','w',compression=zipfile.ZIP_LZMA) as z:
        for p,name in entries:
            assert p.suffix in ['.json','.csv','.log'] and not any(s in name for s in ['strict_cis.npy','selected.pt','actual_adam_epoch'])
            data=p.read_bytes();z.writestr(name,data);members.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(pub/'evidence.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(members)
        for m in members:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    # Keep each connector blob below7.5MB, preserving every original member byte.
    archives={'evidence.zip':sha(pub/'evidence.zip')};member_archive={m['path']:'evidence.zip' for m in members}
    if (pub/'evidence.zip').stat().st_size>7500000:
        full=out/'full_evidence_before_split.zip';(pub/'evidence.zip').rename(full)
        groups=[];bucket=[];size=0
        with zipfile.ZipFile(full) as z:
            for m in members:
                n=z.getinfo(m['path']).compress_size+len(m['path'])*2+200
                if bucket and size+n>7400000:groups.append(bucket);bucket=[];size=0
                assert n<7400000,('Single evidence member requires separate chunk packaging',m['path'],n)
                bucket.append(m);size+=n
            if bucket:groups.append(bucket)
            archives={};member_archive={}
            for i,bucket in enumerate(groups):
                name='evidence.zip' if i==0 else f'evidence_part{i+1:02d}.zip'
                with zipfile.ZipFile(pub/name,'w',compression=zipfile.ZIP_LZMA) as target:
                    for m in bucket:target.writestr(m['path'],z.read(m['path']));member_archive[m['path']]=name
                assert (pub/name).stat().st_size<7500000
                archives[name]=sha(pub/name)
    for m in members:m['archive']=member_archive[m['path']]
    for archive,h in archives.items():
        assert sha(pub/archive)==h
        with zipfile.ZipFile(pub/archive) as z:
            assert z.testzip() is None
            expected=[m for m in members if m['archive']==archive];assert set(z.namelist())=={m['path'] for m in expected}
            for m in expected:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    dump(pub/'evidence_members.json',dict(archives=archives,archive_sha256=archives['evidence.zip'],compression='lossless ZIP_LZMA',members=members,private_signals_states_weights_features_included=False))
    files=[p for p in pub.iterdir() if p.is_file() and p.suffix!='.csv' and p.name not in ['PUBLIC_ARTIFACT_SHA256.json','EVIDENCE_INDEX.md']]
    files += list(pub.glob('*.csv'))
    files += [p for directory in ['src/receiver_r2','configs/receiver_r2','docs/receiver_r2','reports/receiver_r2/preflight'] for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    files += list((ROOT/'scripts').glob('*receiver_r2*.py'))
    archived={'all_finite_risks.csv','actual_fitting_batch_risks.csv','per_RX_TX_date.csv','confusion_counts.csv','C1_C2_RX_pairs.csv','mode_dropout_comparison.csv'}
    dump(pub/'PUBLIC_ARTIFACT_SHA256.json',dict(files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size,delivery=member_archive['tables/'+p.name]+'/tables/'+p.name if p.parent==pub and p.name in archived else 'direct_and_archive' if p.parent==pub and p.suffix=='.csv' else 'direct') for p in files},input_assets=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text())['input_assets'],raw_IQ_weights_full_features_optimizer_states_published=False))
    (pub/'EVIDENCE_INDEX.md').write_text('# R2-A and R2-B evidence\n\nComplete derived-only archives:\n\n'+'\n'.join('- '+name+': '+h for name,h in archives.items())+'\n\n'+str(len(members))+' members verified individually against evidence_members.json, which maps every member to its archive. Includes every job configuration/status/selection/learning curve, risk table, Adam norm/hash statistics and complete process log. All large CSVs are delivered under tables/, with exact archive path/SHA256/bytes in PUBLIC_ARTIFACT_SHA256.json. Read-only offline choices are explicitly separated from controlled trajectories.\n\nLocal parameter/Adam snapshots, raw signals, complete features, predictions and weights remain private; only their hashes and derived aggregate losses/counters are public. Independent numerical/packet/count validation is in verification.json. Extract each archive using `python -m zipfile -e ARCHIVE.zip extracted`.\n\nNo R2-C fitting; final8 RX/date3/equalized values remain sealed. Publication is complete only after remote commit-tree and all changed-byte SHA256 readback, including REPORT.md and DECISION.md.\n')
    print('archive',len(members),'members', (pub/'evidence.zip').stat().st_size,'bytes',sha(pub/'evidence.zip'))

if __name__=='__main__':main()
