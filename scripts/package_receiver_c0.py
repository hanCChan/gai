#!/usr/bin/env python3
"""Derived-only complete C0 audit archive and SHA manifest."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'reports/receiver_c0'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert json.loads((PUB/'verification.json').read_text())['passed'];entries=[]
    for p in PUB.iterdir():
        if p.suffix=='.csv':entries.append((p,'tables/'+p.name))
    for p in (ROOT/'outputs/receiver_c0').iterdir():
        if p.suffix in ['.log','.json']:entries.append((p,'execution/'+p.name))
    members=[]
    with zipfile.ZipFile(PUB/'evidence.zip','w',compression=zipfile.ZIP_LZMA) as z:
        for p,name in sorted(entries):
            assert p.suffix in ['.csv','.json','.log'];b=p.read_bytes();z.writestr(name,b);members.append(dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    with zipfile.ZipFile(PUB/'evidence.zip') as z:
        assert z.testzip() is None
        for m in members:assert hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    (PUB/'evidence_members.json').write_text(json.dumps(dict(archive_sha256=sha(PUB/'evidence.zip'),members=members,signals_weights_optimizer_states_or_full_features_included=False),indent=2)+'\n')
    sources=[p for d in ['src/receiver_c0','configs/receiver_c0','docs/receiver_c0'] for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in str(p)]+list((ROOT/'scripts').glob('*receiver_c0.py'))
    public=[p for p in PUB.iterdir() if p.is_file() and p.name not in ['PUBLIC_SHA256.json','EVIDENCE_INDEX.md']]
    (PUB/'PUBLIC_SHA256.json').write_text(json.dumps(dict(files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sources+public},archive_is_derived_only=True),indent=2)+'\n')
    (PUB/'EVIDENCE_INDEX.md').write_text('# C0 evidence index\n\n'+str(len(members))+' derived-only archive members verified. Archive SHA256: '+sha(PUB/'evidence.zip')+'\n\nEvery complete CSV table and numerical/CPU/GPU execution log is in evidence.zip, mapped in evidence_members.json. Direct reports: SOURCE_AUDIT.md, TRANSFER_REPORT.md, COST_REPORT.md, DECISION.md. Current conditional18-fit protocol is in docs/receiver_c0/R2_C1_CONDITIONAL_PROTOCOL.md and is inactive. No new fitting or final-data access. Input snapshot hashes are in snapshot_index.json; private Adam states/weights and signal arrays remain local. Public source/report SHA256 is in PUBLIC_SHA256.json.\n\nExtract with `python -m zipfile -e evidence.zip extracted`. Remote commit-tree and changed-byte SHA256/report readback are required before calling publication complete.\n')
    print('archive members',len(members),'bytes',(PUB/'evidence.zip').stat().st_size,'sha256',sha(PUB/'evidence.zip'))

if __name__=='__main__':main()
