"""Verify transfer checksums, signal files, and the primary protocol."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true', help='Quick size and primary manifest check')
    args = parser.parse_args()
    errors = []
    index = ROOT / 'inventory/bundle_checksums.json'
    for item in json.loads(index.read_text(encoding='utf8')):
        p = ROOT / item['path']
        if not p.is_file() or p.stat().st_size != item['bytes']:
            errors.append('Missing/size mismatch: ' + item['path'])
            continue
        if not args.data_only:
            h = hashlib.sha256()
            with p.open('rb') as f:
                while block := f.read(16 * 1024 * 1024): h.update(block)
            if h.hexdigest() != item['sha256']:
                errors.append('SHA256 mismatch: ' + item['path'])
    manifest = ROOT / 'data/paper/cross_day_day1to5_source_only.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    splits = Counter(r['split'] for r in rows)
    if splits != Counter(train=72, val=24, test=24):
        errors.append('Unexpected primary split counts: ' + str(splits))
    expected = {'train': {'1','2','3'}, 'val': {'4'}, 'test': {'5'}}
    for split, days in expected.items():
        selected = [r for r in rows if r['split'] == split]
        if {r['day'] for r in selected} != days or {int(r['label']) for r in selected} != set(range(24)):
            errors.append('Wrong days/classes: ' + split)
    for r in rows:
        p = ROOT / r['path']
        if not p.is_file() or p.stat().st_size < 8192 * 8:
            errors.append('Missing/short IQ: ' + r['path'])
    print(json.dumps({'primary_rows':len(rows), 'splits':splits, 'errors':errors}, indent=2))
    return 1 if errors else 0

if __name__ == '__main__':
    raise SystemExit(main())
