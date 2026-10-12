"""Verify original OSU hashes, sample finiteness, metadata and every IQ manifest path."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'logs/osu_download'


def main():
    spec = json.loads((STATE / 'targets.json').read_text())
    iq = [x for x in spec['targets'] if x['kind'] == 'iq']
    errors = []
    valid = 0
    for item in iq:
        p = ROOT / item['path']
        if not p.is_file() or p.stat().st_size != item['bytes'] or Path(str(p) + '.aria2').exists():
            errors.append({'path': item['path'], 'error': 'missing, partial, or wrong size'})
            continue
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            while chunk := stream.read(16 * 1024**2):
                digest.update(chunk)
        if digest.hexdigest() != item['sha256']:
            errors.append({'path': item['path'], 'error': 'SHA256 mismatch with original experiment'})
            continue
        mm = np.memmap(p, dtype=np.complex64, mode='r')
        finite = all(bool(np.isfinite(mm[start:start+8192]).all()) for start in [0, len(mm)//2, len(mm)-8192])
        del mm
        if not finite:
            errors.append({'path': item['path'], 'error': 'nonfinite sampled IQ'})
        else:
            valid += 1
        metadata = p.with_suffix('.sigmf-meta')
        if not metadata.exists():
            errors.append({'path': str(metadata.relative_to(ROOT)), 'error': 'missing metadata'})
        else:
            try:
                json.loads(metadata.read_text(encoding='utf-8-sig'))
            except Exception as e:
                errors.append({'path': str(metadata.relative_to(ROOT)), 'error': str(e)})
        if valid % 25 == 0:
            print(f'IQ SHA256 verified: {valid}/{len(iq)}', flush=True)
    references = set()
    for manifest in (ROOT / 'data').rglob('*.csv'):
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                path = row.get('path', '')
                if path.startswith('data/raw/osu_lora/') and path.endswith('.dat'):
                    references.add(path)
    missing = sorted(x for x in references if not (ROOT / x).is_file())
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'state': 'complete' if valid == len(iq) and not errors and not missing else 'verification_failed', 'iq_expected': len(iq), 'iq_sha256_verified': valid, 'iq_total_GB': spec['iq_bytes']/1e9, 'errors': errors, 'manifest_unique_iq_references': len(references), 'missing_manifest_paths': missing}
    temporary = STATE / 'verification.json.tmp'
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(STATE / 'verification.json')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['state'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
