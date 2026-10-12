"""Freeze original 449 IQ targets, hashes and catalog-confirmed metadata URLs."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'logs/osu_download'
PREFIX = 'data/raw/osu_lora/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['all', 'primary'], default='all')
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT / 'inventory/bundle_checksums.json').read_text(encoding='utf-8-sig'))
    catalog = json.loads((ROOT / 'inventory/osu_remote_catalog_current.json').read_text())
    available = set(catalog['files'])
    primary = {r['path'] for r in csv.DictReader((ROOT / 'data/paper/cross_day_day1to5_source_only.csv').open())}
    iq = [x for x in original if x['path'].startswith(PREFIX) and x['path'].endswith('.dat')]
    if args.scope == 'primary':
        iq = [x for x in iq if '/Diff_Days_Indoor_Setup/' in x['path']]
    iq.sort(key=lambda x: (x['path'] not in primary, '/Diff_Days_Indoor_Setup/' not in x['path'], x['path']))
    targets = []
    missing = []
    for item in iq:
        rel = item['path'][len(PREFIX):]
        url = catalog['base'] + rel
        if url not in available:
            raise SystemExit(f'Original IQ URL absent from official catalog: {url}')
        targets.append(dict(item, url=url, kind='iq'))
        meta_path = item['path'][:-4] + '.sigmf-meta'
        meta_url = catalog['base'] + meta_path[len(PREFIX):]
        if meta_url in available:
            existing = ROOT / meta_path
            if not existing.is_file():
                targets.append({'path': meta_path, 'url': meta_url, 'kind': 'metadata'})
        else:
            missing.append(meta_path)
    remaining = sum(x['bytes'] - min((ROOT / x['path']).stat().st_size, x['bytes']) if (ROOT / x['path']).exists() else x['bytes'] for x in iq)
    # Reserve 8 GiB beyond all outstanding IQ payloads.
    if shutil.disk_usage(ROOT).free < remaining + 8 * 1024**3:
        raise SystemExit('Insufficient disk space for selected IQ targets plus 8 GiB reserve')
    lines = []
    for item in targets:
        destination = ROOT / item['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines.extend([item['url'], f'  dir={destination.parent}', f'  out={destination.name}'])
        if item.get('sha256'):
            lines.append(f"  checksum=sha-256={item['sha256']}")
        if item['kind'] == 'metadata':
            lines.extend(['  split=1', '  max-connection-per-server=1'])
    (STATE / 'targets.aria2').write_text('\n'.join(lines) + '\n')
    (STATE / 'targets.json').write_text(json.dumps({'scope': args.scope, 'iq_files': len(iq), 'iq_bytes': sum(x['bytes'] for x in iq), 'remaining_bytes_by_file_size': remaining, 'missing_official_metadata': missing, 'targets': targets}, indent=2))
    print(json.dumps({'iq_files': len(iq), 'iq_bytes': sum(x['bytes'] for x in iq), 'metadata_to_fetch': len(targets) - len(iq), 'missing_official_metadata': len(missing), 'state': str(STATE)}, indent=2))


if __name__ == '__main__':
    main()
