"""Read every original asset; record text structure, safe checkpoint loads and HDF5 checks."""
import ast
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = {x['path']: x for x in json.loads((ROOT / 'inventory/bundle_checksums.json').read_text(encoding='utf-8-sig'))}
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    results = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith('logs/') or rel.endswith(('osu_remote_catalog_current.json', 'asset_read_audit_current.json')):
            continue
        item = {'path': rel, 'bytes': p.stat().st_size}
        try:
            digest = hashlib.sha256()
            with p.open('rb') as stream:
                while chunk := stream.read(16 * 1024 * 1024):
                    digest.update(chunk)
            item['sha256'] = digest.hexdigest()
            if rel in index:
                item['matches_original_sha256'] = item['sha256'] == index[rel]['sha256']
            if p.suffix == '.h5':
                with h5py.File(p, 'r') as f:
                    item['datasets'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in f.items()}
                    item['attributes'] = {k: str(v) for k, v in f.attrs.items()}
                    labels, counts = np.unique(f['label'][:], return_counts=True)
                    item['labels'] = dict(zip(map(str, labels.tolist()), counts.tolist()))
                    n = len(f['data'])
                    item['sample_packets_finite'] = all(bool(np.isfinite(f['data'][i]).all()) for i in [0, n // 2, n - 1])
            elif p.suffix == '.pt':
                v = torch.load(p, map_location='cpu', weights_only=True)
                item['checkpoint_keys'] = list(v) if isinstance(v, dict) else str(type(v))
                if isinstance(v, dict):
                    item['config'] = v.get('config', v.get('args'))
                    state = v.get('model', v.get('state_dict', v.get('model_state_dict', {})))
                    if isinstance(state, dict):
                        item['tensor_shapes'] = {k: list(x.shape) for k, x in state.items() if hasattr(x, 'shape')}
                del v
            elif p.suffix != '.dat':
                content = p.read_text(encoding='utf-8-sig')
                item['lines'] = len(content.splitlines())
                if p.suffix in ('.json', '.sigmf-meta'):
                    v = json.loads(content)
                    item['top_keys'] = list(v) if isinstance(v, dict) else f'list[{len(v)}]'
                    if p.name in ('metrics.json', 'run_config.json'):
                        item['content'] = v
                    if p.suffix == '.sigmf-meta':
                        item['global'] = v.get('global')
                        item['captures'] = v.get('captures')
                elif p.suffix in ('.csv', '.tsv'):
                    rows = list(csv.reader(content.splitlines(), delimiter='\t' if p.suffix == '.tsv' else ','))
                    item['header'] = rows[0] if rows else []
                    item['rows'] = max(0, len(rows) - 1)
                    if rows and 'path' in rows[0] and 'split' in rows[0]:
                        records = list(csv.DictReader(content.splitlines()))
                        item['split_counts'] = dict(Counter(r['split'] for r in records))
                        item['referenced_files'] = len(set(r['path'] for r in records))
                elif p.suffix == '.py':
                    tree = ast.parse(content)
                    item['definitions'] = [{'name': n.name, 'line': n.lineno, 'doc': ast.get_docstring(n)} for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
                elif p.suffix in ('.sh', '.ps1'):
                    item['commands'] = [line.strip() for line in content.splitlines() if any(word in line for word in ('python', 'manifest', 'checkpoint', 'wget', 'curl', 'aria2', 'winner', 'best_'))][:100]
        except Exception as e:
            item['inspection_error'] = f'{type(e).__name__}: {e}'
        results.append(item)
        if len(results) % 100 == 0:
            print(f'inspected={len(results)} latest={rel}', flush=True)
    summary = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'files': len(results), 'by_extension': dict(Counter(Path(x['path']).suffix for x in results)), 'errors': [x for x in results if 'inspection_error' in x], 'hash_mismatches': [x['path'] for x in results if x.get('matches_original_sha256') is False], 'method': 'Full byte read and SHA256 for every file; full text parsing; safe checkpoint deserialize; HDF5 all labels, dataset headers and three sampled packets.'}
    target = ROOT / 'inventory/asset_read_audit_current.json'
    target.write_text(json.dumps({'summary': summary, 'files': results}, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)


if __name__ == '__main__':
    main()
