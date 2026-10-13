#!/usr/bin/env python3
"""One public Kaggle single-file transfer; preserve partials, no quota polling.

No credentials or redirect query strings are logged. No pickle is executed.
"""
import argparse
import datetime
import hashlib
import json
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, urlparse
import requests

SLUG = 'jeffhei1206/wisig-rf-fingerprinting-dataset'
NAME = 'ManyRx.pkl/ManyRx.pkl'
EXPECTED_MEMBER_BYTES = 2037337668
URL = 'https://www.kaggle.com/api/v1/datasets/download/' + SLUG + '/' + quote(NAME, safe='') + '?datasetVersionNumber=1'

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=Path('/home/keyan/data/wisig-manyrx-kaggle'))
    args = parser.parse_args()
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / 'ManyRx.kaggle.v1.zip'
    part = target.with_suffix('.zip.partial')
    if target.exists() or part.exists():
        raise SystemExit('Existing archive/partial preserved; refusing overwrite or automatic retry')
    started = time.monotonic()
    result = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_url=URL, source_kind='third_party_public_kaggle',
                  official_equivalence='NOT_PROVEN_NO_OFFICIAL_CHECKSUM',
                  kaggle_license='Unknown', original_dataset_license='CC BY-NC-SA 4.0',
                  signals_loaded=False, training_used=False, attempts=1)
    try:
        with requests.get(URL, stream=True, timeout=(30, 90)) as response:
            result.update(http_status=response.status_code,
                          content_type=response.headers.get('Content-Type'),
                          final_host=urlparse(response.url).hostname)
            response.raise_for_status()
            if result['final_host'] != 'storage.googleapis.com':
                raise ValueError('Unexpected final download host')
            expected = int(response.headers.get('Content-Length', '0'))
            if not (0 < expected < 3 * 1024**3):
                raise ValueError('Missing/unexpected transfer size')
            digest = hashlib.sha256()
            count = 0
            last = 0
            with part.open('xb') as stream:
                for block in response.iter_content(4 * 1024**2):
                    if not block:
                        continue
                    if count == 0 and block[:4] != b'PK\x03\x04':
                        raise ValueError('Non-ZIP response')
                    stream.write(block)
                    digest.update(block)
                    count += len(block)
                    if time.monotonic() - last >= 20:
                        result.update(state='DOWNLOADING', received_bytes=count, expected_bytes=expected)
                        dump(dest / 'download_status.json', result)
                        print(json.dumps(dict(received_bytes=count, expected_bytes=expected)), flush=True)
                        last = time.monotonic()
            if count != expected:
                raise ValueError('Content-Length mismatch')
        with zipfile.ZipFile(part) as archive:
            entries = archive.infolist()
            if len(entries) != 1 or entries[0].filename != 'ManyRx.pkl' or entries[0].file_size != EXPECTED_MEMBER_BYTES:
                raise ValueError('Unexpected archive inventory')
            bad = archive.testzip()
            if bad:
                raise ValueError('Archive CRC failure')
            inventory = [dict(name=x.filename, bytes=x.file_size, compressed_bytes=x.compress_size, crc32=x.CRC) for x in entries]
        part.rename(target)
        result.update(state='SEALED_ZIP_VERIFIED_THIRD_PARTY', archive=str(target), bytes=count,
                      sha256=digest.hexdigest(), entries=inventory,
                      seconds=time.monotonic()-started)
    except Exception as error:
        # Exception URLs can contain signed credentials; retain type, not repr.
        result.update(state='STOPPED_SINGLE_ATTEMPT_FAILED', error_type=type(error).__name__,
                      partial_bytes=part.stat().st_size if part.exists() else 0,
                      seconds=time.monotonic()-started)
        dump(dest / 'download_status.json', result)
        raise SystemExit(json.dumps(result))
    dump(dest / 'download_status.json', result)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
