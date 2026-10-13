#!/usr/bin/env python3
"""Finite public-storage range completion, original interrupted prefix preserved."""
import concurrent.futures
import datetime
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import requests
from acquire_wisig_manyrx import URL, EXPECTED_MEMBER_BYTES, dump

def main():
    dest=Path('/home/keyan/data/wisig-manyrx-kaggle')
    original=dest/'ManyRx.kaggle.v1.zip.partial'
    target=dest/'ManyRx.kaggle.v1.zip'
    assert original.is_file() and not target.exists()
    previous=json.loads((dest/'download_status.json').read_text())
    dump(dest/'single_stream_failure.json',previous)
    begin=time.monotonic()
    for attempt in range(1,4):
        try:
            with requests.get(URL,stream=True,timeout=(30,60)) as response:
                response.raise_for_status();signed_url=response.url
                assert urlparse(signed_url).hostname=='storage.googleapis.com'
                total=int(response.headers['Content-Length']);etag=response.headers.get('ETag')
            break
        except Exception:
            if attempt==3:raise RuntimeError('Public single-file endpoint failed after three attempts') from None
            time.sleep(3)
    prefix=original.stat().st_size
    assert prefix<total<3*1024**3
    def fetch(start,end):
        for attempt in range(1,4):
            try:
                with requests.get(signed_url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,75)) as response:
                    assert response.status_code==206
                    assert response.headers['Content-Range']==f'bytes {start}-{end}/{total}'
                    assert not etag or response.headers.get('ETag')==etag
                    data=response.content
                    assert len(data)==end-start+1
                    return data,attempt
            except Exception:
                if attempt==3:raise RuntimeError('Range transfer failed after three finite attempts') from None
                time.sleep(3)
    # Validate both ends of the interrupted prefix against the immutable v1 object.
    with original.open('rb') as stream:
        for start in [0,prefix-65536]:
            stream.seek(start);local=stream.read(65536)
            remote,_=fetch(start,start+len(local)-1)
            assert remote==local
    ranges=dest/'ranges';ranges.mkdir(exist_ok=False)
    chunk=32*1024**2
    intervals=[(start,min(start+chunk,total)-1) for start in range(prefix,total,chunk)]
    def work(interval):
        start,end=interval;data,attempts=fetch(start,end)
        path=ranges/f'{start}-{end}.part';path.write_bytes(data)
        print(json.dumps(dict(start=start,end=end,bytes=len(data),attempts=attempts)),flush=True)
        return dict(start=start,end=end,bytes=len(data),attempts=attempts,sha256=hashlib.sha256(data).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(work,intervals))
    assembled=dest/'ManyRx.kaggle.v1.assembled.zip.partial'
    with assembled.open('xb') as stream:
        with original.open('rb') as source:shutil.copyfileobj(source,stream,8*1024**2)
        for record in sorted(records,key=lambda x:x['start']):
            with (ranges/f"{record['start']}-{record['end']}.part").open('rb') as source:
                shutil.copyfileobj(source,stream,8*1024**2)
    assert assembled.stat().st_size==total
    with zipfile.ZipFile(assembled) as archive:
        entries=archive.infolist();assert len(entries)==1 and entries[0].filename=='ManyRx.pkl'
        assert entries[0].file_size==EXPECTED_MEMBER_BYTES and archive.testzip() is None
        inventory=[dict(name=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,crc32=x.CRC) for x in entries]
    digest=hashlib.sha256()
    with assembled.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    assembled.rename(target)
    record=dict(state='SEALED_ZIP_VERIFIED_THIRD_PARTY',source_url=URL,
        source_kind='third_party_public_kaggle',kaggle_version=1,kaggle_license='Unknown',
        original_dataset_license='CC BY-NC-SA 4.0',official_equivalence='NOT_PROVEN_NO_OFFICIAL_CHECKSUM',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(target),
        bytes=total,sha256=digest.hexdigest(),entries=inventory,seconds=time.monotonic()-begin,
        original_partial_preserved=str(original),ranges=records,max_connections=4,
        max_attempts_per_range=3,signals_loaded=False,training_used=False)
    dump(dest/'download_status.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='ranges'}),flush=True)

if __name__=='__main__':main()
