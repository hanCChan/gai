#!/usr/bin/env python3
"""Official ManyRx only. Finite quota-aware polling; 16 ranged connections.

Downloads remain sealed: ZIP inventory/CRC only, no pickle loading or extraction.
"""
import datetime, hashlib, json, shutil, subprocess, time, urllib.request, zipfile
from pathlib import Path

DEST = Path('/home/keyan/data/wisig-manyrx-sealed')
URL = 'https://drive.usercontent.google.com/download?id=1TtdydJCuhkvDQ1RWb3PkxakkWo2-X5Uv&export=download&confirm=t'

def status(**kwargs):
    value = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 official_page='https://cores.ee.ucla.edu/wisig/downloads/',
                 file_id='1TtdydJCuhkvDQ1RWb3PkxakkWo2-X5Uv',
                 requested_connections=16, signal_loaded=False, **kwargs)
    (DEST/'download_status.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 5):
        try:
            if shutil.disk_usage(DEST).free < 8*1024**3:
                status(state='FAILED_DISK_PREFLIGHT'); return
            with urllib.request.urlopen(URL, timeout=45) as response:
                kind=response.headers.get('Content-Type','')
                if 'text/html' in kind:
                    page=response.read(65536).decode('utf-8',errors='replace')
                    if 'Quota exceeded' in page or 'Too many users' in page:
                        status(state='WAITING_OFFICIAL_QUOTA', attempt=attempt,
                               next_attempt_seconds=900 if attempt<4 else None)
                    else:
                        status(state='BLOCKED_HTML_CONFIRMATION', attempt=attempt); return
                else:
                    first=response.read(4)
                    if first!=b'PK\x03\x04':
                        status(state='FAILED_NONZIP_RESPONSE', attempt=attempt); return
                    status(state='DOWNLOADING', attempt=attempt)
                    p=subprocess.run(['aria2c','--continue=true','--split=16',
                        '--max-connection-per-server=16','--min-split-size=8M',
                        '--max-tries=3','--retry-wait=30','--auto-file-renaming=false',
                        '--allow-overwrite=false','--console-log-level=notice',
                        '--summary-interval=30','--dir='+str(DEST),
                        '--out=ManyRx.pkl.zip',URL])
                    if p.returncode:
                        status(state='FAILED_TRANSFER', returncode=p.returncode); return
                    target=DEST/'ManyRx.pkl.zip'
                    with zipfile.ZipFile(target) as z:
                        entries=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,
                                      crc=i.CRC) for i in z.infolist()]
                        if sum(x['bytes'] for x in entries)>8*1024**3:
                            status(state='BLOCKED_UNEXPECTED_EXPANDED_SIZE',entries=entries); return
                        bad=z.testzip()
                        if bad: status(state='FAILED_CRC',member=bad); return
                    h=hashlib.sha256()
                    with target.open('rb') as f:
                        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
                    status(state='SEALED_VERIFIED',bytes=target.stat().st_size,
                           sha256=h.hexdigest(),entries=entries); return
        except Exception as error:
            status(state='TRANSPORT_ERROR',attempt=attempt,error=str(error)[:400])
        if attempt<4: time.sleep(900)
    status(state='STOPPED_OFFICIAL_QUOTA_OR_TRANSPORT', attempts=4)

if __name__=='__main__': main()
