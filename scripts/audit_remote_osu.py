"""Read official HTML directory indexes only; never download signal binaries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

BASE = 'https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/'

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            link = dict(attrs).get('href')
            if link:
                self.links.append(link)

def fetch(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Paper2-dataset-inventory/1.0'})
        with urllib.request.urlopen(req, timeout=25) as response:
            body = response.read(2 * 1024 * 1024).decode('utf8', errors='replace')
        parser = Links(); parser.feed(body)
        links = []
        for value in parser.links:
            target = urllib.parse.urljoin(url, value)
            if target.startswith(BASE) and '?' not in target and '#' not in target and target != url:
                # Follow descendants only, never parent directory links.
                if target.startswith(url):
                    links.append(target)
        return {'url':url, 'status':200, 'links':sorted(set(links))}
    except urllib.error.HTTPError as error:
        return {'url':url, 'status':error.code, 'error':str(error), 'links':[]}
    except Exception as error:
        return {'url':url, 'status':None, 'error':str(error), 'links':[]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='inventory/osu_remote_catalog_20261011.json')
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    visited=set(); pending={BASE}; pages=[]; files=set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending:
            futures={pool.submit(fetch,url):url for url in sorted(pending)}
            pending=set()
            for future in as_completed(futures):
                result=future.result(); pages.append(result); visited.add(result['url'])
                for target in result['links']:
                    if target.endswith('/'):
                        if target not in visited:
                            pending.add(target)
                    elif target.endswith(('.dat','.sigmf-meta')):
                        files.add(target)
            print(f'directory_pages={len(pages)} pending={len(pending)} signal_and_metadata_links={len(files)}',flush=True)
    probe=fetch(BASE+'Diff_Days_Indoor_Setup/Day2/Device9/')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    result={'checked_utc':datetime.now(timezone.utc).isoformat(), 'base':BASE,
            'scope':'All reachable directory indexes; link presence does not validate binary contents.',
            'pages':sorted(pages,key=lambda x:x['url']), 'files':sorted(files), 'device9_probe':probe}
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'pages':len(pages),'files':len(files),'errors':[x for x in pages if x['status'] != 200],
                      'device9_probe_status':probe['status']}))

if __name__ == '__main__':
    main()
