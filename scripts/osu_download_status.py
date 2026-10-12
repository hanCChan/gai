"""Show real aria2 byte progress, or verified finished status if aria2 has exited."""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'logs/osu_download'


def rpc(method, params=None):
    body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'aria2.' + method, 'params': params or []}).encode()
    request = urllib.request.Request('http://127.0.0.1:16801/jsonrpc', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=5) as response:
        result = json.load(response)
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result['result']


def main():
    targets = json.loads((STATE / 'targets.json').read_text())
    total = targets['iq_bytes']
    verified = STATE / 'verification.json'
    try:
        stat = rpc('getGlobalStat')
        options = rpc('getGlobalOption')
        jobs = rpc('tellActive') + rpc('tellWaiting', [0, 2000]) + rpc('tellStopped', [0, 2000])
        iq_jobs = [x for x in jobs if x.get('files') and x['files'][0]['path'].endswith('.dat')]
        done = sum(int(x['completedLength']) for x in iq_jobs)
        speed = int(stat['downloadSpeed'])
        print(json.dumps({'state': 'downloading', 'iq_total_files': targets['iq_files'], 'iq_complete_files': sum(x['status'] == 'complete' for x in iq_jobs), 'iq_downloaded_GB': round(done / 1e9, 3), 'iq_total_GB': round(total / 1e9, 3), 'percent': round(100 * done / total, 2), 'speed_MiB_s': round(speed / 1024**2, 2), 'eta_minutes_at_current_speed': round((total - done) / speed / 60, 1) if speed else None, 'active_files': stat['numActive'], 'waiting_files': stat['numWaiting'], 'error_files': sum(x['status'] == 'error' for x in jobs), 'max_concurrent': options['max-concurrent-downloads'], 'split': options['split']}, indent=2))
    except Exception as e:
        if verified.exists():
            print(verified.read_text())
        else:
            print(json.dumps({'state': 'aria2_not_running_or_unavailable', 'detail': str(e), 'resume_command': 'bash scripts/run_osu_download_tmux.sh'}, indent=2))


if __name__ == '__main__':
    main()
