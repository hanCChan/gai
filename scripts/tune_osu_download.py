"""Measure increasing file concurrency and retain the best observed average."""
import json
import time

from osu_download_status import STATE, rpc


def main():
    results = []
    for concurrency in [32, 64, 96, 128]:
        rpc('changeGlobalOption', [{'max-concurrent-downloads': str(concurrency)}])
        samples = []
        for _ in range(6):
            time.sleep(5)
            stat = rpc('getGlobalStat')
            samples.append(int(stat['downloadSpeed']))
        active = rpc('tellActive')
        item = {'concurrency': concurrency, 'samples_bytes_per_second': samples, 'mean_bytes_per_second': sum(samples) / len(samples), 'active_files': int(stat['numActive']), 'connections': sum(int(x.get('connections', 0)) for x in active)}
        results.append(item)
        (STATE / 'concurrency_benchmark.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({'concurrency': concurrency, 'mean_MiB_s': round(item['mean_bytes_per_second'] / 1024**2, 3), 'last_MiB_s': round(samples[-1] / 1024**2, 3), 'connections': item['connections']}), flush=True)
    best = max(results, key=lambda x: x['mean_bytes_per_second'])
    rpc('changeGlobalOption', [{'max-concurrent-downloads': str(best['concurrency'])}])
    (STATE / 'max_concurrent').write_text(str(best['concurrency']) + '\n')
    print('Selected concurrency=' + str(best['concurrency']), flush=True)


if __name__ == '__main__':
    main()
