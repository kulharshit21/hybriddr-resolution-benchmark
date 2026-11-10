#!/usr/bin/env python3
"""Wait for a stable set of idle NVIDIA GPUs without stopping any process."""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def parse_csv(text):
    return [[item.strip() for item in row] for row in csv.reader(text.splitlines()) if row]


def select_idle(stats, process_uuids, count, preferred=()):
    eligible = {
        int(row['index']) for row in stats
        if int(row['memory_used_mib']) <= 100
        and int(row['memory_free_mib']) >= 38000
        and int(row['utilization_percent']) <= 5
        and row['uuid'] not in process_uuids
    }
    order = []
    for gpu in [*preferred, *sorted(eligible)]:
        if gpu in eligible and gpu not in order:
            order.append(gpu)
    return order[:count] if len(order) >= count else []


def snapshot():
    gpu_rows = parse_csv(subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True))
    stats = [dict(index=row[0], uuid=row[1], memory_used_mib=row[2],
                  memory_free_mib=row[3], utilization_percent=row[4])
             for row in gpu_rows]
    process_text = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid',
        '--format=csv,noheader,nounits',
    ], text=True)
    process_uuids = {row[0] for row in parse_csv(process_text) if row and row[0]}
    return stats, process_uuids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--checks', type=int, default=15)
    parser.add_argument('--interval', type=float, default=4)
    parser.add_argument('--preferred', default='')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    preferred = [int(item) for item in args.preferred.split(',') if item.strip()]
    stable, passes = None, 0
    while passes < args.checks:
        stats, process_uuids = snapshot()
        selected = select_idle(stats, process_uuids, args.count, preferred)
        if selected and selected == stable:
            passes += 1
        elif selected:
            stable, passes = selected, 1
        else:
            stable, passes = None, 0
        record = {'selected': selected, 'stable_checks': passes,
                  'required_checks': args.checks, 'stats': stats,
                  'compute_process_gpu_uuids': sorted(process_uuids)}
        print(json.dumps(record, sort_keys=True), flush=True)
        if passes < args.checks:
            time.sleep(args.interval)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(','.join(map(str, stable)) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
