"""Verify completed-run resume does not rewrite trained checkpoints."""
import argparse
import subprocess
import sys
from pathlib import Path
from common import digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    paths = sorted(args.out.glob('runs/*/best.pt'))
    if not paths:
        raise ValueError('No completed smoke checkpoints')
    before = {str(p): {'sha256': digest(p), 'mtime_ns': p.stat().st_mtime_ns} for p in paths}
    command = [sys.executable, str(Path(__file__).parent / 'run.py'), 'all', '--smoke', '--out', str(args.out)]
    if args.cpu:
        command.append('--cpu')
    subprocess.run(command, check=True)
    after = {str(p): {'sha256': digest(p), 'mtime_ns': p.stat().st_mtime_ns} for p in paths}
    assert before == after, 'Completed checkpoints changed during resume'
    save_json(args.out / 'resume_verified.json', {'checkpoint_count': len(paths), 'unchanged': True, 'scope': 'completed-run resume'})
    print('VERDICT: PASS — completed-run checkpoint hashes and modification times unchanged')


if __name__ == '__main__':
    main()
