"""Read-only V4 progress display."""
import argparse
from pathlib import Path
import json
import subprocess
from datetime import datetime

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    print('FROZEN CLASSIFIER DIAGNOSTICS |',datetime.now().astimezone().strftime('%d %b %H:%M:%S %Z'),flush=True)
    print(read(args.out/'PROGRESS.json').get('message','Waiting for controller'),flush=True)
    for seed in (11,23):
        root=args.out/f'seed{seed}';p=read(root/'PROGRESS.json');assignment=read(root/'assignment.json')
        print(f'\nSeed {seed} | GPU {assignment.get("gpu","pending")} | completed={(root/"SEED_COMPLETE.json").exists()}',flush=True)
        if p.get('stage')=='EXTRACTION':
            print(f"  {p['split']} features {p['completed']}/{p['total']} | {p['images_per_second']:.1f} images/s",flush=True)
        elif p.get('stage')=='HEAD_ADAPTATION':
            print(f"  {p['mode']} head epoch {p['epoch']}/{p['total']} | latest={p['latest']} | best={p['best']}",flush=True)
    print('\nOutput:',args.out,flush=True)
    if (args.out/'V4_COMPLETE.json').exists():print('V4 COMPLETE',flush=True)
    subprocess.run(['nvidia-smi','-i','0,1,2,3,5','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv'],check=False)

if __name__=='__main__':main()
