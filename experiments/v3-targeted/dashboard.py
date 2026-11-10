"""Read-only five-second dashboard: committed epochs, live batches and grade-1 metrics."""
import argparse
from datetime import datetime,timedelta
from pathlib import Path
import subprocess
from common import read_json


def safe(path,default):
    try:return read_json(path)
    except (OSError,ValueError):return default


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    config=safe(Path(__file__).parent/'config.json',{})
    print('HYBRIDDR TARGETED LOSS STUDY |',datetime.now().astimezone().strftime('%d %b %Y %H:%M:%S %Z'))
    try:print((args.out/'STATUS.txt').read_text().strip())
    except OSError:print('Controller status pending')
    print(f"{'RUN':<39} {'GPU':>3} {'EPOCH':>6} {'BEST QWK':>9} {'G1 RECALL':>10} {'MIN/EPOCH':>9}  LIVE")
    done=epochs=0;durations=[];remaining=[]
    for recipe in config['recipes']:
        for seed in config['seeds']:
            name=f"{recipe['name']}_seed{seed}";run=args.out/'runs'/name
            h=safe(run/'history.json',[]);complete=(run/'complete.json').is_file()
            live=safe(run/'LIVE_PROGRESS.json',{});assignment=safe(run/'assignment.json',{})
            epochs+=len(h);done+=complete
            recent=[r['training_seconds']+r['validation_seconds'] for r in h[-3:]]
            duration=sum(recent)/len(recent) if recent else None
            if duration:durations.append(duration)
            remaining.append((30-len(h),duration))
            best=max(h,key=lambda x:x['mean_source_val_qwk']) if h else {}
            q=f"{best['mean_source_val_qwk']:.4f}" if best else '-'
            g=f"{best['mean_source_grade1_recall']:.4f}" if best else '-'
            speed=f'{duration/60:.2f}' if duration else '-'
            state=live.get('stage','QUEUED')
            if state=='TRAINING':state+=f" epoch {live['epoch']} batch {live['batch']}/{live['total_batches']} ({100*live['batch']/live['total_batches']:.0f}%)"
            print(f"{name:<39} {assignment.get('physical_gpu','-'):>3} {len(h):>2}/30 {q:>9} {g:>10} {speed:>9}  {state}")
    print(f'\nCompleted: {done}/10 runs | {epochs}/300 epochs ({epochs/3:.1f}%) | remaining {300-epochs}')
    if durations:
        # Wave-aware heuristic: schedule unfinished jobs onto five worker slots.
        fallback=sum(durations)/len(durations);slots=[0.]*5
        for count,duration in sorted(remaining,key=lambda x:x[0]*(x[1] or fallback),reverse=True):
            index=slots.index(min(slots));slots[index]+=count*(duration or fallback)
        seconds=max(slots)
        print(f'Five-slot ETA heuristic: {seconds/3600:.2f} h; finish ~{datetime.now().astimezone()+timedelta(seconds=seconds):%d %b %H:%M %Z}')
        print('Assumes five available slots; includes train/validation but not checkpoint I/O or interruptions.')
    print('\nGPU scope: 0,1,2,5,6 only')
    subprocess.run(['nvidia-smi','-i','0,1,2,5,6','--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw','--format=csv'],check=False)


if __name__=='__main__':main()
