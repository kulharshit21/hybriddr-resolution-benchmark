import json,subprocess,argparse
from pathlib import Path
from datetime import datetime
ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path.cwd());a=ap.parse_args()
root=a.project/'runs/v7-controls-448-01'
def read(p,default):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return default
print(datetime.now().astimezone().isoformat())
try:print((root/'STATUS.txt').read_text())
except OSError:print('Not started')
done=0;remaining=[];active=read(root/'ACTIVE.json',{})
for run in sorted((root/'runs').glob('*_seed*')):
    h=read(run/'history.json',[]);done+=len(h)
    live=read(run/'LIVE_PROGRESS.json',{})
    gpu=active.get(run.name,{}).get('gpu','-')
    if not h:print(run.name,'GPU',gpu,'0/30',live);continue
    best=max(h,key=lambda x:x['mean_source_val_qwk'])
    mins=sum(x['training_seconds']+x['validation_seconds'] for x in h[-3:])/len(h[-3:])/60
    remaining.append((30-len(h))*mins)
    print(f"{run.name:30} GPU={gpu} {len(h):2}/30 QWK={best['mean_source_val_qwk']:.4f} min/epoch={mins:.2f} {live.get('stage','')}")
    for source,g in best['validation_per_grade'].items():
        print(f"  {source:8} Grade1 P={g['precision'][1]:.4f} R={g['recall'][1]:.4f} F1={g['f1'][1]:.4f}")
print(f'Epochs {done}/300; validation at QWK-selected checkpoint')
print('Evaluated:',sum((r/'evaluation.json').exists() for r in (root/'runs').glob('*_seed*')),'/10')
print('Ten 448 controls, 300 training epochs; evaluation follows each training job.')
print('Use per-architecture measured minutes/epoch; CNN and gated have different compute costs.')
subprocess.run(['nvidia-smi','-i','0,1,2,3,5','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv'])
