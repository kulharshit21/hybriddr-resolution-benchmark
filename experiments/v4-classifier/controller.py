"""Wait for V3 completion, diagnose validation predictions, run two frozen-head screens."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from analysis import save,diagnose,write_report
from worker import digest

ALLOWED={0,1,2,3,5}
HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())

def status(out,message):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'),message,flush=True)
    save(out/'PROGRESS.json',dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),message=message))

def main():
    import fcntl
    import psutil
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    project=args.project.resolve();out=args.out.resolve()
    if out.parent!=project/'runs' or out.name!='v4-classifier-diagnostic-01':raise ValueError('Isolated output required')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=project/'runs/v3-targeted-loss-01'
        original=project/'experiments/v3-targeted'
        config=read(original/'config.json')
        contract=read(prior/'STUDY_CONTRACT.json')
        if config!=contract['config']:raise ValueError('V3 configuration differs from contract')
        current={p.name:digest(p) for p in original.glob('*.py')}
        if current!=contract['code_sha256']:raise ValueError('V3 frozen source changed')
        for p in (HERE/'baseline').glob('*.py'):
            if digest(p)!=contract['code_sha256'].get(p.name):raise ValueError('Packaged baseline mismatch: '+p.name)
        prepared=Path((project/'current-quality-audit-root.txt').read_text().strip())/'prepared-quarantine-all-candidates-v1'
        if digest(prepared/'manifest.csv')!=contract['manifest_sha256']:raise ValueError('Manifest changed')
        names=[f"{r['name']}_seed{s}" for r in config['recipes'] for s in config['seeds']]
        if len(names)!=10 or config['epochs']!=30 or config['seeds']!=[11,23]:raise ValueError('Unexpected V3 registry')
        while True:
            done=epochs=0
            for name in names:
                run=prior/'runs'/name
                try:h=read(run/'history.json')
                except FileNotFoundError:h=[]
                epochs+=len(h)
                done+=int((run/'complete.json').exists() and len(h)==30)
            status(out,f'WAITING_FOR_V3 runs={done}/10 epochs={epochs}/300')
            if done==10:break
            time.sleep(30)
        status(out,'VERIFYING_V3_CHECKPOINT_HASHES_AND_PREDICTION_IDENTITIES')
        import numpy as np
        import pandas as pd
        validation=pd.read_csv(prepared/'manifest.csv');validation=validation[validation.split=='val']
        for name in names:
            run=prior/'runs'/name;record=read(run/'complete.json')
            if record['smoke_only'] or digest(run/'best.pt')!=record['checkpoint_sha256']:
                raise ValueError('Checkpoint mismatch: '+name)
            job=read(prior/'jobs'/(name+'.json'))
            import hashlib
            expected=hashlib.sha256(json.dumps(dict(job=job,manifest=contract['manifest_sha256']),sort_keys=True).encode()).hexdigest()
            if record['run_hash']!=expected:raise ValueError('Run identity mismatch: '+name)
            history=read(run/'history.json');best=max(history,key=lambda x:x['mean_source_val_qwk'])
            with np.load(run/'best_validation_private.npz',allow_pickle=False) as z:
                if int(z['epoch'])!=best['epoch'] or not np.array_equal(z['labels'],validation.label.to_numpy()) or not np.array_equal(z['sources'],validation.source.to_numpy(dtype=str)):
                    raise ValueError('Saved prediction identity mismatch: '+name)
            status(out,'VERIFIED '+name)
        package={str(p.relative_to(HERE)):digest(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.sh')}
        identity=dict(package=package,manifest=contract['manifest_sha256'],v3_contract=digest(prior/'STUDY_CONTRACT.json'),
            ce_checkpoints={str(s):digest(prior/'runs'/f'aligned_ce_seed{s}'/'best.pt') for s in config['seeds']},
            allowed_gpus=sorted(ALLOWED),head_epochs=20,development_only=True)
        if (out/'IDENTITY.json').exists() and read(out/'IDENTITY.json')!=identity:raise ValueError('V4 identity changed; use a new output version')
        save(out/'IDENTITY.json',identity)
        status(out,'PREDICTION_DIAGNOSTICS_RUNNING')
        diagnose(prior,config,out/'prediction_diagnostics')
        seeds=list(config['seeds']);pending=[];running={};records={}
        for seed in seeds:
            dest=out/f'seed{seed}'
            if (dest/'SEED_COMPLETE.json').exists():
                result=read(dest/'SEED_COMPLETE.json')
                if result['identity']['checkpoint_sha256']!=identity['ce_checkpoints'][str(seed)]:raise ValueError('Completed seed checkpoint changed')
                for name,value in result['artifacts'].items():
                    if digest(dest/name)!=value:raise ValueError('Seed artifact changed: '+name)
                records.update(result['records'])
            else:pending.append(seed)
        cpus=psutil.Process().cpu_affinity()
        if len(cpus)<24:raise ValueError('At least 24 logical CPUs required for two bounded workers')
        try:
            while pending or running:
                for gpu,(proc,log,seed) in list(running.items()):
                    rc=proc.poll()
                    if rc is None:continue
                    log.close();del running[gpu]
                    if rc:raise RuntimeError(f'Seed {seed} failed; inspect seed{seed}/worker.log. Other active workers are stopped.')
                    result=read(out/f'seed{seed}'/'SEED_COMPLETE.json');records.update(result['records'])
                    status(out,f'SEED_COMPLETE seed={seed} gpu={gpu}')
                if pending:
                    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
                    cards=sorted([(int(a),b.strip(),int(c)) for a,b,c in csv.reader(raw.splitlines()) if int(a) in ALLOWED and int(a) not in running and int(c)>=15360],key=lambda x:(-x[2],x[0]))
                    for gpu,uuid,free in cards[:min(len(pending),2-len(running))]:
                        seed=pending.pop(0);dest=out/f'seed{seed}';dest.mkdir(exist_ok=True)
                        slot=seeds.index(seed);env=os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                        # Linux taskset binds the worker and its loader descendants.
                        affinity=cpus[slot::2]
                        log=(dest/'worker.log').open('a')
                        proc=subprocess.Popen(['taskset','-c',','.join(map(str,affinity)),sys.executable,'-u',str(HERE/'worker.py'),
                            '--project',str(project),'--prior',str(prior),'--out',str(dest),'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                        running[gpu]=(proc,log,seed)
                        save(dest/'assignment.json',dict(gpu=gpu,uuid=uuid,cpu_ids=affinity,pid=proc.pid))
                        status(out,f'LAUNCHED seed={seed} gpu={gpu} pid={proc.pid}')
                save(out/'QUEUE.json',dict(pending=pending,active={str(g):r[2] for g,r in running.items()},completed=len(records)//3,allowed_gpus=sorted(ALLOWED)))
                time.sleep(5)
            write_report(records,out/'classifier_comparison')
            save(out/'V4_COMPLETE.json',dict(verdict='COMPLETE',seeds=seeds,identity=identity,scope='exploratory train/validation only'))
            status(out,'COMPLETE; inspect prediction_diagnostics/REPORT.md and classifier_comparison/REPORT.md')
        finally:
            for proc,log,_ in running.values():
                children=psutil.Process(proc.pid).children(recursive=True) if proc.poll() is None else []
                for child in children:
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                if proc.poll() is None:proc.terminate()
                proc.wait();log.close()

if __name__=='__main__':main()
