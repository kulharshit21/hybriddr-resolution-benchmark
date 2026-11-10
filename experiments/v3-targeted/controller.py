"""Five-slot development loss screen, UUID-bound GPUs 0,1,2,5,6 only."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
from common import digest,identity,read_json,resources,save_json
from diagnostics import MANIFEST_SHA

HERE=Path(__file__).resolve().parent
ALLOWED={0,1,2,5,6}


def snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(r[0]),uuid=r[1].strip(),free=int(r[2]),total=int(r[3])) for r in csv.reader(rows.splitlines()) if int(r[0]) in ALLOWED]


def eligible(rows, occupied):
    return sorted([r for r in rows if r['index'] in ALLOWED and r['index'] not in occupied and r['free']>=15360],key=lambda r:(-r['free'],r['index']))


def status(out,message):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'),message,flush=True)
    (out/'STATUS.txt').write_text(message+'\n')


def jobs(config):
    return [dict(config=config,recipe=recipe,seed=seed,cpu_ids=[])
            for recipe in config['recipes'] for seed in config['seeds']]


def frozen_contract(out,config):
    current=dict(config=config,code_sha256={p.name:digest(p) for p in sorted(HERE.glob('*.py'))},manifest_sha256=MANIFEST_SHA)
    path=out/'STUDY_CONTRACT.json'
    if path.exists() and read_json(path)!=current:raise ValueError('Frozen study/code differs; preserve output and use a new version')
    save_json(path,current)
    return identity(current)


def verify_completion(run,job,prepared):
    record=read_json(run/'complete.json')
    if record['smoke_only'] or record['run_hash']!=identity(dict(job=job,manifest=digest(prepared/'manifest.csv'))):raise ValueError('Completed run identity mismatch')
    if len(read_json(run/'history.json'))!=job['config']['epochs']:raise ValueError('Epoch count mismatch')
    if digest(run/'best.pt')!=record['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stop-owned-mamaguard',action='store_true')
    args=ap.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to((args.project/'runs').resolve()) or out.name in {'publication-campaign-v1','v2-ce-screen-shared-01'}:raise ValueError('Isolated output required')
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    import psutil
    with (out/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=read_json(HERE/'config.json')
        study=frozen_contract(out,config)
        prepared=Path((args.project/'current-quality-audit-root.txt').read_text().strip())/'prepared-quarantine-all-candidates-v1'
        assert digest(prepared/'manifest.csv')==MANIFEST_SHA
        environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS']}
        from importlib.metadata import version
        save_json(out/'environment.json',dict(python=sys.version,versions={k:version(k) for k in ['torch','timm','numpy','pandas','scikit-learn']},initial_environment=environment))
        status(out,'DEVELOPMENT_DIAGNOSTICS_RUNNING')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        subprocess.run([sys.executable,'-u',str(HERE/'diagnostics.py'),'--project',str(args.project),'--prepared',str(prepared),'--out',str(out/'diagnostics')],env=env,check=True)
        if args.stop_owned_mamaguard:
            raise ValueError("Site-specific process management is excluded from this public release")
        smoke=out/'GPU_SMOKE_PASS.json'
        if not smoke.exists():
            status(out,'WAITING_FOR_SMOKE_GPU_AT_LEAST_30000_MIB_FREE')
            while True:
                cards=[r for r in snapshot() if r['free']>=30000]
                if cards:break
                time.sleep(4)
            gpu=sorted(cards,key=lambda r:(-r['free'],r['index']))[0]
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1')
            status(out,f"REAL_LARGE_LARGE_SMOKE_RUNNING gpu={gpu['index']}")
            subprocess.run([sys.executable,'-u',str(HERE/'smoke.py'),'--prepared',str(prepared),'--out',str(smoke)],env=env,check=True)
        if read_json(smoke)['config_sha256']!=digest(HERE/'config.json') or read_json(smoke)['verdict']!='PASS':raise ValueError('Smoke mismatch')
        hardware=resources(config)
        if len(hardware['cpu_ids'])<5*(config['threads_per_gpu']+config['workers_per_gpu']):raise RuntimeError('Insufficient CPU budget for frozen five-slot loader allocation')
        all_jobs=jobs(config);pending=[]
        for job in all_jobs:
            name=f"{job['recipe']['name']}_seed{job['seed']}"
            run=out/'runs'/name
            if (run/'complete.json').is_file():verify_completion(run,job,prepared)
            else:pending.append(job)
        running={};attempts={};failed=[]
        try:
            while pending or running:
                for card,record in list(running.items()):
                    proc,log,job,slot=record
                    rc=proc.poll()
                    if rc is None:continue
                    log.close();del running[card]
                    name=f"{job['recipe']['name']}_seed{job['seed']}"
                    if rc:
                        tail=(out/'runs'/name/'train.log').read_text(errors='replace')[-12000:].lower()
                        if ('out of memory' in tail or 'no microbatch fits' in tail) and attempts[name]<3:
                            pending.append(job);status(out,f'MEMORY_RETRY run={name}')
                        else:
                            failed.append(name);status(out,f'FAILED run={name} exit={rc}; see train.log')
                    else:
                        verify_completion(out/'runs'/name,job,prepared)
                        status(out,f'COMPLETE run={name} gpu={card}')
                if pending and len(running)<5 and psutil.virtual_memory().used<=hardware['ram_budget_bytes']:
                    for card in eligible(snapshot(),set(running))[:5-len(running)]:
                        if not pending:break
                        job=pending.pop(0);name=f"{job['recipe']['name']}_seed{job['seed']}"
                        run=out/'runs'/name;run.mkdir(parents=True,exist_ok=True)
                        jobfile=out/'jobs'/(name+'.json')
                        if jobfile.exists() and read_json(jobfile)!=job:raise ValueError('Job changed: '+name)
                        save_json(jobfile,job)
                        slots={r[3] for r in running.values()};slot=next(i for i in range(5) if i not in slots)
                        cpus=hardware['cpu_ids'][slot::5]
                        budget=min(int(card['free']*2**20*.8),int(card['total']*2**20*.85))
                        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],V2_CPU_IDS=','.join(map(str,cpus)),V2_MEMORY_BUDGET_BYTES=str(budget),OMP_NUM_THREADS=str(config['threads_per_gpu']),MKL_NUM_THREADS=str(config['threads_per_gpu']),OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                        log=(run/'train.log').open('a')
                        proc=subprocess.Popen([sys.executable,'-u',str(HERE/'train.py'),'--job',str(jobfile),'--prepared',str(prepared),'--out',str(run),'--device','cuda:0'],env=env,stdout=log,stderr=subprocess.STDOUT)
                        running[card['index']]=(proc,log,job,slot)
                        attempts[name]=attempts.get(name,0)+1
                        save_json(run/'assignment.json',dict(physical_gpu=card['index'],uuid=card['uuid'],cpu_ids=cpus,budget_bytes=budget))
                        status(out,f"LAUNCHED run={name} gpu={card['index']} pid={proc.pid}")
                save_json(out/'QUEUE_PROGRESS.json',dict(total_runs=len(all_jobs),pending=len(pending),active=len(running),failed=failed,allowed_gpus=sorted(ALLOWED),study_id=study))
                time.sleep(4)
            if failed:raise RuntimeError('Failed runs: '+', '.join(failed))
            for job in all_jobs:verify_completion(out/'runs'/f"{job['recipe']['name']}_seed{job['seed']}",job,prepared)
            subprocess.run([sys.executable,'-u',str(HERE/'summarize.py'),'--out',str(out)],check=True)
            save_json(out/'TARGETED_SCREEN_COMPLETE.json',dict(runs=10,epochs=300,study_id=study,scope='development-only exploratory loss screen; no held-out evaluation'))
            status(out,'TARGETED_SCREEN_COMPLETE; review validation and grade1 trade-offs before independent confirmation')
        finally:
            for proc,log,_,_ in running.values():
                if proc.poll() is None:proc.terminate()
                proc.wait();log.close()


if __name__=='__main__':main()
