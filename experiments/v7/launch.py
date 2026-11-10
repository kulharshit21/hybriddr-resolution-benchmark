import argparse,json,os,sys,subprocess,time,csv,hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
ALLOWED={0,1,2,3,5}
def recipes():
    return [dict(name='cnn_ce_r448',architecture='cnn',loss='ce'),dict(name='gated_ce_r448',architecture='gated',loss='ce')]
def available_cards(raw,active,pending_count):
    cards=[(int(i),u.strip(),int(f)) for i,u,f in csv.reader(raw.splitlines()) if int(i) in ALLOWED and int(i) not in active and int(f)>=20000]
    return cards[:min(pending_count,5-len(active))]
def verify_completion(run,job,manifest_hash):
    record=json.loads((run/'complete.json').read_text())
    expected=hashlib.sha256(json.dumps({'job':job,'manifest':manifest_hash},sort_keys=True).encode()).hexdigest()
    if record.get('run_hash')!=expected or record.get('smoke_only') or digest(run/'best.pt')!=record['checkpoint_sha256'] or len(json.loads((run/'history.json').read_text()))!=30:
        raise ValueError('Invalid completion: '+str(run))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    import fcntl,psutil
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args()
    p=args.project;out=p/'runs/v7-controls-448-01';out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepared=Path((p/'current-quality-audit-root.txt').read_text().strip())/'prepared-quarantine-all-candidates-v1'
        if digest(prepared/'manifest.csv')!='7ecdade0df288689e6913cf0e44a23b27daed84a12b340193a4e331629e59de3':raise ValueError('Manifest mismatch')
        identity={str(x.relative_to(HERE)):digest(x) for x in HERE.rglob('*') if x.is_file() and x.suffix in ('.py','.json')}
        identity_path=out/'IDENTITY.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Screen source changed')
        identity_path.write_text(json.dumps(identity,indent=2))
        def status(s):
            print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True);(out/'STATUS.txt').write_text(s)
        status('PREPROCESSING_AUDIT')
        subprocess.run([sys.executable,'-u',str(HERE/'audit.py'),'--project',str(p),'--prepared',str(prepared),'--out',str(out/'audit')],check=True)
        pending=[]
        for recipe in recipes():
            for seed in (11,23,42,67,101):
                cfg=json.loads((HERE/'config.json').read_text());cfg.update(image_size=448,epochs=30,workers_per_gpu=8,threads_per_gpu=4)
                job=dict(config=cfg,recipe=recipe,seed=seed,cpu_ids=[])
                name=f"{recipe['name']}_seed{seed}";run=out/'runs'/name;run.mkdir(parents=True,exist_ok=True)
                jobfile=out/(name+'.json')
                if jobfile.exists() and json.loads(jobfile.read_text())!=job:raise ValueError('Job changed')
                jobfile.write_text(json.dumps(job,indent=2))
                if (run/'complete.json').exists():
                    verify_completion(run,job,digest(prepared/'manifest.csv'))
                if (run/'evaluation.json').exists():
                    if not (run/'complete.json').exists():raise ValueError('Evaluation without training completion')
                    ev=json.loads((run/'evaluation.json').read_text())
                    if ev['checkpoint_sha256']!=digest(run/'best.pt'):raise ValueError('Evaluation mismatch')
                else:pending.append((name,jobfile,run))
        active={};cpus=psutil.Process().cpu_affinity()
        if len(cpus)<60:raise RuntimeError('At least 60 CPUs needed for five workers')
        try:
            while pending or active:
                for gpu,(proc,log,name,slot) in list(active.items()):
                    rc=proc.poll()
                    if rc is None:continue
                    log.close();del active[gpu]
                    if rc:raise RuntimeError(f'{name} failed exit={rc}; inspect train.log')
                    status('COMPLETE '+name)
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
                cards=available_cards(raw,active,len(pending))
                for gpu,uuid,free in cards:
                    name,jobfile,run=pending.pop(0);slot=next(i for i in range(5) if i not in {a[3] for a in active.values()})
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuid,V2_CPU_IDS=','.join(map(str,cpus[slot::5])),V2_MEMORY_BUDGET_BYTES=str(int(free*2**20*.8)),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
                    log=(run/'train.log').open('a');proc=subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py'),'--job',str(jobfile),'--prepared',str(prepared),'--out',str(run),'--device','cuda:0'],env=env,stdout=log,stderr=subprocess.STDOUT)
                    active[gpu]=(proc,log,name,slot);status(f'LAUNCHED {name} gpu={gpu} pid={proc.pid}')
                tmp=out/'ACTIVE.tmp'
                tmp.write_text(json.dumps({v[2]:dict(gpu=k,pid=v[0].pid) for k,v in active.items()}))
                tmp.replace(out/'ACTIVE.json')
                if pending and not active:status('WAITING_FOR_ALLOWED_GPU_WITH_20000_MIB_FREE')
                time.sleep(5)
            results={}
            for run in sorted((out/'runs').glob('*_seed*')):
                verify_completion(run,json.loads((out/(run.name+'.json')).read_text()),digest(prepared/'manifest.csv'))
                h=json.loads((run/'history.json').read_text());results[run.name]=dict(epochs=len(h),best=max(h,key=lambda x:x['mean_source_val_qwk']),latest=h[-1])
            if len(results)!=10:raise ValueError('Expected ten verified runs')
            (out/'RESULTS.json').write_text(json.dumps(results,indent=2))
            subprocess.run([sys.executable,str(HERE/'report.py'),'--out',str(out),'--reference',str(p/'runs/v6-resolution-final-01')],check=True)
            (out/'V7_COMPLETE.json').write_text(json.dumps(dict(runs=10,epochs=300,results_sha256=digest(out/'RESULTS.json')),indent=2))
            status('V7_SCREEN_COMPLETE; exploratory validation only')
        finally:
            for proc,log,_,_ in active.values():
                if proc.poll() is None:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        try:child.terminate()
                        except psutil.NoSuchProcess:pass
                    proc.terminate()
                proc.wait();log.close()
if __name__=='__main__':main()
