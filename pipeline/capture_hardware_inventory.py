"""Read current hardware/software metadata; does not initialize CUDA or alter jobs."""
import argparse,datetime,hashlib,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path

def command(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:return {'command':args,'error':str(e)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Choose a fresh inventory filename')
    versions={}
    for name in ('torch','torchvision','timm','numpy','pandas','scikit-learn','Pillow','matplotlib','psutil'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result={'recorded_at':datetime.datetime.now().astimezone().isoformat(),'python':sys.version,
        'python_executable':sys.executable,'platform':platform.platform(),'logical_cpu_count':os.cpu_count(),
        'available_cpu_affinity_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'packages':versions,'cpu':command(['lscpu','-J']),
        'gpus':command(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,power.limit','--format=csv']),
        'scope':'Current installed environment only; not proof of historical run environment, exclusive allocation, power consumption or training duration.'}
    if Path('/proc/meminfo').is_file():result['memory_kib']={k.strip():v.strip() for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2))
    print('Saved:',a.out);print('SHA256:',hashlib.sha256(a.out.read_bytes()).hexdigest())
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
