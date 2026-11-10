import argparse,json,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image,ImageOps

def inspect(task):
    row,data,prepared=task
    with Image.open(Path(data)/row['path']) as im:
        original=ImageOps.exif_transpose(im).convert('RGB');w,h=original.size
        reference=original.resize((448,448),Image.Resampling.LANCZOS)
    with Image.open(Path(prepared)/row['cache']) as im:
        cache=im.convert('RGB');cw,ch=cache.size
        a=np.asarray(cache.resize((448,448)),dtype=float)
    b=np.asarray(reference,dtype=float)
    result=dict(source=row['source'],label=int(row['label']),width=w,height=h,
        aspect_ratio=w/h,cache_width=cw,cache_height=ch,
        cache_reference_mae=float(np.abs(a-b).mean()),
        original_to_224_scale=min(224/w,224/h),original_to_448_scale=min(448/w,448/h))
    for size in (224,448):
        x=np.asarray(cache.resize((size,size),Image.Resampling.BILINEAR).convert('L'),dtype=float)
        result[f'gradient_energy_{size}']=float(np.mean(np.abs(np.diff(x,axis=0)))+np.mean(np.abs(np.diff(x,axis=1))))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.prepared/'manifest.csv').open() as f:rows=list(csv.DictReader(f))
    selected=[];counts={}
    rng=np.random.default_rng(20261108)
    for source in ('EyePACS','APTOS'):
        for grade in (0,1):
            subset=[r for r in rows if r['source']==source and int(r['label'])==grade and r['split'] in ('train','val')]
            counts[f'{source}_{grade}']=len(subset)
            if not subset:raise ValueError(f'Missing development stratum {source}/{grade}')
            selected.extend(subset[i] for i in rng.choice(len(subset),min(100,len(subset)),replace=False))
    results=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for r in pool.map(inspect,[(r,str(args.project/'Datasets'),str(args.prepared)) for r in selected]):
            results.append(r)
            if len(results)%20==0:print(f'AUDIT {len(results)}/{len(selected)}',flush=True)
    with (args.out/'audit_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary=dict(sample_size=len(results),development_counts=counts,
        non_square_originals=sum(abs(r['aspect_ratio']-1)>.01 for r in results),
        undersized_caches=sum(min(r['cache_width'],r['cache_height'])<448 for r in results),
        max_cache_reference_mae=max(r['cache_reference_mae'] for r in results),
        limitations=['Gradient energy is not lesion visibility or a diagnostic quality label.',
        'This audit cannot establish clinical label correctness or prove microaneurysm loss.',
        'Only train/validation images sampled; external/internal/calibration splits excluded.'])
    summary['groups']={}
    for source in ('EyePACS','APTOS'):
        for grade in (0,1):
            group=[r for r in results if r['source']==source and r['label']==grade]
            summary['groups'][f'{source}_{grade}']={k:float(np.median([r[k] for r in group])) for k in ('width','height','aspect_ratio','cache_reference_mae','gradient_energy_224','gradient_energy_448')}
    (args.out/'AUDIT.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    if summary['undersized_caches']:raise RuntimeError('448 cache unavailable; do not run resolution screen')
    if summary['max_cache_reference_mae']>1:raise RuntimeError('Cache differs from expected preprocessing; inspect before training')

if __name__=='__main__':main()
