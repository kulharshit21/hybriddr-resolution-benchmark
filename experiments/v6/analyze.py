import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,confusion_matrix,cohen_kappa_score,precision_recall_fscore_support
def summarize(root):
    rows=[]
    for run in sorted((root/'runs').glob('*_seed*')):
        with np.load(run/'best_validation_private.npz',allow_pickle=False) as z:
            for source in np.unique(z['sources']):
                mask=z['sources']==source;y=z['labels'][mask];p=z['probabilities'][mask];pred=p.argmax(1)
                pr,re,f1,s=precision_recall_fscore_support(y,pred,labels=range(5),zero_division=0)
                rows.append(dict(run=run.name,source=str(source),qwk=float(cohen_kappa_score(y,pred,weights='quadratic')),
                    grade1_precision=float(pr[1]),grade1_recall=float(re[1]),grade1_f1=float(f1[1]),
                    grade1_ap=float(average_precision_score(y==1,p[:,1])),grade1_support=int(s[1]),
                    confusion_matrix=confusion_matrix(y,pred,labels=range(5)).tolist()))
    paired=[]
    for seed in (11,23,42,67,101):
        for source in ('EyePACS','APTOS'):
            a=next(r for r in rows if r['run']==f'aligned_ce_r224_seed{seed}' and r['source']==source)
            b=next(r for r in rows if r['run']==f'aligned_ce_r448_seed{seed}' and r['source']==source)
            paired.append(dict(seed=seed,source=source,delta_448_minus_224={k:b[k]-a[k] for k in ('qwk','grade1_precision','grade1_recall','grade1_f1','grade1_ap')}))
    result=dict(checkpoint_selection='mean source validation QWK',runs=rows,paired_deltas=paired,
        limitation='Three-seed exploratory validation screen; not confirmatory held-out evidence')
    (root/'COMPARISON.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(summarize(a.out),indent=2))
