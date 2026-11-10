import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,average_precision_score
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt

def report(root,reference=None):
    rows=[]
    runs=sorted((root/'runs').glob('*_seed*'))
    if reference is not None:
        marker=json.loads((reference/'V6_COMPLETE.json').read_text())
        if marker['runs']!=10 or marker['epochs']!=300:raise ValueError('V6 incomplete')
        runs+=sorted((reference/'runs').glob('aligned_ce_r448_seed*'))
    for run in runs:
        complete=json.loads((run/'complete.json').read_text());ev=json.loads((run/'evaluation.json').read_text())
        if ev['checkpoint_sha256']!=complete['checkpoint_sha256']:raise ValueError('Evaluation checkpoint mismatch')
        for cohort,scores in ev['scores'].items():
            f=pd.read_csv(run/(cohort+'_predictions.csv'));y=f.label.to_numpy();p=f[[f'p{k}' for k in range(5)]].to_numpy();pred=p.argmax(1)
            pr,re,f1,support=precision_recall_fscore_support(y,pred,labels=range(5),zero_division=0)
            row=dict(run=run.name,architecture=run.name.split('_ce_')[0],seed=int(run.name.split('seed')[1]),cohort=cohort,**{k:v for k,v in scores['raw'].items() if isinstance(v,(int,float))})
            for k in range(5):
                row.update({f'grade{k}_precision':pr[k],f'grade{k}_recall':re[k],f'grade{k}_f1':f1[k],f'grade{k}_ap':average_precision_score(y==k,p[:,k]) if (y==k).any() else None})
            rows.append(row)
            cm=confusion_matrix(y,pred,labels=range(5));fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
            for i in range(5):
                for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
            ax.set(xlabel='Predicted grade',ylabel='True grade',title=f'{run.name}\n{cohort}',xticks=range(5),yticks=range(5));fig.tight_layout()
            dest=root/'figures';dest.mkdir(exist_ok=True)
            for ext in ('png','svg'):fig.savefig(dest/(run.name+'_'+cohort+'.'+ext),dpi=180)
            plt.close(fig)
    table=pd.DataFrame(rows);table.to_csv(root/'evaluation_metrics.csv',index=False)
    metrics=['qwk','macro_f1','grade1_precision','grade1_recall','grade1_f1','grade1_ap']
    table.groupby(['architecture','cohort'])[metrics].agg(['mean','std','count']).to_csv(root/'summary.csv')
    contrasts=[];rng=np.random.default_rng(20261108)
    for cohort,group in table.groupby('cohort'):
        for metric in metrics:
            paired=group.pivot(index='seed',columns='architecture',values=metric).dropna()
            if len(paired)!=5:raise ValueError('Expected five paired seeds')
            pairs=[('gated','cnn')]+([('aligned','cnn'),('aligned','gated')] if 'aligned' in paired else [])
            for candidate,control in pairs:
                delta=(paired[candidate]-paired[control]).to_numpy();boot=rng.choice(delta,(10000,5),replace=True).mean(1)
                contrasts.append(dict(cohort=cohort,metric=metric,contrast=candidate+' minus '+control,mean_delta=float(delta.mean()),seed_deltas=delta.tolist(),seed_bootstrap_95=np.quantile(boot,[.025,.975]).tolist()))
    (root/'PAIRED_COMPARISON.json').write_text(json.dumps(dict(contrasts=contrasts,limitation='Exploratory; five-seed bootstrap measures seed variability only, not patient uncertainty. Test cohorts previously inspected.'),indent=2))
    (root/'REPORT.md').write_text('# 448 architecture controls\n\nTen fresh 30-epoch CNN/gated controls; fixed validation-QWK selection. V6 aligned448 reference included when available. See summary.csv and PAIRED_COMPARISON.json.\n\nExploratory comparisons on previously inspected cohorts. No automatic superiority claim.\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reference',type=Path);a=ap.parse_args();report(a.out,a.reference)
