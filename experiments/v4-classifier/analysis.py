"""Development prediction diagnostics; no decision thresholds are fitted."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (average_precision_score, cohen_kappa_score,
    confusion_matrix, precision_recall_fscore_support, precision_recall_curve)


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def scores(y, p, sources):
    y = np.asarray(y); p = np.asarray(p); sources = np.asarray(sources)
    if p.shape != (len(y), 5) or not np.isfinite(p).all():
        raise ValueError('Invalid prediction array')
    if not np.allclose(p.sum(1), 1, atol=1e-4) or (p < 0).any():
        raise ValueError('Invalid probabilities')
    result = {}
    for source in sorted(set(sources.tolist())):
        mask = sources == source; a = y[mask]; b = p[mask]; pred = b.argmax(1)
        pr, re, f1, support = precision_recall_fscore_support(a, pred, labels=range(5), zero_division=0)
        q = float(cohen_kappa_score(a, pred, labels=range(5), weights='quadratic'))
        if not np.isfinite(q): raise ValueError('Undefined QWK')
        result[source] = dict(n=len(a), qwk=q, precision=pr.tolist(), recall=re.tolist(),
            f1=f1.tolist(), support=support.tolist(), confusion=confusion_matrix(a,pred,labels=range(5)).tolist(),
            grade1_ap=float(average_precision_score(a==1,b[:,1])) if (a==1).any() else None,
            prediction_counts=np.bincount(pred,minlength=5).tolist(),
            accuracy=float((a==pred).mean()), mae=float(np.abs(a-pred).mean()),
            nll=float(-np.log(b[np.arange(len(a)),a].clip(1e-12)).mean()),
            brier=float(((b-np.eye(5)[a])**2).sum(1).mean()))
    macro = dict(qwk=float(np.mean([v['qwk'] for v in result.values()])))
    for key in ('precision','recall','f1'):
        macro['grade1_'+key] = float(np.mean([v[key][1] for v in result.values()]))
    aps=[v['grade1_ap'] for v in result.values() if v['grade1_ap'] is not None]
    macro['grade1_ap']=float(np.mean(aps)) if aps else None
    return dict(sources=result, macro=macro)


def write_report(records, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    save(out/'metrics.json',records)
    keys=['qwk','grade1_precision','grade1_recall','grade1_f1','grade1_ap']
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['run','source']+keys)
        for name, rec in sorted(records.items()):
            w.writerow([name,'source_macro']+[rec['macro'][k] for k in keys])
            for source,v in rec['sources'].items():
                w.writerow([name,source,v['qwk'],v['precision'][1],v['recall'][1],v['f1'][1],v['grade1_ap']])
    lines=['# Exploratory validation results','',
        'Each checkpoint is selected by source-macro validation QWK. Two seeds do not establish significance.','',
        '| Run | QWK | G1 precision | G1 recall | G1 F1 | G1 AP |',
        '|---|---:|---:|---:|---:|---:|']
    for name,rec in sorted(records.items()):
        lines.append('| '+name+' | '+' | '.join('NA' if rec['macro'][k] is None else f"{rec['macro'][k]:.4f}" for k in keys)+' |')
    lines += ['', '## Same-seed differences from unchanged CE head', '',
        '| Run | delta QWK | delta G1 precision | delta G1 recall | delta G1 F1 | delta G1 AP |',
        '|---|---:|---:|---:|---:|---:|']
    for name,rec in sorted(records.items()):
        seed=name.rsplit('_seed',1)[-1]
        base=records.get('unchanged_ce_seed'+seed) or records.get('aligned_ce_seed'+seed)
        if base:
            lines.append('| '+name+' | '+' | '.join('NA' if rec['macro'][k] is None or base['macro'][k] is None else f"{rec['macro'][k]-base['macro'][k]:+.4f}" for k in keys)+' |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def figures(y,p,sources,out,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    for source in sorted(set(sources.tolist())):
        m=sources==source; a=y[m]; b=p[m]; pred=b.argmax(1)
        fig,ax=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
        cm=confusion_matrix(a,pred,labels=range(5)); ax[0].imshow(cm,cmap='Blues')
        for i in range(5):
            for j in range(5): ax[0].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8)
        ax[0].set(xlabel='Predicted grade',ylabel='True grade',xticks=range(5),yticks=range(5))
        precision,recall,_=precision_recall_curve(a==1,b[:,1])
        ax[1].plot(recall,precision);ax[1].axhline((a==1).mean(),ls='--',color='grey')
        ax[1].set(xlabel='Grade 1 recall',ylabel='Grade 1 precision',xlim=(0,1),ylim=(0,1))
        fig.suptitle(name+' / '+source+' / validation')
        for ext in ('png','pdf'): fig.savefig(out/(name+'_'+source+'.'+ext),dpi=180)
        plt.close(fig)


def diagnose(prior, config, out):
    records={}
    for recipe in config['recipes']:
        for seed in config['seeds']:
            name=f"{recipe['name']}_seed{seed}"
            with np.load(prior/'runs'/name/'best_validation_private.npz',allow_pickle=False) as z:
                y,p,s=z['labels'],z['probabilities'],z['sources']
            records[name]=scores(y,p,s)
            figures(y,p,s,out/'figures',name)
            print('PREDICTION_DIAGNOSTICS',name,flush=True)
    write_report(records,out)
    return records
