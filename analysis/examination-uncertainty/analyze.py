"""Paired examination bootstrap for mean metrics over fixed training seeds."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
SEEDS=(11,23,42,67,101)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def metrics(c):
    c=np.asarray(c,dtype=float);total=c.sum((-2,-1));actual=c.sum(-1);pred=c.sum(-2)
    w=(np.arange(5)[:,None]-np.arange(5)[None,:])**2
    observed=(c*w).sum((-2,-1));expected=(actual[..., :,None]*pred[...,None,:]*w).sum((-2,-1))/total
    qwk=1-observed/expected
    tp=np.diagonal(c,axis1=-2,axis2=-1)
    f1=np.divide(2*tp,actual+pred,out=np.zeros_like(tp),where=actual+pred>0)
    recall=np.divide(tp[...,1],actual[...,1],out=np.full_like(total,np.nan),where=actual[...,1]>0)
    return np.stack([qwk,f1.mean(-1),recall,f1[...,1]],axis=-1)
def run(project,out,repeats):
    v7=project/'runs/v7-controls-448-01';v6=project/'runs/v6-resolution-final-01'
    for root,version in ((v7,'V7'),(v6,'V6')):
        record=read(root/(version+'_COMPLETE.json'))
        if record['runs']!=10 or record['epochs']!=300 or sha(root/'RESULTS.json')!=record['results_sha256']:raise ValueError('Campaign identity mismatch')
    candidates=[]
    for d in (project/'runs').glob('messidor-pairing-*'):
        if (d/'PAIRING_AUDIT.json').is_file() and (d/'EXAM_MAP.csv').is_file():candidates.append(d)
    if not candidates:raise ValueError('No completed examination audit')
    pairing=max(candidates,key=lambda d:d.stat().st_mtime)
    audit=read(pairing/'PAIRING_AUDIT.json')
    if audit['mirror_sha256']!='d52696ff9e289feb9cf08a7f1ccc5e9a8f63bb5d6a305cc55f5b873fa06d22d3':raise ValueError('Unexpected pairing mirror')
    mapping=table(pairing/'EXAM_MAP.csv');lookup={r['path']:r for r in mapping}
    if len(mapping)!=1467 or len(lookup)!=1467:raise ValueError('Invalid map size/uniqueness')
    raw=pairing/'mirror_pairing_original.csv'
    if sha(raw)!=audit['mirror_sha256']:raise ValueError('Pairing source bytes modified')
    original=[r for r in csv.reader(raw.read_text().splitlines(),delimiter=';') if r][1:]
    expected={Path(x.strip()).stem.lower():'exam_'+hashlib.sha256('|'.join(sorted(Path(y.strip()).stem.lower() for y in pair)).encode()).hexdigest()[:20] for pair in original for x in pair}
    if any(expected.get(r['image_id_stem'])!=r['exam_cluster'] for r in mapping):raise ValueError('Examination map changed')
    clusters=sorted({r['exam_cluster'] for r in mapping});index={x:i for i,x in enumerate(clusters)}
    if len(clusters)!=835:raise ValueError('Unexpected examination count')
    rng=np.random.default_rng(20261109)
    # Shared resamples across every seed/model preserve pairing. <=34 MiB weights.
    weights=rng.multinomial(len(clusters),np.full(len(clusters),1/len(clusters)),size=repeats)
    weights=np.concatenate([np.ones((1,len(clusters)),dtype=weights.dtype),weights])
    values={};hashes={};checkpoint_ids={}
    for model,root in [('aligned224',v6),('aligned448',v6),('cnn448',v7),('gated448',v7)]:
        arch='aligned' if model.startswith('aligned') else model[:-3];size=224 if model.endswith('224') else 448
        seeds=[]
        for seed in SEEDS:
            r=root/'runs'/f'{arch}_ce_r{size}_seed{seed}'
            ev=read(r/'evaluation.json');co=read(r/'complete.json')
            if ev['checkpoint_sha256']!=co['checkpoint_sha256'] or ev.get('smoke_only') or co.get('smoke_only'):raise ValueError('Evaluation identity mismatch')
            if len(read(r/'history.json'))!=30:raise ValueError('Incomplete epochs')
            p=r/'MESSIDOR2_external_predictions.csv';rows=table(p)
            if len(rows)!=1467 or {x['path'] for x in rows}!=set(lookup):raise ValueError('Prediction map mismatch')
            probs=np.array([[float(x[f'p{k}']) for k in range(5)] for x in rows])
            if not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any() or not np.allclose(probs.sum(1),1,atol=1e-4):raise ValueError('Invalid probabilities')
            y=np.array([int(x['label']) for x in rows]);pred=probs.argmax(1)
            if any(str(x['label'])!=str(lookup[x['path']]['label']) for x in rows):raise ValueError('Label mismatch')
            cm=np.zeros((len(clusters),5,5),dtype=float)
            np.add.at(cm,([index[lookup[x['path']]['exam_cluster']] for x in rows],y,pred),1)
            sampled=(weights@cm.reshape(len(clusters),25)).reshape(-1,5,5)
            seeds.append(metrics(sampled));hashes[str(p)]=sha(p);checkpoint_ids[r.name]=co['checkpoint_sha256']
            print(f'BOOTSTRAP {model} seed={seed}',flush=True)
        values[model]=np.mean(seeds,axis=0)
    result={'scope':'Exploratory MESSIDOR-2 examination sampling uncertainty conditional on five fixed seeds; no seed resampling or ensemble prediction.',
        'pairing_source':'Documented public mirror; direct ADCIS authentication pending',
        'map_sha256':sha(pairing/'EXAM_MAP.csv'),'repeats':repeats,'seed':20261109,'exams':835,
        'prediction_sha256':hashes,'checkpoint_identities':checkpoint_ids,'contrasts':[],
        'limitations':'Unadjusted descriptive 95% percentile intervals; multiple contrasts, inspected cohorts and uncertain cross-exam patient identity prohibit confirmatory significance claims.'}
    for a,b in [('aligned448','aligned224'),('aligned448','cnn448'),('aligned448','gated448')]:
        delta=values[a]-values[b]
        for k,name in enumerate(('qwk','macro_f1','grade1_recall','grade1_f1')):
            valid=delta[1:,k][np.isfinite(delta[1:,k])]
            if len(valid)<.99*repeats:raise ValueError('Too many undefined bootstrap replicates')
            result['contrasts'].append(dict(contrast=a+' minus '+b,metric=name,point_delta=float(delta[0,k]),
                examination_percentile95=np.quantile(valid,[.025,.975]).tolist(),finite_replicates=len(valid)))
    (out/'EXAM_BOOTSTRAP.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'COMPLETE.json').write_text(json.dumps({'result_sha256':sha(out/'EXAM_BOOTSTRAP.json')}))

def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=5000);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('Already queued');return
        def status(s):print(s,flush=True);(a.out/'STATUS.txt').write_text(s+'\n')
        if (a.out/'COMPLETE.json').exists():
            if sha(a.out/'EXAM_BOOTSTRAP.json')!=read(a.out/'COMPLETE.json')['result_sha256']:raise ValueError('Prior result modified')
            status('COMPLETE');return
        try:
            status('WAITING_FOR_V7_COMPLETE')
            while not (a.project/'runs/v7-controls-448-01/V7_COMPLETE.json').is_file():time.sleep(30)
            status('RUNNING_EXAM_BOOTSTRAP')
            run(a.project,a.out,a.repeats);status('COMPLETE')
        except Exception as e:status('FAILED '+str(e));raise
if __name__=='__main__':main()
