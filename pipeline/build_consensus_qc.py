#!/usr/bin/env python3
"""Build conservative machine-evidence buckets and a stratified QC queue."""
import argparse, hashlib, json, math
from collections import defaultdict
from pathlib import Path

LARGE_LOW=0.7880438749790194; MIXUP_LOW=0.8424341114163402
LARGE_POS=0.8778188228607178; MIXUP_POS=0.8959682583808899

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def rows(path):
    result={}
    for number,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        x=json.loads(line); pid=x.get('pair_id')
        if not pid or pid in result: raise RuntimeError(f'Invalid/duplicate pair_id at {number}')
        result[pid]=x
    return result

def stratified_sample(items,n,seed):
    groups=defaultdict(list)
    for x in items:
        groups[(tuple(x['sources']),tuple(x['splits']),x['selected_classification'])].append(x)
    total=len(items); allocation={}; fractions=[]
    for key,group in groups.items():
        exact=n*len(group)/total; allocation[key]=math.floor(exact)
        fractions.append((exact-allocation[key],key))
    for _,key in sorted(fractions,reverse=True)[:n-sum(allocation.values())]: allocation[key]+=1
    chosen=[]
    for key,group in groups.items():
        ranked=sorted(group,key=lambda x:hashlib.sha256(f"{seed}:{x['pair_id']}".encode()).hexdigest())
        chosen.extend(x['pair_id'] for x in ranked[:allocation[key]])
    if len(chosen)!=n or len(set(chosen))!=n: raise RuntimeError('QC allocation invariant failed')
    return sorted(chosen)

def build(pair_path,sscd_path,lightglue_path,out,qc_size=1000,seed=20261104):
    pair,sscd,lg=rows(pair_path),rows(sscd_path),rows(lightglue_path)
    if not(set(pair)==set(sscd)==set(lg)): raise RuntimeError('Evidence ID sets differ')
    buckets=[]
    for pid in sorted(pair):
        p=pair[pid]; s=sscd[pid]['similarities']; g=lg[pid]; e=p['evidence']
        s_pos=s['sscd_large']>=LARGE_POS and s['sscd_mixup']>=MIXUP_POS
        s_low=s['sscd_large']<=LARGE_LOW and s['sscd_mixup']<=MIXUP_LOW
        l_pos=g['homography_inliers']>=22 and g['homography_inlier_ratio']>=.55
        geom_pos=e.get('ssim',-1)>=.90 and e.get('ncc',-1)>=.95 and e.get('nmae',1)<=.12 and e.get('vessel_dice',0)>=.75
        category='triple_positive_candidate' if s_pos and l_pos and geom_pos else ('machine_low_candidate' if s_low and not l_pos else 'unresolved')
        buckets.append({'pair_id':pid,'category':category,'sources':p['sources'],'splits':p['splits'],'selected_classification':p['selected_classification']})
    low=[x for x in buckets if x['category']=='machine_low_candidate']
    qc=stratified_sample(low,qc_size,seed)
    required=sorted({x['pair_id'] for x in buckets if x['category']!='machine_low_candidate'}|set(qc))
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    ledger=out/'consensus_candidates.jsonl'
    ledger.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in buckets),encoding='utf-8')
    (out/'qc_sample.json').write_text(json.dumps(qc,indent=2)+'\n')
    (out/'review_required_consensus.json').write_text(json.dumps(required,indent=2)+'\n')
    counts={k:sum(x['category']==k for x in buckets) for k in ['triple_positive_candidate','machine_low_candidate','unresolved']}
    summary={'purpose':'machine triage plus human QC queue; not adjudication','counts':counts,'qc_sample':len(qc),'review_required':len(required),'seed':seed,'inputs':{'pairs':digest(pair_path),'sscd':digest(sscd_path),'lightglue':digest(lightglue_path)},'outputs':{'consensus_candidates.jsonl':digest(ledger),'qc_sample.json':digest(out/'qc_sample.json'),'review_required_consensus.json':digest(out/'review_required_consensus.json')}}
    (out/'CONSENSUS_QC_COMPLETE.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,required=True);p.add_argument('--sscd',type=Path,required=True);p.add_argument('--lightglue',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--qc-size',type=int,default=1000);a=p.parse_args();print(json.dumps(build(a.pairs,a.sscd,a.lightglue,a.out,a.qc_size),indent=2))
if __name__=='__main__':main()
