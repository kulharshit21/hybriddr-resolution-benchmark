#!/usr/bin/env python3
"""Create a conservative child manifest by quarantining every candidate endpoint."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import pandas as pd

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def quarantine_audit_id(parent_hash,candidate_hash):
 identity={'policy':'quarantine-all-candidate-endpoints-v1','parent_manifest_sha256':parent_hash,'candidate_ledger_sha256':candidate_hash}
 return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load_pairs(p):
 rows=[];seen=set()
 for n,line in enumerate(Path(p).read_text(encoding='utf-8').splitlines(),1):
  x=json.loads(line);pid=x.get('pair_id')
  if not pid or pid in seen or not x.get('left') or not x.get('right'):raise RuntimeError(f'invalid pair line {n}')
  seen.add(pid);rows.append(x)
 return rows
def build(parent,pairs_path,out):
 parent,pairs_path,out=map(Path,(parent,pairs_path,out));stage=Path(str(out)+'.staging')
 if out.exists() or stage.exists():raise FileExistsError(f'output/staging exists: {out}')
 manifest_path=parent/'manifest.csv';parent_hash=digest(manifest_path);frame=pd.read_csv(manifest_path)
 pairs=load_pairs(pairs_path);quarantine={x[k] for x in pairs for k in ('left','right')}
 known=set(frame.path);missing=quarantine-known
 if missing:raise RuntimeError(f'{len(missing)} candidate paths missing from parent manifest')
 child=frame[~frame.path.isin(quarantine)].copy().reset_index(drop=True)
 if set(child.path)&quarantine:raise RuntimeError('quarantine invariant failed')
 stage.mkdir(parents=True)
 try:
  child.to_csv(stage/'manifest.csv',index=False)
  inherited_path=parent/'exclusions.json';inherited=json.loads(inherited_path.read_text()) if inherited_path.exists() else []
  exclusions=inherited+[{'path':p,'reason':'conservative_candidate_endpoint_quarantine'} for p in sorted(quarantine)]
  (stage/'exclusions.json').write_text(json.dumps(exclusions,indent=2,sort_keys=True)+'\n')
  for rel in sorted(set(child.cache)):
   src=parent/rel;dst=stage/rel
   if not src.is_file():raise FileNotFoundError(src)
   dst.parent.mkdir(parents=True,exist_ok=True)
   try:os.link(src,dst)
   except OSError:shutil.copy2(src,dst)
  if digest(manifest_path)!=parent_hash:raise RuntimeError('parent manifest changed during build')
  candidate_hash=digest(pairs_path);child_hash=digest(stage/'manifest.csv')
  audit_id=quarantine_audit_id(parent_hash,candidate_hash)
  summary={'purpose':'conservative no-human-review quarantine child','audit_id':audit_id,'parent_manifest_sha256':parent_hash,'candidate_ledger_sha256':candidate_hash,'manifest_sha256':child_hash,'child_manifest_sha256':child_hash,'parent_images':len(frame),'quarantined_images':len(quarantine),'retained_images':len(child),'candidate_pairs':len(pairs),'counts':{str(k):int(v) for k,v in child.groupby(['source','split']).size().to_dict().items()},'limitations':['All candidate endpoints are removed; no human adjudication is claimed','Residual duplicates outside candidate generation may remain']}
  (stage/'prepared.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
  (stage/'QUARANTINE_COMPLETE').write_text(summary['manifest_sha256']+'\n')
  os.replace(stage,out);return summary
 except Exception:
  raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.parent,a.pairs,a.out),indent=2))
if __name__=='__main__':main()
