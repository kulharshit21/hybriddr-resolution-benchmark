"""Metadata-only MESSIDOR reconciliation; no images decoded or checkpoints read."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def table(path):
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def stem(value): return Path(value.replace('\\','/')).stem

def manifest(path):
    rows=[r for r in table(path) if r['source']=='MESSIDOR2']
    result={}
    for r in rows:
        key=stem(r['path'])
        if key in result: raise ValueError('Ambiguous manifest ID: '+key)
        result[key]=r
    return result

def audit(data, parent, child, out):
    if out.exists(): raise FileExistsError('Preserve previous output; choose a new directory')
    labels_path=data/'messidor_data.csv'
    labels=table(labels_path)
    raw={}
    for row in labels:
        key=stem(row['image_id'])
        if key in raw: raise ValueError('Duplicate label identifier: '+key)
        raw[key]=row
    before=manifest(parent/'manifest.csv'); after=manifest(child/'manifest.csv')
    if set(after)-set(before): raise ValueError('Child contains unknown parent IDs')
    if set(before)-set(raw): raise ValueError('Parent IDs missing label rows')
    files=defaultdict(list); metadata=[]
    for p in data.rglob('*'):
        if not p.is_file(): continue
        if p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff'):
            files[p.stem].append(str(p.relative_to(data)))
        elif p.suffix.lower() in ('.csv','.txt','.md','.xls','.xlsx','.json','.html','.pdf'):
            metadata.append(dict(path=str(p.relative_to(data)),bytes=p.stat().st_size,sha256=sha(p)))
    reasons=defaultdict(list)
    for directory in (parent,child):
        p=directory/'exclusions.json'
        if p.exists():
            for row in json.loads(p.read_text()):
                if 'path' in row: reasons[stem(row['path'])].append(row.get('reason','unspecified'))
    ledger=[]; issues=[]
    for key,row in raw.items():
        grade=row['adjudicated_dr_grade'].strip(); gradable=row['adjudicated_gradable'].strip()
        eligible=gradable in ('1','1.0') and grade!=''
        if eligible:
            number=float(grade)
            if not number.is_integer() or int(number) not in range(5): raise ValueError('Invalid DR grade: '+key)
            for frame in (before,after):
                if key in frame and float(frame[key]['label'])!=number: raise ValueError('Label mismatch: '+key)
        if key in after:
            status='retained'
            if not eligible: issues.append('Ineligible label retained: '+key)
            if after[key].get('split')!='external': issues.append('Unexpected split: '+key)
        elif not eligible: status='ungradable_or_missing_grade'
        elif key not in before:
            status='preparation_excluded' if reasons[key] else 'UNEXPLAINED_before_parent'
        else:
            status='candidate_quarantine' if 'conservative_candidate_endpoint_quarantine' in reasons[key] else 'UNEXPLAINED_parent_to_child'
        if len(files[key])!=1: issues.append(f'Image file matches={len(files[key])}: '+key)
        ledger.append(dict(image_id=row['image_id'],grade=grade,gradable=gradable,
            status=status,parent=key in before,retained=key in after,
            image_file_matches=len(files[key]),reasons=';'.join(sorted(set(reasons[key]))),
            manifest_patient=after.get(key,before.get(key,{})).get('patient','')))
    extras=sorted(set(files)-set(raw))
    if extras: issues.append('Image IDs without label rows: '+str(len(extras)))
    counts=dict(Counter(r['status'] for r in ledger))
    excluded=len(raw)-len(after)
    if len(raw)!=1748: issues.append('Label row count differs from official 1748-image universe; full official-ID join required')
    if len(after)!=1467: issues.append('Retained count differs from expected 1467')
    if any(k.startswith('UNEXPLAINED') for k in counts): issues.append('Unexplained exclusions remain')
    result=dict(status='RECONCILIATION_NEEDS_REVIEW' if issues else 'COUNTS_RECONCILED_PROVENANCE_REVIEW_REQUIRED',
        official_image_count=1748,label_rows=len(raw),parent_images=len(before),retained_images=len(after),
        labeled_images_excluded=excluded,official_minus_retained=1748-len(after),counts=counts,issues=issues,
        label_csv_sha256=sha(labels_path),parent_manifest_sha256=sha(parent/'manifest.csv'),
        child_manifest_sha256=sha(child/'manifest.csv'),metadata_files=metadata,unlabelled_image_ids=extras,
        source_candidate='https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades',
        cited_grade_reference='https://doi.org/10.1016/j.ophtha.2018.01.034',
        official_dataset='https://www.adcis.net/en/third-party/messidor2/',
        provenance_verified=False,license_verified=False,official_exam_pairing_verified=False,
        warning='README citation and matching column names establish a source lead, not authenticated release provenance. Manifest patient fields are not verified exam IDs.')
    out.mkdir(parents=True)
    with (out/'EXCLUSION_LEDGER.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# MESSIDOR label and exclusion audit','',f'Label rows: {len(raw)}; parent: {len(before)}; retained: {len(after)}.',f'Official-minus-retained: {1748-len(after)}; labelled rows excluded: {excluded}.','', '## Exclusion flow']
    lines += [f'- {k}: {v}' for k,v in counts.items()]
    lines += ['', '## Issues']+[f'- {x}' for x in issues]
    lines += ['',result['warning'],'','Verify original download record, exact release/version, file digest against publisher release, license and official examination pairing before closing provenance. No filename-derived patient IDs were invented.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','label_rows','parent_images','retained_images','counts','issues')},indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--child',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    audit(a.data,a.parent,a.child,a.out)
