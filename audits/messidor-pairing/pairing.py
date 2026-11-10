"""Validate a pinned public mirror and join examination IDs. No model changes."""
import argparse,csv,hashlib,io,json,urllib.request
from pathlib import Path
from collections import Counter

COMMIT='b8c9322863b5e2d79df964e524d74ad53d1d1b7a'
URL=f'https://raw.githubusercontent.com/ok1zjf/RFM/{COMMIT}/datasets/messidor2_pairs.csv'
PAIR_HASH='d52696ff9e289feb9cf08a7f1ccc5e9a8f63bb5d6a305cc55f5b873fa06d22d3'
LABEL_HASH='bb8bbd82eb4bfddc8a0eb9aa45d84274d814e913534f8027f6a4ff0a310fda72'
MANIFEST_HASH='7ecdade0df288689e6913cf0e44a23b27daed84a12b340193a4e331629e59de3'
def sha(b):return hashlib.sha256(b).hexdigest()
def key(s):return Path(str(s).strip().replace('\\','/')).stem.lower()
def read_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def validate(raw,known):
    rows=[r for r in csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline=None),delimiter=';') if r and any(c.strip() for c in r)]
    if [c.strip().lower() for c in rows[0]]!=['left','right']:raise ValueError('Unexpected pair header')
    pairs=[];seen=set()
    for row in rows[1:]:
        if len(row)!=2:raise ValueError('Expected exactly two cells per examination')
        ids=[key(c) for c in row]
        if ids[0]==ids[1] or seen.intersection(ids):raise ValueError('Repeated image or self pair')
        if not set(ids)<=known:raise ValueError('Pair contains unknown image')
        seen.update(ids);pairs.append(ids)
    if len(pairs)!=874 or seen!=known:raise ValueError('Must cover exactly 1748 IDs in 874 examinations')
    return pairs

def join(pairs,rows):
    lookup={image:'exam_'+sha('|'.join(sorted(pair)).encode())[:20] for pair in pairs for image in pair}
    result=[];seen=set()
    for row in rows:
        if row['source']!='MESSIDOR2':continue
        image=key(row['path'])
        if image in seen:raise ValueError('Duplicate retained image ID')
        if image not in lookup:raise ValueError('Retained image missing from pairing')
        if row['split']!='external':raise ValueError('MESSIDOR image outside external split')
        seen.add(image)
        result.append(dict(path=row['path'],image_id_stem=image,exam_cluster=lookup[image],label=row['label'],split=row['split']))
    return result

def acquire(out,labels):
    if out.exists():raise FileExistsError('Choose new output directory')
    out.mkdir(parents=True)
    b=labels.read_bytes()
    if sha(b)!=LABEL_HASH:raise ValueError('Label release hash mismatch')
    known={key(r['image_id']) for r in read_csv(labels)}
    with urllib.request.urlopen(URL,timeout=40) as f:raw=f.read(100000)
    if sha(raw)!=PAIR_HASH:raise ValueError('Pinned mirror file modified')
    pairs=validate(raw,known)
    (out/'mirror_pairing_original.csv').write_bytes(raw)
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/ok1zjf/RFM/{COMMIT}/README.md',timeout=40) as f:
        readme=f.read(200000)
    (out/'MIRROR_README.md').write_bytes(readme)
    return pairs,known,dict(source_url=URL,commit=COMMIT,mirror_sha256=PAIR_HASH,
        readme_sha256=sha(readme),exams=874,images=1748,missing_ids=0,repeated_ids=0,
        original_adcIS_bytes_authenticated=False,cross_exam_patient_identity_established=False,
        interpretation='Public research mirror attributed to official ADCIS pairing by its maintainers; structural and ID coverage verified. Direct ADCIS byte comparison remains pending.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path);p.add_argument('--labels',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    pairs,known,report=acquire(a.out,a.labels)
    if a.project:
        root=Path((a.project/'current-quality-audit-root.txt').read_text().strip())
        manifest=root/'prepared-quarantine-all-candidates-v1/manifest.csv'
        if sha(manifest.read_bytes())!=MANIFEST_HASH:raise ValueError('Training manifest identity mismatch')
        rows=join(pairs,read_csv(manifest))
        if len(rows)!=1467:raise ValueError('Expected exactly 1467 retained MESSIDOR images')
        labels={key(r['image_id']):r['adjudicated_dr_grade'] for r in read_csv(a.labels)}
        if any(float(labels[r['image_id_stem']])!=float(r['label']) for r in rows):raise ValueError('Retained label mismatch')
        with (a.out/'EXAM_MAP.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        counts=Counter(r['exam_cluster'] for r in rows)
        report.update(retained_images=len(rows),retained_exams=len(counts),bilateral_exams=sum(v==2 for v in counts.values()),
            single_eye_exams=sum(v==1 for v in counts.values()),zero_retained_exams=874-len(counts),manifest_sha256=MANIFEST_HASH)
        # Read only completed evaluations; do not inspect partially written V7 predictions.
        report['completed_prediction_joins']=[]
        expected={key(r['path']):float(r['label']) for r in rows}
        for name in ('v6-resolution-final-01','v7-controls-448-01'):
            for run in sorted((a.project/'runs'/name/'runs').glob('*_seed*')):
                if not (run/'evaluation.json').exists():continue
                pred=run/'MESSIDOR2_external_predictions.csv'
                records=read_csv(pred)
                observed={key(r['path']):float(r['label']) for r in records}
                if len(observed)!=len(records) or observed!=expected:raise ValueError('Prediction IDs/labels differ: '+str(pred))
                report['completed_prediction_joins'].append(dict(run=run.name,rows=len(records),sha256=sha(pred.read_bytes())))
        report['status']='EXAM_JOIN_COMPLETE_USING_DOCUMENTED_MIRROR'
    else:report['status']='FULL_MIRROR_CONTENT_VALIDATED_LOCAL_ONLY'
    (a.out/'PAIRING_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
