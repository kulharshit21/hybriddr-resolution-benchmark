"""Authenticate labels; inventory pairing metadata without changing datasets."""
import argparse,csv,hashlib,io,json,os,re,urllib.request,zipfile
from pathlib import Path
from datetime import datetime,timezone

EXPECTED={'messidor_data.csv':'bb8bbd82eb4bfddc8a0eb9aa45d84274d814e913534f8027f6a4ff0a310fda72',
          'messidor_readme.txt':'73d70083b8bf0355ba4c4b5120ecaffd1cfc1662d45bd5febeb32410a2554415'}
KAGGLE='https://www.kaggle.com/api/v1/datasets/download/google-brain/messidor2-dr-grades?datasetVersionNumber=1'
OFFICIAL='https://www.adcis.net/en/third-party/messidor2/'

def sha(b): return hashlib.sha256(b).hexdigest()
def fetch(url,limit=2000000):
    with urllib.request.urlopen(url,timeout=30) as f:
        b=f.read(limit+1)
    if len(b)>limit: raise ValueError('Metadata response too large')
    return b
def key(s): return Path(str(s).strip().replace('\\','/')).stem.lower()

def check_pairs(rows,known):
    """Validate content only. This cannot establish official provenance."""
    seen=set();pairs=[]
    for row in rows:
        ids=[key(c) for c in row if key(c) in known]
        if not ids: continue
        if len(ids)!=2 or ids[0]==ids[1] or seen.intersection(ids):
            raise ValueError('Ambiguous/repeated pairing row')
        seen.update(ids);pairs.append(ids)
    if len(pairs)!=874 or seen!=known:
        raise ValueError(f'Incomplete pairing: {len(pairs)} exams / {len(seen)} image IDs')
    return pairs

def run(project,data,out,pairs_path=None,official_source=None):
    if out.exists(): raise FileExistsError('Choose new output directory')
    out.mkdir(parents=True)
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'label_release':'google-brain/messidor2-dr-grades','version':1,
        'label_license_listed':'CC0: Public Domain','image_terms_url':OFFICIAL,
        'image_terms_summary':'Research/educational use; copying, redistribution and unauthorized commercial use prohibited; acknowledge LaTIM and Messidor program partners.',
        'label_files':{},'network_errors':[],'pairing_candidates':[],
        'exam_pairing_verified':False,'cross_exam_patient_identity_verified':False}
    raw=(data/'messidor_data.csv').read_bytes()
    known={key(r['image_id']) for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))}
    if len(known)!=1748: raise ValueError('Expected 1748 unique labelled IDs')
    for name,expected in EXPECTED.items():
        p=data/name
        result['label_files'][name]={'expected_sha256':expected,'actual_sha256':sha(p.read_bytes()) if p.exists() else None}
    result['pinned_label_release_matches']=all(x['expected_sha256']==x['actual_sha256'] for x in result['label_files'].values())
    try:
        archive=fetch(KAGGLE);z=zipfile.ZipFile(io.BytesIO(archive))
        result['live_release_sha256']=sha(archive)
        result['live_release_files_match']=all(sha(z.read(n))==EXPECTED[n] for n in EXPECTED)
        (out/'label-release-download.json').write_text(json.dumps({'url':KAGGLE,'archive_sha256':sha(archive),'member_hashes':{n:sha(z.read(n)) for n in EXPECTED}},indent=2))
    except Exception as e: result['network_errors'].append('Label release: '+str(e))
    for filename,url in [('adcismessidor2.html',OFFICIAL),('kaggle-metadata.json','https://www.kaggle.com/api/v1/datasets/list?search=messidor2-dr-grades')]:
        try:
            b=fetch(url);(out/filename).write_bytes(b)
            if filename.endswith('.json'):
                matches=[x for x in json.loads(b) if x.get('ref')=='google-brain/messidor2-dr-grades']
                if len(matches)!=1: raise ValueError('Ambiguous Kaggle release')
                result['live_label_license']=matches[0].get('licenseName')
                result['live_version']=matches[0].get('currentVersionNumber')
        except Exception as e: result['network_errors'].append(filename+': '+str(e))
    skip={'.git','__pycache__','cache','node_modules','.venv-cpu-prep','weights'}
    for root,dirs,files in os.walk(project):
        dirs[:]=[d for d in dirs if d not in skip and not d.startswith('.venv')]
        for name in files:
            p=Path(root)/name
            if p.suffix.lower() not in ('.csv','.xlsx','.xls','.ods'): continue
            if not re.search(r'pair|messidor|latim',name,re.I): continue
            if p.stat().st_size>10*1024**2: continue
            record={'path':str(p),'sha256':sha(p.read_bytes())}
            try:
                if p.suffix.lower()!='.csv':
                    record['status']='SPREADSHEET_REQUIRES_FORMAT_INSPECTION'
                else:
                    text=p.read_text(encoding='utf-8-sig')
                    try: dialect=csv.Sniffer().sniff(text[:8192],delimiters=',;\t')
                    except csv.Error: dialect=csv.excel
                    rows=list(csv.reader(io.StringIO(text),dialect))
                    pairs=check_pairs(rows,known)
                    record.update(status='COMPLETE_PAIR_CONTENTS_OFFICIAL_SOURCE_UNVERIFIED',exams=len(pairs))
            except (ValueError,UnicodeError,csv.Error) as e: record['status']=str(e)
            result['pairing_candidates'].append(record)
    if pairs_path:
        if not official_source or not official_source.startswith('https://www.adcis.net/'):
            raise ValueError('Supply documented ADCIS source URL for explicitly selected pairing CSV')
        with pairs_path.open(newline='',encoding='utf-8-sig') as f:
            text=f.read()
        try: dialect=csv.Sniffer().sniff(text[:8192],delimiters=',;\t')
        except csv.Error: dialect=csv.excel
        pairs=check_pairs(csv.reader(io.StringIO(text),dialect),known)
        selected=Path((project/'current-quality-audit-root.txt').read_text().strip())/'prepared-quarantine-all-candidates-v1/manifest.csv'
        with selected.open(newline='',encoding='utf-8') as f:
            retained={key(r['path']) for r in csv.DictReader(f) if r['source']=='MESSIDOR2'}
        if not retained<=known: raise ValueError('Unknown retained IDs')
        with (out/'EXAM_MAP.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['image_id_stem','exam_cluster','retained'])
            for pair in pairs:
                cluster='exam_'+sha('|'.join(sorted(pair)).encode())[:20]
                for image in pair:w.writerow([image,cluster,image in retained])
        result.update(exam_pairing_verified=True,pairing_source_declared_by_operator=official_source,
            pairing_csv_sha256=sha(pairs_path.read_bytes()),exams=874,retained_images=len(retained),
            retained_exams=sum(bool(set(p)&retained) for p in pairs))
    result['status']='LABELS_MATCH_PAIRING_PENDING' if result['pinned_label_release_matches'] else 'LABEL_RELEASE_MISMATCH'
    if result['exam_pairing_verified']:result['status']='LABELS_AND_EXAM_PAIR_CONTENTS_VERIFIED_SOURCE_DECLARED' if result['pinned_label_release_matches'] else 'LABEL_RELEASE_MISMATCH'
    (out/'SOURCE_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pairs',type=Path);p.add_argument('--official-source');a=p.parse_args()
    run(a.project,a.data,a.out,a.pairs,a.official_source)
