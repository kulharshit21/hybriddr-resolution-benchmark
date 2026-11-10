"""Read-only development label/cache and prior validation diagnostics."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import digest, save_json
from models import training_objective

MANIFEST_SHA='7ecdade0df288689e6913cf0e44a23b27daed84a12b340193a4e331629e59de3'


def verify_development_labels(frame, data):
    eye=pd.read_csv(data/'EyePACS/trainLabels.csv')
    aptos=pd.read_csv(data/'APTOS2019/train.csv')
    if eye.image.duplicated().any() or aptos.id_code.duplicated().any():
        raise ValueError('Ambiguous original label join')
    truth={f'EyePACS/train/{r.image}.jpeg':int(r.level) for r in eye.itertuples()}
    truth.update({f'APTOS2019/train_images/{r.id_code}.png':int(r.diagnosis) for r in aptos.itertuples()})
    dev=frame[frame.source.isin(['EyePACS','APTOS'])]
    errors=[r.path for r in dev.itertuples() if truth.get(r.path)!=int(r.label)]
    if errors:raise ValueError(f'Original CSV label mismatch/missing key: {errors[:5]}')
    return len(dev)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=args.prepared/'manifest.csv'
    assert digest(manifest)==MANIFEST_SHA,'Manifest mismatch'
    args.out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(manifest)
    if frame.path.duplicated().any() or frame.patient.isna().any() or not frame.label.isin(range(5)).all():
        raise ValueError('Invalid paths/grades')
    if frame.groupby('patient').split.nunique().max()!=1:
        raise ValueError('Patient split overlap in available identifiers')
    checked=verify_development_labels(frame,args.project/'Datasets')
    print(f'Development source-CSV label joins: PASS ({checked} images)',flush=True)
    dev=frame[frame.split.isin(['train','val'])]
    root=args.prepared.resolve()
    def valid_cache(cache):
        path=(root/cache).resolve()
        return path.is_relative_to(root) and path.is_file()
    missing=0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for i,valid in enumerate(pool.map(valid_cache,dev.cache),1):
            missing+=not valid
            if i%2000==0 or i==len(dev):print(f'Development cache paths: {i}/{len(dev)}',flush=True)
    if missing:raise ValueError(f'Missing/outside-root development caches: {missing}')
    distribution=frame.groupby(['source','split','label']).size().rename('n').reset_index()
    distribution.to_csv(args.out/'child_class_counts.csv',index=False)
    parent=args.prepared.parent/'prepared-quality/manifest.csv'
    parent_hash=None
    if parent.is_file():
        original=pd.read_csv(parent)
        if not set(frame.path).issubset(set(original.path)):raise ValueError('Child is not subset of parent')
        excluded=original[~original.path.isin(frame.path)]
        excluded.groupby(['source','split','label']).size().rename('excluded').reset_index().to_csv(args.out/'exclusion_class_counts.csv',index=False)
        original.groupby(['source','split','label']).size().rename('parent_n').reset_index().to_csv(args.out/'parent_class_counts.csv',index=False)
        parent_hash=digest(parent)
    counts=torch.tensor(np.bincount(frame[frame.split=='train'].label,minlength=5),dtype=torch.float32)
    records=[]
    config=json.loads((Path(__file__).parent/'config.json').read_text())
    for recipe in config['recipes']:
        for epoch in [0,15,29]:
            logits=torch.zeros(5,5,requires_grad=True)
            labels=torch.arange(5)
            loss=training_objective(logits,labels,recipe,epoch,30,counts,.1)
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(logits.grad).all():raise ValueError('Loss probe nonfinite')
            records.append(dict(recipe=recipe['name'],epoch=epoch,loss=float(loss.detach()),gradient_norm=float(logits.grad.norm())))
    pd.DataFrame(records).to_csv(args.out/'controlled_loss_probes.csv',index=False)
    old=[]
    for path in sorted((args.project/'runs/publication-campaign-v1/primary-large-large/runs').glob('*/history.json')):
        history=json.loads(path.read_text())
        if not history:continue
        best=max(history,key=lambda r:r['mean_source_val_qwk'])
        for source,values in best['validation'].items():
            cm=np.asarray(values['confusion_matrix'])
            old.append(dict(run=path.parent.name,source=source,grade1_support=int(cm[1].sum()),
                grade1_recall=float(cm[1,1]/max(1,cm[1].sum())),grade1_predictions=int(cm[:,1].sum()),
                best_validation_qwk=best['mean_source_val_qwk']))
    pd.DataFrame(old).to_csv(args.out/'prior_validation_grade1.csv',index=False)
    save_json(args.out/'DIAGNOSTICS_PASS.json',dict(verdict='PASS',manifest_sha256=digest(manifest),
        development_label_joins=checked,cache_paths_checked=len(dev),missing_caches=missing,
        training_counts=counts.int().tolist(),parent_manifest_sha256=parent_hash,
        limitations=['CSV consistency is not expert regrading','No held-out predictions read or used for selection',
                    'Cache existence check is not full decode verification','Clinical grade-map semantics and licenses require source documentation']))
    print('TARGETED_DIAGNOSTICS: PASS',flush=True)


if __name__=='__main__':main()
