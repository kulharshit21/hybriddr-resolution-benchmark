"""Frozen CE feature extraction and final-linear-head adaptation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from analysis import scores,save,figures,write_report


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def reconstruct_check(actual, expected, atol=1e-4):
    if actual.shape!=expected.shape or not np.allclose(actual,expected,atol=atol,rtol=0):
        raise ValueError('Baseline reconstruction failed; adaptation is blocked')


def fit_heads(x,y,v,vy,sources,weight,bias,seed,out,epochs=20):
    """Matched natural and balanced head updates, initialized from original CE."""
    records={}; device=x.device
    base=nn.Linear(x.shape[1],5).to(device)
    with torch.no_grad(): base.weight.copy_(weight);base.bias.copy_(bias)
    with torch.no_grad(): bp=base(v).softmax(1).cpu().numpy()
    baseline=scores(vy,bp,sources); records['unchanged_ce_seed'+str(seed)]=baseline
    save(out/'unchanged_ce.json',baseline)
    save(out/'head_initialization.json',dict(trainable='head.weight, head.bias only',
        parameters=base.weight.numel()+base.bias.numel(),epochs=epochs,batch=256,
        learning_rate=1e-4,weight_decay=.05,label_smoothing=.1,
        cached_views='deterministic; frozen feature extractor in evaluation mode'))
    for mode in ('natural','balanced'):
        torch.manual_seed(seed)
        head=nn.Linear(x.shape[1],5).to(device)
        with torch.no_grad(): head.weight.copy_(weight);head.bias.copy_(bias)
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.05)
        generator=torch.Generator(device='cpu').manual_seed(seed)
        weights=1/torch.bincount(y.cpu(),minlength=5).clamp_min(1).float()
        sample_weights=weights[y.cpu()]
        best=baseline;best_epoch=0;history=[]
        best_state={k:t.detach().cpu().clone() for k,t in head.state_dict().items()}
        best_probs=bp
        for epoch in range(1,epochs+1):
            started=time.monotonic()
            order=(torch.randperm(len(y),generator=generator) if mode=='natural' else
                torch.multinomial(sample_weights,len(y),replacement=True,generator=generator))
            loss_sum=0.
            head.train()
            for indices in order.split(256):
                indices=indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(head(x[indices]),y[indices],label_smoothing=.1)
                if not torch.isfinite(loss):raise ValueError('Nonfinite classifier loss')
                loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(indices)
            head.eval()
            with torch.no_grad(): p=head(v).softmax(1).cpu().numpy()
            result=scores(vy,p,sources)
            history.append(dict(epoch=epoch,loss=loss_sum/len(y),seconds=time.monotonic()-started,metrics=result))
            if result['macro']['qwk']>best['macro']['qwk']:
                best=result;best_epoch=epoch;best_probs=p
                best_state={k:t.detach().cpu().clone() for k,t in head.state_dict().items()}
            save(out/(mode+'_history.json'),history)
            save(out/'PROGRESS.json',dict(stage='HEAD_ADAPTATION',mode=mode,epoch=epoch,total=epochs,
                latest=result['macro'],best=best['macro'],best_epoch=best_epoch))
            print(f'HEAD seed={seed} mode={mode} epoch={epoch}/{epochs} QWK={result["macro"]["qwk"]:.4f} G1_F1={result["macro"]["grade1_f1"]:.4f}',flush=True)
        name=f'{mode}_head_seed{seed}'
        records[name]=best
        torch.save(dict(head=best_state,selected_epoch=best_epoch,seed=seed,mode=mode,
            feature_identity=read(out/'FEATURE_IDENTITY.json') if (out/'FEATURE_IDENTITY.json').exists() else None),out/(mode+'_head.pt'))
        np.savez_compressed(out/(mode+'_best_validation.npz'),labels=vy,probabilities=best_probs,sources=sources)
        save(out/(mode+'_selection.json'),dict(selected_epoch=best_epoch,selected=best,latest=history[-1]['metrics']))
        figures(vy,best_probs,sources,out/'figures',name)
    write_report(records,out/'comparison')
    return records


def read(path):return json.loads(Path(path).read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    # Import a packaged frozen snapshot; never mutate the V3 scientific source.
    sys.path.insert(0,str(Path(__file__).parent/'baseline'))
    import pandas as pd
    from models import HybridDR
    from data import RetinalDataset
    from torch.utils.data import DataLoader
    torch.set_num_threads(4)
    args.out.mkdir(parents=True,exist_ok=True)
    prepared=Path((args.project/'current-quality-audit-root.txt').read_text().strip())/'prepared-quarantine-all-candidates-v1'
    run=args.prior/'runs'/f'aligned_ce_seed{args.seed}'
    checkpoint=run/'best.pt';print('READING_CE_CHECKPOINT',checkpoint,flush=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=state['config'];recipe=state['recipe']
    if recipe['name']!='aligned_ce' or recipe['loss']!='ce':raise ValueError('CE checkpoint required')
    if cfg.get('tiny') or cfg['image_size']!=224:raise ValueError('Unexpected model configuration')
    model=HybridDR(cfg,recipe,initialize=False)
    model.load_state_dict(state['model'],strict=True)
    if not isinstance(model.head,nn.Linear):raise ValueError('Final linear classifier required')
    for param in model.parameters():param.requires_grad_(False)
    model=model.cuda().eval()
    capture=[]
    hook=model.head.register_forward_pre_hook(lambda module,inputs:capture.append(inputs[0].detach().float().cpu()))
    frame=pd.read_csv(prepared/'manifest.csv')
    identity=dict(checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(prepared/'manifest.csv'),
        worker_sha256=digest(__file__),seed=args.seed,feature='input to final linear head; extractor.eval()',
        baseline_epoch=state['epoch']+1)
    save(args.out/'FEATURE_IDENTITY.json',identity)
    extracted={}
    for split in ('train','val'):
        subset=frame[frame.split==split].reset_index(drop=True)
        loader=DataLoader(RetinalDataset(subset,prepared,cfg['image_size'],False),
            batch_size=cfg['batch_size'],shuffle=False,num_workers=8,pin_memory=True)
        features=[];predictions=[];labels=[];completed=0;started=time.monotonic()
        with torch.no_grad():
            for images,target,_ in loader:
                capture.clear()
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(images.cuda(non_blocking=True))
                if len(capture)!=1:raise ValueError('Feature hook capture count mismatch')
                features.append(capture[0]);predictions.append(logits.float().softmax(1).cpu());labels.append(target)
                completed+=len(target)
                save(args.out/'PROGRESS.json',dict(stage='EXTRACTION',split=split,completed=completed,total=len(subset),
                    images_per_second=completed/max(time.monotonic()-started,1e-9)))
                if len(features)%10==0:print(f'FEATURES seed={args.seed} {split} {completed}/{len(subset)}',flush=True)
        extracted[split]=dict(features=torch.cat(features),labels=torch.cat(labels),
            probabilities=torch.cat(predictions).numpy(),sources=subset.source.to_numpy(dtype=str))
        np.savez_compressed(args.out/(split+'_features_private.npz'),**{
            k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in extracted[split].items()})
    hook.remove()
    with np.load(run/'best_validation_private.npz',allow_pickle=False) as old:
        val=extracted['val']
        if not np.array_equal(old['labels'],val['labels'].numpy()) or not np.array_equal(old['sources'],val['sources']):
            raise ValueError('Validation cohort order mismatch')
        reconstruct_check(val['probabilities'],old['probabilities'])
    weight=model.head.weight.detach().float().clone();bias=model.head.bias.detach().float().clone()
    with torch.no_grad():
        reconstructed=F.linear(extracted['val']['features'].cuda(),weight,bias).softmax(1).cpu().numpy()
    # FP32 head arithmetic differs slightly from original BF16 inference.
    reconstruct_check(reconstructed,extracted['val']['probabilities'],atol=.005)
    agreement=float((reconstructed.argmax(1)==extracted['val']['probabilities'].argmax(1)).mean())
    if agreement<.999:raise ValueError('FP32 head prediction agreement below 99.9%')
    save(args.out/'BASELINE_RECONSTRUCTION.json',dict(verdict='PASS',full_model_atol=1e-4,
        fp32_head_max_abs_difference=float(np.abs(reconstructed-extracted['val']['probabilities']).max()),
        fp32_head_prediction_agreement=agreement))
    del model,state;torch.cuda.empty_cache()
    train=extracted['train'];val=extracted['val']
    records=fit_heads(train['features'].cuda(),train['labels'].cuda(),val['features'].cuda(),
        val['labels'].numpy(),val['sources'],weight,bias,args.seed,args.out)
    save(args.out/'SEED_COMPLETE.json',dict(identity=identity,records=records,
        artifacts={p.name:digest(p) for p in args.out.rglob('*') if p.is_file() and p.suffix in ('.pt','.npz')}))
    print('SEED_COMPLETE',args.seed,flush=True)


if __name__=='__main__':main()
