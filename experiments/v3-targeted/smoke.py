"""Real pretrained Large/Large BF16/Adam smoke for all targeted loss arms."""
import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from common import digest,read_json,save_json,seed_all
from data import RetinalDataset
from models import HybridDR,training_objective
from diagnostics import MANIFEST_SHA


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 GPU required')
    assert digest(args.prepared/'manifest.csv')==MANIFEST_SHA
    config=read_json(Path(__file__).parent/'config.json')
    frame=pd.read_csv(args.prepared/'manifest.csv')
    train=frame[frame.split=='train'].reset_index(drop=True)
    samples=train.groupby('label',sort=True).head(1).sort_values('label').reset_index(drop=True)
    dataset=RetinalDataset(samples,args.prepared,224,False)
    images=torch.stack([dataset[i][0] for i in range(5)]).to('cuda:0')
    labels=torch.tensor(samples.label.tolist(),device='cuda:0')
    counts=torch.bincount(torch.tensor(train.label.to_numpy(),device='cuda:0'),minlength=5).float()
    torch.cuda.set_per_process_memory_fraction(.80,torch.device('cuda:0'))
    records=[]
    for recipe in config['recipes']:
        seed_all(11)
        model=HybridDR(config,recipe).to('cuda:0').train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
        torch.cuda.reset_peak_memory_stats()
        for epoch in [0,15]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=training_objective(model(images),labels,recipe,epoch,30,counts,.1)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite smoke loss')
            loss.backward()
            gradient=model.vexchange.raw_scale.grad
            if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum():raise RuntimeError('Distance-bias gradient missing')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        records.append(dict(recipe=recipe['name'],steps=2,final_loss=float(loss.detach()),peak_bytes=torch.cuda.max_memory_allocated()))
        print('TARGETED_GPU_SMOKE_PASS',records[-1],flush=True)
        del optimizer,model,loss
        gc.collect();torch.cuda.empty_cache()
    save_json(args.out,dict(verdict='PASS',config_sha256=digest(Path(__file__).parent/'config.json'),records=records,scope='hardware and gradient smoke; no outcome claim'))


if __name__=='__main__':main()
