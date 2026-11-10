"""Two real-image BF16 optimizer steps for each CE fusion-screen architecture."""
import argparse
import gc
import os
import subprocess
from pathlib import Path

import torch
import pandas as pd
from common import digest, read_json, save_json, seed_all
from data import RetinalDataset
from models import HybridDR, training_objective


def check_shared_gpu(prior, physical_gpu):
    if physical_gpu is None:
        raise ValueError('Shared smoke requires --physical-gpu')
    selected = {int(value.strip()) for value in (prior / 'selected-gpus.txt').read_text().split(',')}
    if physical_gpu in selected:
        raise ValueError('Smoke GPU belongs to the V1 pool')
    row = subprocess.check_output([
        'nvidia-smi', '-i', str(physical_gpu), '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip()
    uuid, free = [value.strip() for value in row.split(',')]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('Bind CUDA_VISIBLE_DEVICES to the selected GPU UUID')
    if int(free) < 16000:
        raise ValueError('Shared smoke requires at least 16000 MiB currently free')
    return uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--prior-campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shared-smoke', action='store_true')
    parser.add_argument('--physical-gpu', type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.shared_smoke:
        check_shared_gpu(args.prior_campaign, args.physical_gpu)
    elif not (args.prior_campaign / 'FINAL_CAMPAIGN_COMPLETE').is_file():
        raise ValueError('Wait for V1 completion')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('An allocated BF16-capable GPU is required')
    if args.shared_smoke:
        torch.cuda.set_per_process_memory_fraction(.35)
    here = Path(__file__).resolve().parent
    config = read_json(here / 'config_screen_ce.json')
    if (config.get('tiny') or config['pretrained'] is not True or
            config['vit'] != 'vit_large_patch14_dinov2.lvd142m' or
            config['cnn'] != 'convnext_large.fb_in22k_ft_in1k'):
        raise ValueError('Expected pretrained Large/Large screen')
    manifest = args.prepared / 'manifest.csv'
    if digest(manifest) != '7ecdade0df288689e6913cf0e44a23b27daed84a12b340193a4e331629e59de3':
        raise ValueError('Unexpected data manifest')
    frame = pd.read_csv(manifest)
    frame = frame[frame.split == 'train'].reset_index(drop=True)
    counts = torch.bincount(torch.tensor(frame.label.values), minlength=5).float().cuda()
    dataset = RetinalDataset(frame, args.prepared, config['image_size'], False)
    images = torch.stack([dataset[i][0] for i in range(2)]).cuda()
    labels = torch.tensor([dataset[i][1] for i in range(2)]).cuda()
    records = []
    for recipe in config['recipes']:
        seed_all(11)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = HybridDR(config, recipe).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images)
                loss = training_objective(logits, labels, recipe, step, 10, counts, .1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite smoke loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        records.append(dict(recipe=recipe['name'], steps=2, final_loss=float(loss.detach()),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        print('GPU_SMOKE_PASS', records[-1], flush=True)
        del optimizer, model, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    save_json(args.out, dict(verdict='PASS', purpose='hardware smoke', recipes=records,
                            shared_smoke=args.shared_smoke, physical_gpu=args.physical_gpu,
                            models_sha256=digest(here / 'models.py'),
                            base_models_sha256=digest(here / 'base_models.py'),
                            manifest_sha256=digest(manifest)))


if __name__ == '__main__':
    main()
