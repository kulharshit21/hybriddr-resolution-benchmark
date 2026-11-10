#!/usr/bin/env python3
"""One-step hardware preflight for the publication Large/Large model."""
import argparse
import json
import os
import time
from pathlib import Path


EXPECTED_VIT = 'vit_large_patch14_dinov2.lvd142m'
EXPECTED_CNN = 'convnext_large.fb_in22k_ft_in1k'


def validate_config(config):
    errors = []
    if config.get('tiny'):
        errors.append('tiny mode must be disabled')
    if config.get('vit') != EXPECTED_VIT:
        errors.append('unexpected ViT encoder')
    if config.get('cnn') != EXPECTED_CNN:
        errors.append('unexpected CNN encoder')
    if config.get('pretrained') is not True:
        errors.append('pretrained encoders must be enabled')
    if config.get('image_size') != 224:
        errors.append('publication image size must be 224')
    if errors:
        raise ValueError('; '.join(errors))
    return config


def sha256(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def run(config_path, output_path, device_name):
    import torch
    from common import save_json, seed_all
    from models import HybridDR, objective

    config_path = Path(config_path)
    output_path = Path(output_path)
    config = validate_config(json.loads(config_path.read_text(encoding='utf-8')))
    recipe = next(item for item in config['recipes'] if item['name'] == 'aligned_ce')
    device = torch.device(device_name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA device is required')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is required')

    seed_all(20261104)
    free_before, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    model = HybridDR(config, recipe, initialize=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    image = torch.zeros(1, 3, config['image_size'], config['image_size'], device=device)
    label = torch.tensor([2], device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(image)
        loss = objective(logits, label, recipe['loss'], 0, config['epochs'],
                         torch.ones(5, device=device))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss')
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite logits')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    result = {
        'purpose': 'publication Large/Large hardware preflight; not a research result',
        'verdict': 'PASS',
        'physical_gpu_id': os.environ.get('PHYSICAL_GPU_ID'),
        'device': torch.cuda.get_device_name(device),
        'torch': torch.__version__,
        'vit': config['vit'],
        'cnn': config['cnn'],
        'recipe': recipe['name'],
        'image_size': config['image_size'],
        'batch_size': 1,
        'parameters': parameters,
        'trainable_parameters': trainable,
        'loss': float(loss.detach().cpu()),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
        'free_vram_before_bytes': free_before,
        'total_vram_bytes': total,
        'elapsed_seconds': time.time() - started,
        'config_sha256': sha256(config_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device), indent=2,
                     sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
