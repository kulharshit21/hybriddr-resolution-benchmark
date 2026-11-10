"""Run on the laptop, then again on DGX; no scientific performance claim."""
import argparse
import gc
import time
from pathlib import Path
import torch
from common import read_json, save_json, seed_all
from models import HybridDR, objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--tiny', action='store_true')
    args = p.parse_args()
    cfg = read_json(Path(__file__).parent / 'config.json')
    cfg.update(pretrained=False, tiny=args.tiny)
    recipe = next(r for r in cfg['recipes'] if r['name'] == 'aligned_ce')
    torch.set_num_threads(2)
    seed_all(42)
    device = torch.device(args.device)
    model = HybridDR(cfg, recipe).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for step in range(3):
        x = torch.rand(2, 3, cfg['image_size'], cfg['image_size'], device=device)
        y = torch.tensor([1, 4], device=device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(x)
            loss = objective(logits, y, 'ce', 0, 30, torch.ones(5, device=device))
        loss.backward()
        grad = model.vexchange.attention.in_proj_weight.grad
        assert torch.isfinite(grad).all() and grad.norm() > 0
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if step:
            samples.append(time.perf_counter() - started)
    result = {'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
              'torch': torch.__version__, 'tiny': args.tiny, 'pretrained': False,
              'steps': 3, 'warmup_steps': 1, 'measured_step_seconds': samples,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
              'parameters': sum(p.numel() for p in model.parameters()), 'finite_gradient': True,
              'purpose': 'Random-input hardware/correctness check, not research training'}
    save_json(args.out, result)
    print(result)
    print('VERDICT: PASS')


if __name__ == '__main__':
    main()
