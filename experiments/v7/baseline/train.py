"""Single-GPU worker; scheduler assigns one independent experiment per device."""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import atomic_checkpoint, bind_cpu, digest, identity, read_json, save_json, seed_all
from data import RetinalDataset
from metrics import measure
from models import HybridDR, training_objective, probabilities


def process_memory_limit(total, config):
    requested = int(os.environ.get('V2_MEMORY_BUDGET_BYTES', total))
    return min(requested, int(total * config.get('vram_fraction', .85)))


def resolve_device(name):
    device = torch.device(name)
    return torch.device('cuda:0') if device.type == 'cuda' and device.index is None else device


def build_scheduler(optimizer, epochs, warmup_epochs):
    if not 0 <= warmup_epochs < epochs:
        raise ValueError('Warmup must be shorter than training')
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    def multiplier(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return .5 * (1 + math.cos(math.pi * min(progress, 1.)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def tune_batch(model, config, device, out):
    """Probe forward/backward; reserve FP32 Adam moments and leave VRAM headroom."""
    marker = out / 'batch_profile.json'
    if marker.exists() and 'V2_MEMORY_BUDGET_BYTES' not in os.environ:
        return read_json(marker)['selected_batch']
    if device.type != 'cuda' or not config.get('auto_batch'):
        return config['batch_size']
    records, selected = [], None
    total = torch.cuda.get_device_properties(device).total_memory
    budget = process_memory_limit(total, config)
    optimizer_reserve = sum(p.numel() * 8 for p in model.parameters() if p.requires_grad)
    for batch in (1, 2, 4, 8, 16, 32, 64):
        if batch > config['effective_batch_size']:
            break
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = logits = loss = None
        try:
            x = torch.zeros(batch, 3, config['image_size'], config['image_size'], device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = logits.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            fits = peak + optimizer_reserve < budget
            records.append({'batch': batch, 'peak_without_adam': peak, 'adam_reserve': optimizer_reserve, 'fits': fits})
            del x, logits, loss
            model.zero_grad(set_to_none=True)
            if not fits:
                break
            selected = batch
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            x = logits = loss = None
            records.append({'batch': batch, 'oom': True})
            break
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    if selected is None:
        raise RuntimeError('No microbatch fits VRAM headroom; reduce resolution/model before training')
    save_json(marker, {'selected_batch': selected, 'measurements': records, 'gpu': torch.cuda.get_device_name(device), 'process_budget_bytes': budget})
    return selected


def loader(frame, prepared, config, train=False, seed=0):
    ds = RetinalDataset(frame, prepared, config['image_size'], train)
    generator = torch.Generator().manual_seed(seed)
    workers = config['workers_per_gpu']
    options = {'prefetch_factor': 2, 'persistent_workers': False} if workers else {}
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=train, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=generator, **options)


@torch.no_grad()
def predict(model, data_loader, device):
    model.eval()
    outputs, labels = [], []
    for images, target, _ in data_loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(images.to(device, non_blocking=True))
        outputs.append(logits.float().cpu())
        labels.append(target)
    return torch.cat(outputs).numpy(), torch.cat(labels).numpy()


def train(job_path, prepared, out, device_name):
    job = read_json(job_path)
    config, recipe, seed = job['config'], job['recipe'], job['seed']
    cpu_ids = [int(value) for value in os.environ['V2_CPU_IDS'].split(',')] if 'V2_CPU_IDS' in os.environ else job['cpu_ids']
    bind_cpu(cpu_ids, config['threads_per_gpu'])
    seed_all(seed)
    device = resolve_device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 required; hardware preflight failed')
    if device.type == 'cuda' and 'V2_MEMORY_BUDGET_BYTES' in os.environ:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(process_memory_limit(total, config) / total, device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'runtime_cpu_ids.json', {'cpu_ids': cpu_ids})
    run_hash = identity({'job': job, 'manifest': digest(Path(prepared) / 'manifest.csv')})
    if (out / 'complete.json').exists():
        if read_json(out / 'complete.json')['run_hash'] != run_hash:
            raise RuntimeError('Completed run identity mismatch')
        return
    df = pd.read_csv(Path(prepared) / 'manifest.csv')
    training = df[df.split == 'train']
    validation = df[df.split == 'val']
    save_json(out / 'LIVE_PROGRESS.json', dict(stage='MODEL_INITIALIZATION'))
    print('MODEL_INITIALIZATION', recipe['name'], flush=True)
    model = HybridDR(config, recipe, initialize=not (out / 'last.pt').exists()).to(device)
    print('MICROBATCH_PROBING', flush=True)
    config['batch_size'] = tune_batch(model, config, device, out)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    seed_all(seed)
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith(('vit.', 'cnn.')) else heads).append(parameter)
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': config['learning_rate']},
                                  {'params': heads, 'lr': config['learning_rate'] * config['head_lr_multiplier']}],
                                 weight_decay=config['weight_decay'])
    scheduler = build_scheduler(optimizer, config['epochs'], recipe.get('warmup_epochs', 0))
    counts = torch.tensor(np.bincount(training.label, minlength=5), device=device, dtype=torch.float32)
    start, best, history = 0, -float('inf'), []
    if (out / 'last.pt').exists():
        state = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        if state['run_hash'] != run_hash:
            raise RuntimeError('Resume identity mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, best, history = state['epoch'] + 1, state['best'], state['history']
    elapsed_start = time.monotonic()
    for epoch in range(start, config['epochs']):
        epoch_started = time.monotonic()
        seed_all(seed + epoch)
        model.train()
        train_loader = loader(training, prepared, config, True, seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        total_loss, samples, accumulated = 0., 0, 0
        for step, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(images)
                loss = training_objective(logits, labels, recipe, epoch, config['epochs'], counts, config['label_smoothing'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            (loss * len(labels)).backward()
            accumulated += len(labels)
            samples += len(labels)
            total_loss += float(loss.detach()) * len(labels)
            if accumulated >= config['effective_batch_size'] or step == len(train_loader) - 1:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(accumulated)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
            if step % 10 == 0 or step == len(train_loader) - 1:
                save_json(out / 'LIVE_PROGRESS.json', dict(stage='TRAINING', epoch=epoch+1,
                    batch=step+1, total_batches=len(train_loader), samples=samples,
                    images_per_second=samples/max(time.monotonic()-epoch_started,1e-9)))
        training_seconds = time.monotonic() - epoch_started
        validation_started = time.monotonic()
        save_json(out / 'LIVE_PROGRESS.json', dict(stage='VALIDATION', epoch=epoch+1))
        logits, labels = predict(model, loader(validation, prepared, config), device)
        validation_seconds = time.monotonic() - validation_started
        probs = probabilities(torch.from_numpy(logits), recipe['loss'] == 'ordinal').numpy()
        scores = {}
        grade_details = {}
        for source in sorted(validation.source.unique()):
            mask = validation.source.to_numpy() == source
            scores[source] = measure(labels[mask], probs[mask])
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, support = precision_recall_fscore_support(
                labels[mask], probs[mask].argmax(1), labels=list(range(5)), zero_division=0)
            grade_details[source] = dict(precision=precision.tolist(), recall=recall.tolist(),
                f1=f1.tolist(), support=support.tolist(),
                prediction_counts=np.bincount(probs[mask].argmax(1), minlength=5).tolist())
        score = float(np.mean([s['qwk'] for s in scores.values() if s['qwk'] is not None]))
        if not np.isfinite(score):
            raise ValueError('Undefined validation selection score')
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': total_loss / samples,
                        'mean_source_val_qwk': score, 'validation': scores,
                        'training_seconds': training_seconds,
                        'validation_seconds': validation_seconds,
                        'training_images_per_second': samples / max(training_seconds, 1e-9),
                        'microbatch_size': config['batch_size']})
        history[-1]['validation_per_grade'] = grade_details
        history[-1]['mean_source_grade1_recall'] = float(np.mean([g['recall'][1] for g in grade_details.values()]))
        if score > best:
            best = score
            np.savez_compressed(out / 'best_validation_private.npz', labels=labels, probabilities=probs,
                                sources=validation.source.to_numpy(dtype=str), epoch=epoch)
            atomic_checkpoint(out / 'best.pt', {'model': model.state_dict(), 'config': config, 'recipe': recipe,
                                                'epoch': epoch, 'run_hash': run_hash, 'seed': seed})
        atomic_checkpoint(out / 'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best': best,
                                            'history': history, 'run_hash': run_hash})
        save_json(out / 'history.json', history)
        # Private development predictions, refreshed after each validation epoch.
        np.savez_compressed(out / 'latest_validation_private.npz', labels=labels, probabilities=probs)
        print(f"grade1_recall={history[-1]['mean_source_grade1_recall']:.4f}", flush=True)
        print(f"epoch={epoch} loss={total_loss / samples:.5f} validation_qwk={score:.5f} "
              f"train_images_per_second={samples / max(training_seconds, 1e-9):.3f} "
              f"microbatch={config['batch_size']}", flush=True)
    save_json(out / 'complete.json', {'run_hash': run_hash, 'best_validation_qwk': best,
                                     'wall_seconds_this_invocation': time.monotonic() - elapsed_start,
                                     'checkpoint_sha256': digest(out / 'best.pt'),
                                     'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                                     'smoke_only': config.get('tiny', False)})
    save_json(out / 'LIVE_PROGRESS.json', dict(stage='COMPLETE', epoch=config['epochs']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--prepared', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    train(args.job, args.prepared, args.out, args.device)
