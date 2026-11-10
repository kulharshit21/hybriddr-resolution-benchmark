"""Locked-checkpoint calibration and independent cohort evaluation."""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import bind_cpu, digest, read_json, save_json
from metrics import fit_temperature, measure
from models import HybridDR, probabilities
from train import loader, predict


def evaluate(job_path, prepared, out, device_name):
    job = read_json(job_path)
    config = job['config']
    bind_cpu([int(x) for x in os.environ['V2_CPU_IDS'].split(',')] if 'V2_CPU_IDS' in os.environ else job['cpu_ids'], config['threads_per_gpu'])
    config['batch_size']=min(8,config['batch_size'])
    out = Path(out)
    complete = read_json(out / 'complete.json')
    if digest(out / 'best.pt') != complete['checkpoint_sha256']:
        raise ValueError('Selected checkpoint modified')
    marker = out / 'evaluation.json'
    if marker.exists():
        if read_json(marker)['checkpoint_sha256'] != complete['checkpoint_sha256']:
            raise ValueError('Evaluation checkpoint mismatch')
        return
    device = torch.device(device_name)
    state = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    model = HybridDR(config, job['recipe'], initialize=False)
    model.load_state_dict(state['model'], strict=True)
    model.to(device)
    df = pd.read_csv(Path(prepared) / 'manifest.csv')
    ordinal = job['recipe']['loss'] == 'ordinal'
    cal = df[df.split == 'calibration']
    logits, labels = predict(model, loader(cal, prepared, config), device)
    temperature = fit_temperature(logits, labels, ordinal)
    cal_probs = probabilities(torch.tensor(logits), ordinal, temperature).numpy()
    referral = cal_probs[:, 2:].sum(1)
    # Freeze a development operating point targeting 95% sensitivity, no external retuning.
    threshold = float(np.quantile(referral[labels >= 2], .05)) if (labels >= 2).any() else .5
    scores = {}
    selected = df[(df.split == 'internal') | (df.split == 'external')]
    for (source, split), cohort in selected.groupby(['source', 'split']):
        cohort = cohort.reset_index(drop=True)
        logits, labels = predict(model, loader(cohort, prepared, config), device)
        raw = probabilities(torch.tensor(logits), ordinal).numpy()
        calibrated = probabilities(torch.tensor(logits), ordinal, temperature).numpy()
        key = source + '_' + split
        table = cohort[['path', 'source', 'patient', 'label']].copy()
        table['pred'] = raw.argmax(1)
        table['calibrated_pred'] = calibrated.argmax(1)
        for k in range(5):
            table[f'p{k}'] = raw[:, k]
            table[f'cal_p{k}'] = calibrated[:, k]
        table.to_csv(out / (key + '_predictions.csv'), index=False)
        referred = calibrated[:, 2:].sum(1) >= threshold
        positive = labels >= 2
        scores[key] = {'raw': measure(labels, raw), 'calibrated': measure(labels, calibrated),
                       'frozen_referral_sensitivity': float(referred[positive].mean()) if positive.any() else None,
                       'frozen_referral_specificity': float((~referred[~positive]).mean()) if (~positive).any() else None}
    save_json(marker, {'checkpoint_sha256': complete['checkpoint_sha256'], 'temperature': temperature,
                       'referral_threshold': threshold, 'scores': scores, 'smoke_only': config.get('tiny', False)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--prepared', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    evaluate(a.job, a.prepared, a.out, a.device)
