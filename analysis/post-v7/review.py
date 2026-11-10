"""Read-only evidence review. Uses standard library; never starts training."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

SEEDS = (11, 23, 42, 67, 101)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def predictions(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        key = row['path']
        if key in out:
            raise ValueError('Duplicate prediction path: ' + key)
        label = int(row['label'])
        p = [float(row['p' + str(k)]) for k in range(5)]
        if label not in range(5) or any(not math.isfinite(x) or x < 0 or x > 1 for x in p) or abs(sum(p)-1) > 1e-4:
            raise ValueError('Invalid label/probabilities: ' + key)
        out[key] = (label, row['source'], row.get('patient', ''))
    if not out:
        raise ValueError('Empty predictions: ' + str(path))
    return out

def paired(values):
    if len(values) != 5 or any(not math.isfinite(x) for x in values):
        raise ValueError('Need exactly five finite paired seed differences')
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    margin = 2.7764451051977987 * sd / math.sqrt(5)
    return dict(seed_deltas=values, mean_delta=mean, sd_delta=sd,
                seed_t95_assumption_sensitive=[mean-margin, mean+margin],
                positive_seeds=sum(x > 0 for x in values),
                limitation='Training randomness only; not patient uncertainty, not multiplicity-adjusted, not confirmatory.')

def review(project):
    v6 = project/'runs/v6-resolution-final-01'
    v7 = project/'runs/v7-controls-448-01'
    missing = []
    artifacts = {}
    groups = {}
    reference_rows = {}
    for architecture, root in [('aligned', v6), ('cnn', v7), ('gated', v7)]:
        groups[architecture] = {}
        for seed in SEEDS:
            run = root/'runs'/f'{architecture}_ce_r448_seed{seed}'
            required = [run/'complete.json', run/'evaluation.json', run/'history.json']
            if any(not p.is_file() for p in required):
                missing.append(str(run))
                continue
            complete, evaluation, history = [read(p) for p in required]
            if len(history) != 30 or complete.get('smoke_only') or evaluation.get('smoke_only'):
                raise ValueError('Invalid training completion: ' + str(run))
            if not complete.get('checkpoint_sha256') or evaluation['checkpoint_sha256'] != complete['checkpoint_sha256']:
                raise ValueError('Checkpoint identity mismatch: ' + str(run))
            scores = evaluation['scores']
            if not scores:
                raise ValueError('Empty cohort registry')
            for cohort in scores:
                path = run/(cohort+'_predictions.csv')
                rows = predictions(path)
                if cohort in reference_rows and rows != reference_rows[cohort]:
                    raise ValueError('Unpaired image/label/source/patient identities: ' + str(path))
                reference_rows[cohort] = rows
                artifacts[str(path)] = digest(path)
            for p in required:
                artifacts[str(p)] = digest(p)
            groups[architecture][seed] = scores
    result = dict(status='WAITING' if missing else 'READY_FOR_PROVENANCE_REVIEW', missing_runs=missing,
                  artifacts_sha256=artifacts, checkpoint_binary_hashes_recomputed=False,
                  provenance_verified=False, patient_cluster_inference_performed=False,
                  interpretation='Exploratory; interim V7 validation already inspected. No automatic superiority claim.')
    if missing:
        return result
    expected = set(groups['aligned'][SEEDS[0]])
    if any(set(scores) != expected for runs in groups.values() for scores in runs.values()):
        raise ValueError('Cohort registry mismatch')
    result['cohort_images'] = {k: len(v) for k, v in reference_rows.items()}
    result['contrasts'] = []
    for cohort in sorted(expected):
        for control in ('cnn', 'gated'):
            for metric in ('qwk', 'macro_f1', 'balanced_accuracy', 'accuracy', 'mae'):
                values = []
                for seed in SEEDS:
                    a = groups['aligned'][seed][cohort]['raw']
                    b = groups[control][seed][cohort]['raw']
                    if metric not in a or metric not in b:
                        break
                    values.append(float(a[metric])-float(b[metric]))
                if len(values) == 5:
                    result['contrasts'].append(dict(cohort=cohort, metric=metric,
                        contrast='aligned minus '+control, seeds=list(SEEDS), **paired(values)))
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output file; preserve previous reviews')
    result = review(a.project)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(result['status'])
    print('Missing runs:', len(result['missing_runs']))
    print('Saved:', a.out)

if __name__ == '__main__':
    main()
