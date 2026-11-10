"""Read completed HybridDR artifacts and write supplemental publication evidence."""
import argparse
import hashlib
from pathlib import Path
import sys

import torch  # Load before SciPy-linked libraries on Windows.
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def checked_predictions(path, cohort):
    frame = pd.read_csv(path)
    columns = ['path', 'source', 'patient', 'label']
    pd.testing.assert_frame_equal(frame[columns], cohort[columns].reset_index(drop=True))
    if frame.path.duplicated().any() or not frame.label.isin(range(5)).all():
        raise ValueError(f'Invalid prediction rows: {path}')
    for prefix, name in [('p', 'pred'), ('cal_p', 'calibrated_pred')]:
        scores = frame[[f'{prefix}{i}' for i in range(5)]].to_numpy()
        if (not np.isfinite(scores).all() or (scores < 0).any() or
                not np.allclose(scores.sum(axis=1), 1, atol=1e-5) or
                not np.array_equal(scores.argmax(axis=1), frame[name])):
            raise ValueError(f'Invalid probabilities or grades: {path}')
    return frame


def grading_rows(frame, prediction):
    labels, grades = frame.label, frame[prediction]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, grades, labels=list(range(5)), zero_division=0)
    overall = {'accuracy': float((labels == grades).mean()),
               'weighted_f1': float(np.average(f1, weights=support)), 'n': len(frame)}
    per_grade = [dict(grade=i, support=int(support[i]), precision=precision[i],
                      recall=recall[i], f1=f1[i]) for i in range(5)]
    return overall, per_grade


def metrics(runs, cohorts, out):
    overall, per_grade = [], []
    for profile, name, run, evaluation in runs:
        recipe, seed = name.rsplit('_seed', 1)
        for key, cohort in cohorts.items():
            frame = checked_predictions(run / f'{key}_predictions.csv', cohort)
            for mode, column in [('raw', 'pred'), ('calibrated', 'calibrated_pred')]:
                meta = dict(profile=profile, recipe=recipe, seed=int(seed), cohort=key,
                            mode=mode, checkpoint_sha256=evaluation['checkpoint_sha256'])
                summary, grades = grading_rows(frame, column)
                overall.append(dict(meta, **summary))
                per_grade.extend(dict(meta, **grade) for grade in grades)
    for filename, records, fields, grouping in [
        ('overall', overall, ['accuracy', 'weighted_f1'], []),
        ('per_grade', per_grade, ['precision', 'recall', 'f1'], ['grade']),
    ]:
        frame = pd.DataFrame(records)
        frame.to_csv(out / f'{filename}_by_seed.csv', index=False)
        groups = ['profile', 'recipe', 'cohort', 'mode'] + grouping
        frame.groupby(groups)[fields].agg(['mean', 'std', 'count']).to_csv(
            out / f'{filename}_summary.csv')


def sample_cases(frame, per_group):
    # Stratify by reference grade and error category. Stable hash ordering avoids
    # selecting cases after inspecting maps. A short stratum uses all its cases.
    frame = frame.assign(error=(frame.pred - frame.label).abs())
    frame = frame.assign(category=np.where(frame.error == 0, 'correct',
                                np.where(frame.error == 1, 'adjacent', 'two_or_more')))
    frame = frame.assign(order=frame.path.map(
        lambda path: hashlib.sha256(f'20261106:{path}'.encode()).hexdigest()))
    return frame.sort_values('order').groupby(['label', 'category']).head(per_group)


def gradcam_pp(features, gradients):
    # First-gradient-power implementation of Grad-CAM++ weighting. The
    # positive common exponential score factor cancels during normalization.
    squared = gradients.square()
    denominator = 2 * squared + features.sum(dim=(2, 3), keepdim=True) * squared * gradients
    valid = denominator.abs() > torch.finfo(denominator.dtype).tiny
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    alpha = torch.where(valid, squared / safe, torch.zeros_like(squared))
    weights = (alpha * gradients.relu()).sum(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1).relu()
    if not torch.isfinite(cam).all():
        raise ValueError('Non-finite Grad-CAM++ map')
    cam = cam - cam.amin()
    return cam / cam.amax().clamp_min(torch.finfo(cam.dtype).tiny)


def explain(run, evaluation, cohorts, prepared, out, device, per_group):
    from math import isqrt
    from PIL import Image
    from common import digest
    from data import RetinalDataset
    from models import HybridDR

    checkpoint = run / 'best.pt'
    if digest(checkpoint) != evaluation['checkpoint_sha256']:
        raise ValueError('Checkpoint digest mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = dict(state['config'], gradient_checkpointing=False)
    if state['recipe']['loss'] != 'ce' or state['recipe']['architecture'] not in ('cnn', 'gated', 'spatial'):
        raise ValueError('Expected a CNN, gated, or spatial CE checkpoint with a CNN feature branch')
    model = HybridDR(config, state['recipe'], initialize=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.to(device).eval().requires_grad_(False)
    captured = []

    def capture(_, inputs):
        captured.append(inputs[0])

    handle = model.cproj.register_forward_pre_hook(capture)
    records = []
    try:
        for key, cohort in cohorts.items():
            frame = checked_predictions(run / f'{key}_predictions.csv', cohort)
            cases = sample_cases(frame, per_group)
            dataset = RetinalDataset(cohort, prepared, config['image_size'], False)
            for index, row in cases.iterrows():
                tensor, _, _ = dataset[index]
                captured.clear()
                logits = model(tensor.unsqueeze(0).to(device).requires_grad_(True))
                tokens, = captured
                gradient, = torch.autograd.grad(logits[0, int(row.pred)], tokens)
                side = isqrt(tokens.shape[1])
                features = tokens.detach().transpose(1, 2).reshape(1, -1, side, side)
                gradient = gradient.detach().transpose(1, 2).reshape_as(features)
                heat = gradcam_pp(features.float(), gradient.float())[0].cpu().numpy()
                stem = f'{key}_{len(records):03d}'
                np.save(out / f'{stem}.npy', heat)
                rgb = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                mask = Image.fromarray((heat * 128).round().astype(np.uint8)).resize(rgb.size, Image.Resampling.BILINEAR)
                panel = Image.new('RGB', (rgb.width * 2, rgb.height))
                panel.paste(rgb, (0, 0))
                panel.paste(Image.composite(Image.new('RGB', rgb.size, 'red'), rgb, mask), (rgb.width, 0))
                panel.save(out / f'{stem}.png')
                records.append(dict(case=stem, cohort=key, path=row.path, label=row.label,
                                    target_grade=row.pred, fp32_prediction=int(logits.argmax()),
                                    category=row.category, grid_side=side, empty_map=not heat.any()))
    finally:
        handle.remove()
    pd.DataFrame(records).to_csv(out / 'case_ledger_private.csv', index=False)
    print(f'Grad-CAM++ maps written: {len(records)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['metrics', 'cam'])
    for flag in ('pipeline', 'campaign', 'prepared', 'out'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--per-stratum', type=int, default=1)
    args = parser.parse_args()
    if args.per_stratum < 1:
        parser.error('--per-stratum must be positive')
    sys.path.insert(0, str(args.pipeline.resolve()))
    from common import digest, read_json, save_json

    if not (args.campaign / 'FINAL_CAMPAIGN_COMPLETE').is_file():
        raise ValueError('Wait for final campaign completion')
    manifest_path = args.prepared / 'manifest.csv'
    manifest_hash = digest(manifest_path)
    manifest = pd.read_csv(manifest_path)
    cohorts = {f'{source}_{split}': frame.reset_index(drop=True)
               for (source, split), frame in manifest[manifest.split.isin(['internal', 'external'])]
               .groupby(['source', 'split'])}
    if set(cohorts) != {'EyePACS_internal', 'APTOS_internal', 'MESSIDOR2_external'}:
        raise ValueError('Unexpected publication cohorts')
    runs = []
    for profile, config_name in [('primary-large-large', 'config_final.json'),
                                 ('scale-base-small', 'config_scale_ablation_final.json')]:
        root = args.campaign / profile
        config = read_json(args.pipeline / config_name)
        if config != read_json(root / 'config.json'):
            raise ValueError(f'Frozen configuration mismatch: {profile}')
        roster = read_json(root / 'evaluation_roster.json')
        expected = {f"{recipe['name']}_seed{seed}" for recipe in config['recipes'] for seed in config['seeds']}
        if set(roster['runs']) != expected or roster['manifest_sha256'] != manifest_hash:
            raise ValueError(f'Roster or manifest mismatch: {profile}')
        for name in sorted(expected):
            run = root / 'runs' / name
            evaluation = read_json(run / 'evaluation.json')
            if (evaluation['smoke_only'] or set(evaluation['scores']) != set(cohorts) or
                    evaluation['checkpoint_sha256'] != roster['runs'][name] or
                    evaluation['checkpoint_sha256'] != read_json(run / 'complete.json')['checkpoint_sha256']):
                raise ValueError(f'Evaluation identity mismatch: {name}')
            runs.append((profile, name, run, evaluation))
    if len(runs) != 75:
        raise ValueError('Expected 75 completed research runs')
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = dict(stage=args.stage, manifest_sha256=manifest_hash, evaluated_runs=len(runs),
                    script_sha256=digest(__file__))
    if args.stage == 'metrics':
        metrics(runs, cohorts, args.out)
    else:
        _, name, run, evaluation = next(item for item in runs if item[:2] == ('primary-large-large', 'aligned_ce_seed42'))
        explain(run, evaluation, cohorts, args.prepared, args.out, args.device, args.per_stratum)
        evidence.update(run=name, checkpoint_sha256=evaluation['checkpoint_sha256'],
                        target='saved raw predicted grade logit', layer='CNN features before cproj',
                        precision='FP32', per_stratum=args.per_stratum)
    save_json(args.out / 'POSTPROCESS_COMPLETE.json', evidence)
    print(f'Completed {args.stage}: {args.out}')


if __name__ == '__main__':
    main()


