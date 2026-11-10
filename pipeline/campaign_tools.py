#!/usr/bin/env python3
"""Validate pilot/campaign artifacts and report resumable progress."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def expected_runs(config):
    return [f"{recipe['name']}_seed{seed}"
            for recipe in config['recipes'] for seed in config['seeds']]


def inspect(root, config_path, require_evaluation=False):
    root, config_path = Path(root), Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    runs = []
    for name in expected_runs(config):
        run = root / 'runs' / name
        complete_path = run / 'complete.json'
        history_path = run / 'history.json'
        evaluation_path = run / 'evaluation.json'
        record = {'run': name, 'training_complete': complete_path.is_file(),
                  'evaluation_complete': evaluation_path.is_file()}
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding='utf-8'))
            history = json.loads(history_path.read_text(encoding='utf-8'))
            if len(history) != config['epochs']:
                raise RuntimeError(f'{name}: expected {config["epochs"]} epochs, got {len(history)}')
            checkpoint = run / 'best.pt'
            if digest(checkpoint) != complete['checkpoint_sha256']:
                raise RuntimeError(f'{name}: selected checkpoint hash mismatch')
            record.update(epochs=len(history), best_validation_qwk=complete['best_validation_qwk'],
                          wall_seconds=complete['wall_seconds_this_invocation'],
                          peak_cuda_bytes=complete['peak_cuda_bytes'],
                          microbatch_size=history[-1].get('microbatch_size'),
                          last_epoch_images_per_second=history[-1].get('training_images_per_second'))
        if require_evaluation and not evaluation_path.is_file():
            raise RuntimeError(f'{name}: evaluation is missing')
        runs.append(record)
    complete = sum(item['training_complete'] for item in runs)
    evaluated = sum(item['evaluation_complete'] for item in runs)
    return {'profile': config['profile'], 'expected_runs': len(runs),
            'completed_training_runs': complete, 'completed_evaluations': evaluated,
            'all_training_complete': complete == len(runs),
            'all_evaluations_complete': evaluated == len(runs), 'runs': runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['status', 'validate-pilot', 'validate-campaign'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    require_evaluation = args.command == 'validate-campaign'
    result = inspect(args.root, args.config, require_evaluation=require_evaluation)
    if args.command == 'validate-pilot':
        if not result['all_training_complete']:
            raise RuntimeError('Full-data pilot is incomplete')
        for run in result['runs']:
            if not run.get('last_epoch_images_per_second') or run['last_epoch_images_per_second'] <= 0:
                raise RuntimeError(f"{run['run']}: throughput evidence is missing")
    if args.command == 'validate-campaign':
        if not result['all_training_complete'] or not result['all_evaluations_complete']:
            raise RuntimeError('Campaign is incomplete')
        for name in ('results.csv', 'summary_by_recipe.csv', 'primary_contrasts.json', 'REPORT.md'):
            if not (args.root / name).is_file():
                raise RuntimeError(f'Missing report artifact: {name}')
    result['verdict'] = 'PASS' if args.command != 'status' else 'STATUS'
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
