"""Single-command, resumable data -> experiment queue -> locked evaluation -> reports."""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import pandas as pd
import psutil

from common import digest, identity, read_json, resources, save_json

HERE = Path(__file__).resolve().parent


def configure_smoke(config, gpu_id=0):
    """Apply the non-research smoke profile to one explicit physical GPU."""
    if gpu_id < 0:
        raise ValueError('Smoke GPU ID must be non-negative')
    config.update(tiny=True, pretrained=False, auto_batch=False, epochs=2,
                  batch_size=8, effective_batch_size=16, image_size=56,
                  cache_size=64, workers_per_gpu=0, threads_per_gpu=2,
                  audit_workers=2, seeds=[42], gpu_ids=[gpu_id],
                  bootstrap_repeats=20)
    config['recipes'] = [config['recipes'][0], config['recipes'][-1]]
    for recipe in config['recipes']:
        if recipe.get('warmup_epochs', 0):
            recipe['warmup_epochs'] = 1
    return config


def parse_gpu_ids(value):
    try:
        gpu_ids = [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise ValueError('GPU IDs must be comma-separated non-negative integers') from error
    if not gpu_ids or any(item < 0 for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('GPU IDs must be unique non-negative integers')
    return gpu_ids


def validate_data_review(summary, review):
    if review.get('approved') is not True:
        raise RuntimeError('Data review is not approved')
    for key in ('manifest_sha256', 'parent_manifest_sha256', 'audit_id'):
        if not summary.get(key) or review.get(key) != summary.get(key):
            raise RuntimeError(f'Data review {key} is missing or does not match')


def queue(jobs, prepared, root, config, hardware, stage, cpu=False, gpu_ids=None):
    assigned = config['gpu_ids'] if gpu_ids is None else gpu_ids
    devices = ['cpu'] if cpu else [str(g) for g in assigned]
    if not cpu and any(int(g) >= hardware['gpu_count'] for g in devices):
        raise RuntimeError('Requested GPU unavailable; use a smoke profile or fix gpu_ids')
    running, pending = {}, list(jobs)
    try:
        while pending or running:
            for gpu in devices:
                if gpu in running or not pending:
                    continue
                if psutil.virtual_memory().used > hardware['ram_budget_bytes']:
                    if not running:
                        raise RuntimeError('RAM budget exceeded before launch')
                    continue
                job = pending.pop(0)
                name = job['recipe']['name'] + '_seed' + str(job['seed'])
                out = root / 'runs' / name
                out.mkdir(parents=True, exist_ok=True)
                jobfile = root / 'jobs' / (name + '.json')
                if jobfile.exists() and read_json(jobfile) != job:
                    raise RuntimeError('Frozen job changed: ' + name)
                save_json(jobfile, job)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = '' if cpu else gpu
                env['OMP_NUM_THREADS'] = str(config['threads_per_gpu'])
                env['MKL_NUM_THREADS'] = str(config['threads_per_gpu'])
                slot = devices.index(gpu)
                runtime_cpus = hardware['cpu_ids'][slot::len(devices)] or hardware['cpu_ids'][:1]
                env['V2_CPU_IDS'] = ','.join(map(str, runtime_cpus))
                log = open(out / (stage + '.log'), 'a', encoding='utf-8')
                command = [sys.executable, str(HERE / (stage + '.py')), '--job', str(jobfile),
                           '--prepared', str(prepared), '--out', str(out), '--device', 'cpu' if cpu else 'cuda']
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = (process, log, name)
                print(f'{stage}: launched {name} on {gpu}', flush=True)
            for gpu, (process, log, name) in list(running.items()):
                rc = process.poll()
                if rc is not None:
                    log.close()
                    del running[gpu]
                    if rc:
                        raise RuntimeError(f'{stage} failed: {name}; see runs/{name}/{stage}.log')
                    print(f'{stage}: completed {name}', flush=True)
            if running:
                time.sleep(2)
    finally:
        for process, log, _ in running.values():
            process.terminate()
            process.wait()
            log.close()


def jobs_for(config, hardware, slots=None):
    slots = max(1, slots if slots is not None else len(config['gpu_ids']))
    # Every job uses the same bounded CPU allocation: no per-GPU multiplication of 179 cores.
    # Allocation is based on job index; concurrent waves retain a stable five-slot sequence.
    jobs = []
    index = 0
    for recipe in config['recipes']:
        for seed in config['seeds']:
            cpus = hardware['cpu_ids'][index % slots::slots] or hardware['cpu_ids'][:1]
            local = copy.deepcopy(config)
            # Physical GPU assignment is runtime state, not scientific job identity.
            local.pop('gpu_ids', None)
            local['workers_per_gpu'] = min(config['workers_per_gpu'], max(0, len(cpus) - config['threads_per_gpu']))
            jobs.append({'config': local, 'recipe': recipe, 'seed': seed, 'cpu_ids': cpus})
            index += 1
    return jobs


def summarize(root, config):
    from metrics import paired_bootstrap
    rows = []
    for path in sorted((root / 'runs').glob('*/evaluation.json')):
        record = read_json(path)
        for cohort, scores in record['scores'].items():
            rows.append({'run': path.parent.name, 'cohort': cohort,
                         **{k: v for k, v in scores['raw'].items() if not isinstance(v, list)}})
    results = pd.DataFrame(rows)
    results.to_csv(root / 'results.csv', index=False)
    results['recipe'] = results.run.str.rsplit('_seed', n=1).str[0]
    results.groupby(['recipe', 'cohort'])[['qwk', 'macro_f1', 'mae', 'nll', 'ece_15_equal_width']].agg(['mean', 'std', 'count']).to_csv(root / 'summary_by_recipe.csv')
    # Main contrast chosen exclusively from validation before external release.
    roster = read_json(root / 'evaluation_roster.json')
    comparator, candidate = roster['comparator'], roster['candidate']
    seed = 42 if 42 in config['seeds'] else config['seeds'][0]
    contrasts = {}
    for path in sorted((root / 'runs' / f'{candidate}_seed{seed}').glob('*_predictions.csv')):
        other = root / 'runs' / f'{comparator}_seed{seed}' / path.name
        contrasts[path.stem] = paired_bootstrap(pd.read_csv(path), pd.read_csv(other), config['bootstrap_repeats'])
    save_json(root / 'primary_contrasts.json', contrasts)
    text = ['# Generated experiment report', '',
            'SMOKE TEST ONLY — NOT RESEARCH RESULTS' if config.get('tiny') else 'Results from the frozen run registry.',
            '', f'Primary candidate: {candidate}; validation-selected comparator: {comparator}; prespecified seed: {seed}.',
            '', 'See results.csv for per-run outcomes and primary_contrasts.json for paired bootstrap intervals.',
            'Bootstrap intervals quantify cohort sampling, not training-seed uncertainty.',
            'APTOS and MESSIDOR patient identity is unavailable; their bootstrap uses image identity.',
            'No prospective clinical validation. No claim of novel superiority follows from running this software.']
    (root / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    from data import prepare
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preflight', 'train'])
    parser.add_argument('--data', type=Path, default=HERE.parent / 'Dataset')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--prepared-source', type=Path,
                        help='verified prepared dataset to bind instead of running prepare')
    parser.add_argument('--gpu-ids',
                        help='comma-separated physical GPU IDs for non-smoke execution')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-gpu', type=int, default=0,
                        help='physical GPU ID for the one-GPU smoke run')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--prior-campaign', type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config.get('study_version') != 2:
        raise ValueError('Expected isolated version-2 configuration')
    prior = args.prior_campaign.resolve()
    if not (prior / 'FINAL_CAMPAIGN_COMPLETE').is_file():
        raise RuntimeError('Wait for version-1 training and evaluation to finish')
    if args.out.resolve().is_relative_to(prior):
        raise ValueError('Version-2 output must be outside the version-1 campaign')
    runtime_gpu_ids = list(config['gpu_ids'])
    if args.smoke:
        if args.gpu_ids:
            raise ValueError('--gpu-ids cannot be combined with --smoke; use --smoke-gpu')
        configure_smoke(config, args.smoke_gpu)
    elif args.smoke_gpu != 0:
        raise ValueError('--smoke-gpu is only valid together with --smoke')
    elif args.gpu_ids:
        runtime_gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.smoke:
        runtime_gpu_ids = list(config['gpu_ids'])
    if args.prepared_source and args.stage == 'prepare':
        raise ValueError('--prepared-source cannot be combined with the prepare-only stage')
    if args.cpu and not (args.smoke or config.get('tiny')):
        raise ValueError('CPU training is restricted to smoke tests')
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    hardware = resources(config)
    save_json(root / 'hardware.json', hardware)
    if args.stage == 'preflight':
        if not args.cpu and any(gpu >= hardware['gpu_count'] for gpu in runtime_gpu_ids):
            raise RuntimeError('Insufficient GPUs for selected configuration')
        print(hardware)
        return
    frozen = root / 'config.json'
    if frozen.exists() and read_json(frozen) != config:
        raise RuntimeError('Configuration changed; use a new output directory')
    save_json(frozen, config)
    prepared = args.prepared_source.resolve() if args.prepared_source else root / 'prepared'
    if args.stage in ('prepare', 'all') and not args.prepared_source:
        print(prepare(args.data.resolve(), prepared, config, limit=24 if args.smoke else None), flush=True)
    if args.stage == 'prepare':
        return
    summary = read_json(prepared / 'prepared.json')
    if digest(prepared / 'manifest.csv') != summary['manifest_sha256']:
        raise RuntimeError('Manifest hash changed')
    binding = {'manifest_sha256': summary['manifest_sha256'],
               'prepared_metadata_sha256': digest(prepared / 'prepared.json'),
               'audit_id': summary.get('audit_id'),
               'parent_manifest_sha256': summary.get('parent_manifest_sha256')}
    binding_path = root / 'prepared_binding.json'
    if binding_path.exists() and read_json(binding_path) != binding:
        raise RuntimeError('Prepared dataset binding changed')
    save_json(binding_path, binding)
    if not args.smoke:
        review_path = root / 'DATA_REVIEW.json'
        if not review_path.exists():
            raise RuntimeError('Data review required: inspect near_duplicate_candidates.json and provenance; see README DATA_REVIEW.json schema')
        review = read_json(review_path)
        validate_data_review(summary, review)
    jobs = jobs_for(config, hardware, slots=len(runtime_gpu_ids))
    if args.stage in ('train', 'all'):
        queue(jobs, prepared, root, config, hardware, 'train', args.cpu,
              gpu_ids=runtime_gpu_ids)
    if args.stage in ('evaluate', 'all'):
        aggregates = {}
        for recipe in config['recipes']:
            values = [read_json(root / 'runs' / f"{recipe['name']}_seed{seed}" / 'complete.json')['best_validation_qwk'] for seed in config['seeds']]
            aggregates[recipe['name']] = sum(values) / len(values)
        controls = [r['name'] for r in config['recipes'] if r['architecture'] in ('concat', 'gated') and r['loss'] == 'ce']
        comparator = max(controls, key=lambda n: (aggregates[n], n))
        roster = {'candidate': 'aligned_ce', 'comparator': comparator,
                  'manifest_sha256': summary['manifest_sha256'], 'validation_aggregates': aggregates,
                  'runs': {j['recipe']['name'] + '_seed' + str(j['seed']): read_json(root / 'runs' / (j['recipe']['name'] + '_seed' + str(j['seed'])) / 'complete.json')['checkpoint_sha256'] for j in jobs}}
        roster_path = root / 'evaluation_roster.json'
        if roster_path.exists() and read_json(roster_path) != roster:
            raise RuntimeError('Frozen evaluation roster changed')
        save_json(roster_path, roster)
        queue(jobs, prepared, root, config, hardware, 'evaluate', args.cpu,
              gpu_ids=runtime_gpu_ids)
    if args.stage in ('report', 'all'):
        summarize(root, config)
    print('VERDICT: PASS', flush=True)


if __name__ == '__main__':
    main()
