"""Create an immutable, quality-filtered child preparation for retinal auditing."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import pandas as pd

from common import digest, save_json
from near_duplicates import AuditConfig, fingerprint_one


def _quality_task(task):
    relative, absolute, expected_sha256, config_values = task
    absolute = Path(absolute)
    actual = digest(absolute)
    if actual != expected_sha256:
        raise RuntimeError(f'Source digest mismatch: {absolute}')
    try:
        record = fingerprint_one(absolute, AuditConfig(config_values))
        return {'path': relative, 'status': 'valid',
                'input_sha256': record.input_sha256}
    except (ValueError, OSError) as exc:
        return {'path': relative, 'status': 'excluded',
                'input_sha256': actual, 'reason': 'retinal_quality_failure',
                'detail': str(exc)}


def run_quality(data_root: Path, prepared: Path, output: Path,
                config_path: Path, workers: int) -> dict:
    data_root, prepared, output = map(Path, (data_root, prepared, output))
    if output.exists():
        raise FileExistsError(f'Quality output already exists: {output}')
    manifest_path = prepared / 'manifest.csv'
    manifest = pd.read_csv(manifest_path)
    if manifest.sha256.isna().any():
        raise RuntimeError('Frozen manifest lacks source SHA-256 values')
    config = AuditConfig.from_json(config_path)
    output.mkdir(parents=True)
    (output / 'QUALITY_INCOMPLETE').write_text('quality audit in progress\n', encoding='utf-8')
    tasks = [(row.path, str(data_root / row.path), row.sha256, config.values)
             for row in manifest.itertuples()]
    evidence_tmp = output / 'quality_evidence.jsonl.tmp'
    records = []
    with evidence_tmp.open('w', encoding='utf-8') as handle:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for number, record in enumerate(pool.map(_quality_task, tasks, chunksize=4), 1):
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                if number % 100 == 0:
                    handle.flush()
                if number % 500 == 0 or number == len(tasks):
                    print(f'Quality images: {number}/{len(tasks)}', flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(evidence_tmp, output / 'quality_evidence.jsonl')
    excluded = [row for row in records if row['status'] == 'excluded']
    excluded_paths = {row['path'] for row in excluded}
    child = manifest[~manifest.path.isin(excluded_paths)].copy().reset_index(drop=True)
    child.to_csv(output / 'manifest.csv', index=False)
    inherited_path = prepared / 'exclusions.json'
    inherited = json.loads(inherited_path.read_text(encoding='utf-8')) \
        if inherited_path.is_file() else []
    save_json(output / 'exclusions.json', inherited + excluded)
    summary = {
        'parent_manifest_sha256': digest(manifest_path),
        'manifest_sha256': digest(output / 'manifest.csv'),
        'input_images': len(manifest), 'retained_images': len(child),
        'quality_exclusions': len(excluded),
        'quality_code_sha256': digest(Path(__file__)),
        'config_sha256': digest(config_path),
    }
    save_json(output / 'quality_summary.json', summary)
    source_cache = prepared / 'cache'
    if source_cache.exists():
        os.symlink(source_cache.resolve(), output / 'cache', target_is_directory=True)
    (output / 'QUALITY_INCOMPLETE').unlink()
    (output / 'QUALITY_COMPLETE').write_text(summary['manifest_sha256'] + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=min(160, os.cpu_count() or 1))
    args = parser.parse_args()
    print(run_quality(args.data, args.prepared, args.out, args.config, args.workers), flush=True)


if __name__ == '__main__':
    main()
