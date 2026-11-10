"""CLI for calibrated, resumable retinal near-duplicate auditing."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import hashlib
import shutil

import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from common import digest, identity, save_json
from near_duplicates import (
    AuditConfig, AuditState, HammingIndex, PairEvidence,
    build_audit_identity, classify_pair, confirm_pair, fingerprint_one,
    dino_candidate_record, embed_retinas, exact_cosine_neighbors, load_dino,
    normalize_retina, perceptual_hash,
)


def wilson_upper(errors: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0 or not 0 <= errors <= total:
        raise ValueError('Wilson interval requires 0 <= errors <= total and total > 0')
    proportion = errors / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return (center + margin) / denominator


def calibration_binding(manifest: pd.DataFrame, config: AuditConfig) -> dict:
    development = manifest[manifest.role == 'development'].sort_values('path')
    columns = [column for column in development.columns if column != 'split']
    records = development[columns].where(pd.notna(development[columns]), None).to_dict('records')
    code = {
        'audit_duplicates.py': digest(Path(__file__)),
        'near_duplicates.py': digest(Path(__file__).with_name('near_duplicates.py')),
    }
    return {
        'config_sha256': identity(config.values),
        'development_manifest_sha256': identity(records),
        'calibration_code_sha256': identity(code),
    }


def lock_calibration(positive_distances: list[int], negative_evidence: list[PairEvidence],
                     config: AuditConfig, output: Path,
                     binding: dict | None = None,
                     positive_evidence: list[PairEvidence] | None = None,
                     negative_distances: list[int] | None = None) -> dict:
    if not positive_distances or not negative_evidence:
        raise ValueError('Calibration needs positive transformations and hard negatives')
    radius = int(config.phash['candidate_hamming_radius'])
    recall = sum(distance <= radius for distance in positive_distances) / len(positive_distances)
    if negative_distances is None:
        negative_distances = [0] * len(negative_evidence)
    if len(negative_distances) != len(negative_evidence):
        raise ValueError('Negative distances and evidence must align')
    false_confirmations = sum(
        distance <= radius and classify_pair(item, config.confirmation) == 'confirmed'
        for distance, item in zip(negative_distances, negative_evidence))
    upper = wilson_upper(false_confirmations, len(negative_evidence))
    confirmation_recall = None
    if positive_evidence is not None:
        if len(positive_evidence) != len(positive_distances):
            raise ValueError('Positive distances and evidence must align')
        confirmation_recall = sum(
            distance <= radius and classify_pair(item, config.confirmation) == 'confirmed'
            for distance, item in zip(positive_distances, positive_evidence)
        ) / len(positive_distances)
    target = config.calibration
    if recall < float(target['minimum_retrieval_recall']):
        raise RuntimeError(f'Calibration retrieval recall failed: {recall:.6f}')
    if (confirmation_recall is not None and confirmation_recall <
            float(target.get('minimum_confirmation_recall', 0.0))):
        raise RuntimeError(
            f'Calibration confirmation recall failed: {confirmation_recall:.6f}')
    if upper >= float(target['maximum_false_confirmation_upper_95']):
        raise RuntimeError(f'Calibration false-confirmation bound failed: {upper:.6f}')
    record = {
        'locked': True,
        'config_sha256': identity(config.values),
        'positive_transformations': len(positive_distances),
        'retrieval_recall': recall,
        'confirmation_recall': confirmation_recall,
        'hard_negative_pairs': len(negative_evidence),
        'false_confirmations': false_confirmations,
        'false_confirmation_upper_95': upper,
        'thresholds': config.confirmation,
    }
    if binding:
        record.update(binding)
    save_json(output, record)
    return record


def require_locked_calibration_before_external(manifest: pd.DataFrame,
                                               calibration_path: Path,
                                               config: AuditConfig) -> None:
    if not (manifest['role'] == 'external').any():
        return
    if not Path(calibration_path).is_file():
        raise RuntimeError('External cohort is sealed until calibration is locked')
    record = json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    if record.get('locked') is not True:
        raise RuntimeError('External cohort is sealed: calibration is not locked')
    expected = calibration_binding(manifest, config)
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f'External cohort is sealed: calibration {key} mismatch')


def _validate_source_task(task):
    path, expected = task
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f'Source digest mismatch: {path}')
    return str(path)


def validate_source_files(manifest: pd.DataFrame, data_root: Path,
                          workers: int) -> None:
    if 'sha256' not in manifest.columns or manifest.sha256.isna().any():
        raise RuntimeError('Frozen manifest lacks source SHA-256 values')
    tasks = [(str(Path(data_root) / row.path), row.sha256) for row in manifest.itertuples()]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        validated = list(pool.map(_validate_source_task, tasks, chunksize=8))
    if len(validated) != len(manifest):
        raise RuntimeError('Source validation count mismatch')


def load_resume_records(path: Path, key: str, audit_id: str) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    records = {}
    with path.open('r', encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Corrupt JSONL at {path}:{number}') from exc
            if record.get('audit_id') != audit_id:
                raise RuntimeError(f'Resume identity mismatch at {path}:{number}')
            if key not in record or record[key] in records:
                raise RuntimeError(f'Missing or duplicate {key} at {path}:{number}')
            records[record[key]] = record
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def _fingerprint_task(task):
    relative, absolute, config_values = task
    record = fingerprint_one(Path(absolute), AuditConfig(config_values))
    result = asdict(record)
    result['path'] = relative
    result['phash'] = f"{record.phash:064x}"
    result['phash_flipped'] = f"{record.phash_flipped:064x}"
    return result


def _confirmation_task(task):
    left_relative, right_relative, data_root, config_values = task
    config = AuditConfig(config_values)
    with Image.open(Path(data_root) / left_relative) as image:
        left = normalize_retina(image, config, left_relative)
    with Image.open(Path(data_root) / right_relative) as image:
        right = normalize_retina(image, config, right_relative)
    evidence = confirm_pair(left, right, config)
    return asdict(evidence), classify_pair(evidence, config.confirmation)


def _positive_calibration_task(task):
    relative, data_root, config_values = task
    config = AuditConfig(config_values)
    path = Path(data_root) / relative
    with Image.open(path) as image:
        original = image.convert('RGB')
    base = normalize_retina(original, config, relative)
    base_hashes = (perceptual_hash(base.gray, config),
                   perceptual_hash(base.gray[:, ::-1], config))
    records = []
    for transform_number, transformed in enumerate(_transformed_images(original)):
        other = normalize_retina(transformed, config,
                                 f'{relative}:synthetic:{transform_number}')
        hashes = (perceptual_hash(other.gray, config),
                  perceptual_hash(other.gray[:, ::-1], config))
        distance = min((left ^ right).bit_count()
                       for left in base_hashes for right in hashes)
        evidence = confirm_pair(base, other, config)
        records.append({'transform_number': transform_number,
                        'hamming_distance': distance,
                        'phash': f'{hashes[0]:064x}',
                        'phash_flipped': f'{hashes[1]:064x}',
                        'evidence': asdict(evidence),
                        'classification': classify_pair(evidence, config.confirmation)})
    return relative, base_hashes[0], records


def _hash_words(value: str) -> np.ndarray:
    integer = int(value, 16)
    return np.asarray([(integer >> shift) & ((1 << 64) - 1)
                       for shift in (192, 128, 64, 0)], dtype=np.uint64)


def _bounded_candidate_chunk(task):
    start, stop, paths, hashes, radius, top_k = task
    query = hashes[start:stop]
    xor = np.bitwise_xor(query[:, :, None, None, :],
                         hashes[None, None, :, :, :])
    distances = np.bitwise_count(xor).sum(axis=-1).transpose(0, 2, 1, 3)
    best_distance = distances.min(axis=(2, 3)).astype(np.int16)
    best_orientation = distances.reshape(stop - start, len(paths), 4).argmin(axis=2)
    result = []
    orientation_names = (('normal', 'normal'), ('normal', 'flipped'),
                         ('flipped', 'normal'), ('flipped', 'flipped'))
    for local, path_index in enumerate(range(start, stop)):
        row = best_distance[local]
        row[path_index] = 32767
        eligible = np.flatnonzero(row <= radius)
        if len(eligible) > top_k:
            kth = int(np.partition(row, top_k - 1)[top_k - 1])
            lower = np.flatnonzero(row < kth)
            equal = np.flatnonzero(row == kth)
            eligible = np.concatenate((lower, equal[:top_k - len(lower)]))
        selected = sorted(eligible.tolist(), key=lambda index: (int(row[index]), paths[index]))
        for other_index in selected:
            left, right = sorted((paths[path_index], paths[other_index]))
            query_orientation, other_orientation = orientation_names[
                int(best_orientation[local, other_index])]
            orientations = ([query_orientation, other_orientation]
                            if left == paths[path_index]
                            else [other_orientation, query_orientation])
            result.append({
                'pair_id': identity([left, right]), 'left': left, 'right': right,
                'hamming_distance': int(row[other_index]),
                'orientations': orientations,
            })
    return result


def _candidate_pairs(records: dict[str, dict], radius: int,
                     top_k: int | None = None, workers: int = 1,
                     chunk_size: int = 8) -> list[dict]:
    paths = sorted(records)
    if top_k is None:
        top_k = max(1, len(paths) - 1)
    hashes = np.empty((len(paths), 2, 4), dtype=np.uint64)
    for index, path in enumerate(paths):
        hashes[index, 0] = _hash_words(records[path]['phash'])
        hashes[index, 1] = _hash_words(records[path]['phash_flipped'])
    tasks = [(start, min(len(paths), start + chunk_size), paths, hashes, radius, top_k)
             for start in range(0, len(paths), chunk_size)]
    pairs = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        for task_number, candidates in enumerate(pool.map(_bounded_candidate_chunk, tasks), 1):
            for candidate in candidates:
                existing = pairs.get(candidate['pair_id'])
                if existing is None or candidate['hamming_distance'] < existing['hamming_distance']:
                    pairs[candidate['pair_id']] = candidate
            if task_number % 125 == 0 or task_number == len(tasks):
                processed = min(len(paths), task_number * chunk_size)
                print(f'Candidate retrieval images: {processed}/{len(paths)}', flush=True)
    return sorted(pairs.values(), key=lambda row: (
        row['hamming_distance'], row['left'], row['right']))


def _validate_bounded_positive_retrieval(fingerprints: dict[str, dict],
                                         positives_path: Path,
                                         config: AuditConfig,
                                         output: Path) -> dict:
    positives = [json.loads(line) for line in Path(positives_path).read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if not positives:
        raise RuntimeError('Bounded retrieval validation lacks positive evidence')
    paths = sorted(fingerprints)
    path_index = {path: index for index, path in enumerate(paths)}
    corpus = np.empty((len(paths), 2, 4), dtype=np.uint64)
    for index, path in enumerate(paths):
        corpus[index, 0] = _hash_words(fingerprints[path]['phash'])
        corpus[index, 1] = _hash_words(fingerprints[path]['phash_flipped'])
    radius = int(config.phash['candidate_hamming_radius'])
    top_k = int(config.phash['max_neighbors_per_image'])
    ranks = []
    passed = 0
    for record in positives:
        source = record['path']
        if source not in path_index:
            raise RuntimeError(f'Positive source missing from filtered manifest: {source}')
        query = np.stack((_hash_words(record['phash']),
                          _hash_words(record['phash_flipped'])))
        distances = np.bitwise_count(np.bitwise_xor(
            query[:, None, None, :], corpus[None, :, :, :])).sum(axis=-1)
        best = distances.min(axis=(0, 2)).astype(np.int16)
        source_distance = int(best[path_index[source]])
        rank = int(int(np.sum(best < source_distance)) +
                   sum(best[index] == source_distance and path < source
                       for index, path in enumerate(paths)) + 1)
        accepted = (source_distance <= radius and rank <= top_k and
                    record['classification'] == 'confirmed')
        passed += int(accepted)
        ranks.append({'path': source,
                      'transform_number': record['transform_number'],
                      'source_distance': source_distance, 'source_rank': rank,
                      'confirmation': record['classification'],
                      'accepted': accepted})
    recall = passed / len(positives)
    result = {
        'positive_queries': len(positives), 'accepted': passed,
        'bounded_end_to_end_recall': recall, 'radius': radius, 'top_k': top_k,
        'maximum_source_rank': max(row['source_rank'] for row in ranks),
    }
    save_json(Path(output) / 'bounded_positive_ranks.json', ranks)
    save_json(Path(output) / 'bounded_retrieval_validation.json', result)
    minimum = float(config.calibration['minimum_confirmation_recall'])
    if recall < minimum:
        raise RuntimeError(
            f'Bounded full-corpus retrieval recall failed: {recall:.6f} < {minimum:.6f}')
    print(f'Bounded full-corpus positive recall: {recall:.6f}', flush=True)
    return result


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _artifact(path: Path, lines: bool = False) -> dict:
    record = {'sha256': digest(path)}
    if lines:
        record['lines'] = sum(1 for line in path.read_text(encoding='utf-8').splitlines()
                              if line.strip())
    return record


def _verify_stage(output: Path, marker_name: str,
                  parent_manifest_sha256: str) -> dict:
    output = Path(output)
    marker = output / marker_name
    if not marker.is_file():
        raise RuntimeError(f'Missing completion marker: {marker_name}')
    stage = marker_name.removesuffix('_COMPLETE.json')
    for incomplete in output.glob('*INCOMPLETE*'):
        raise RuntimeError(f'Incomplete audit marker present: {incomplete.name}')
    record = json.loads(marker.read_text(encoding='utf-8'))
    if record.get('parent_manifest_sha256') != parent_manifest_sha256:
        raise RuntimeError(f'{stage} parent manifest mismatch')
    artifacts = record.get('artifacts')
    if not isinstance(artifacts, dict):
        raise RuntimeError(f'{stage} completion lacks artifact evidence')
    for relative, expected in artifacts.items():
        path = output / relative
        if not path.is_file():
            raise RuntimeError(f'{stage} artifact missing: {relative}')
        if digest(path) != expected.get('sha256'):
            raise RuntimeError(f'{stage} artifact hash mismatch: {relative}')
        if 'lines' in expected:
            actual = sum(1 for line in path.read_text(encoding='utf-8').splitlines()
                         if line.strip())
            if actual != expected['lines']:
                raise RuntimeError(f'{stage} artifact count mismatch: {relative}')
    return record


def run_cpu(data_root: Path, prepared: Path, output: Path,
            config_path: Path, workers: int) -> dict:
    data_root, prepared, output = map(Path, (data_root, prepared, output))
    config = AuditConfig.from_json(config_path)
    manifest_path = prepared / 'manifest.csv'
    manifest = pd.read_csv(manifest_path)
    calibration_path = output / 'calibration.json'
    require_locked_calibration_before_external(manifest, calibration_path, config)
    validate_source_files(manifest, data_root, workers)
    audit_identity = build_audit_identity(
        manifest_path, data_root, config,
        [Path(__file__), Path(__file__).with_name('near_duplicates.py')],
        Path(__file__).with_name('requirements.txt'), None)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / 'audit_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != audit_identity:
        raise RuntimeError('Audit output belongs to a different identity')
    save_json(identity_path, audit_identity)
    audit_id = audit_identity['audit_id']
    parent_hash = audit_identity['parent_manifest_sha256']
    complete_path = output / 'CPU_COMPLETE.json'
    if complete_path.exists() and not (output / 'CPU_INCOMPLETE').exists():
        completed_stage = _verify_stage(output, 'CPU_COMPLETE.json', parent_hash)
        return {key: value for key, value in completed_stage.items() if key != 'artifacts'}
    (output / 'CPU_INCOMPLETE').write_text(audit_id + '\n', encoding='utf-8')
    complete_path.unlink(missing_ok=True)
    fingerprints_path = output / 'fingerprints.jsonl'
    _touch(fingerprints_path)
    fingerprints = load_resume_records(fingerprints_path, 'path', audit_id)
    pending = [(row.path, str(data_root / row.path), config.values)
               for row in manifest.itertuples() if row.path not in fingerprints]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for result in pool.map(_fingerprint_task, pending, chunksize=4):
            result['audit_id'] = audit_id
            _append_jsonl(fingerprints_path, result)
            fingerprints[result['path']] = result
    if len(fingerprints) != len(manifest):
        raise RuntimeError(f'Fingerprint count mismatch: {len(fingerprints)} != {len(manifest)}')
    bounded_validation = None
    if calibration_path.is_file():
        bounded_validation = _validate_bounded_positive_retrieval(
            fingerprints, output / 'calibration_positive_evidence.jsonl',
            config, output)
    metadata = manifest.set_index('path')[['source', 'split', 'patient', 'label']].to_dict('index')
    candidates = _candidate_pairs(
        fingerprints, int(config.phash['candidate_hamming_radius']),
        top_k=int(config.phash['max_neighbors_per_image']),
        workers=min(workers, int(config.phash.get('retrieval_threads', workers))),
        chunk_size=int(config.phash.get('retrieval_chunk_size', 8)))
    maximum_candidates = int(config.phash['maximum_candidate_pairs'])
    print(f'Bounded candidate pairs: {len(candidates)} / {maximum_candidates}', flush=True)
    if len(candidates) > maximum_candidates:
        raise RuntimeError(
            f'Candidate-pair safety limit exceeded: {len(candidates)} > {maximum_candidates}')
    pairs_path = output / 'candidate_pairs.jsonl'
    _touch(pairs_path)
    completed = load_resume_records(pairs_path, 'pair_id', audit_id)
    pending_pairs = [pair for pair in candidates if pair['pair_id'] not in completed]
    tasks = [(pair['left'], pair['right'], str(data_root), config.values) for pair in pending_pairs]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for pair, (evidence, classification) in zip(pending_pairs, pool.map(
                _confirmation_task, tasks, chunksize=1)):
            left_meta, right_meta = metadata[pair['left']], metadata[pair['right']]
            record = {
                **pair, 'audit_id': audit_id, 'evidence': evidence,
                'classification': classification,
                'protected_boundary': left_meta['split'] != right_meta['split'],
                'cross_source': left_meta['source'] != right_meta['source'],
                'sources': sorted({left_meta['source'], right_meta['source']}),
                'splits': sorted({left_meta['split'], right_meta['split']}),
            }
            _append_jsonl(pairs_path, record)
            completed[pair['pair_id']] = record
    if len(completed) != len(candidates):
        raise RuntimeError(f'Candidate count mismatch: {len(completed)} != {len(candidates)}')
    counts = pd.Series([row['classification'] for row in completed.values()]).value_counts().to_dict()
    summary = {
        'audit_id': audit_id,
        'parent_manifest_sha256': audit_identity['parent_manifest_sha256'],
        'fingerprints': len(fingerprints),
        'candidate_pairs': len(candidates),
        'protected_boundary_candidates': sum(bool(row['protected_boundary'])
                                             for row in completed.values()),
        'cross_source_candidates': sum(bool(row['cross_source'])
                                       for row in completed.values()),
        'classifications': {str(key): int(value) for key, value in counts.items()},
    }
    if bounded_validation is not None:
        summary['bounded_positive_retrieval'] = bounded_validation
    validate_source_files(manifest, data_root, workers)
    save_json(output / 'cpu_summary.json', summary)
    artifacts = {
        'fingerprints.jsonl': _artifact(fingerprints_path, lines=True),
        'candidate_pairs.jsonl': _artifact(pairs_path, lines=True),
    }
    if calibration_path.is_file():
        artifacts['calibration.json'] = _artifact(calibration_path)
        artifacts['bounded_positive_ranks.json'] = _artifact(
            output / 'bounded_positive_ranks.json')
        artifacts['bounded_retrieval_validation.json'] = _artifact(
            output / 'bounded_retrieval_validation.json')
    completion = {**summary, 'artifacts': artifacts}
    save_json(complete_path, completion)
    (output / 'CPU_INCOMPLETE').unlink()
    return summary


def _transformed_images(image: Image.Image):
    image = image.convert('RGB')
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=82)
    yield Image.open(BytesIO(buffer.getvalue())).convert('RGB')
    yield image.resize((max(32, image.width * 3 // 4), max(32, image.height * 3 // 4)),
                       Image.Resampling.LANCZOS)
    yield image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    yield image.rotate(3, resample=Image.Resampling.BICUBIC, fillcolor='black')


def select_hard_negative_pairs(frame: pd.DataFrame, fingerprints: dict[str, int],
                               target: int) -> list[tuple[int, str, str]]:
    patient = dict(zip(frame.path, frame.patient))
    paths = sorted(fingerprints)
    possible = []
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if patient[left] == patient[right]:
                continue
            possible.append(((fingerprints[left] ^ fingerprints[right]).bit_count(),
                             left, right))
    possible.sort()
    if len(possible) < target:
        raise RuntimeError(f'Only {len(possible)} unique cross-patient hard negatives; need {target}')
    return possible[:target]


def run_calibration(data_root: Path, prepared: Path, output: Path,
                    config_path: Path, sample_images: int = 250,
                    workers: int = 64) -> dict:
    data_root, prepared, output = map(Path, (data_root, prepared, output))
    config = AuditConfig.from_json(config_path)
    manifest = pd.read_csv(prepared / 'manifest.csv')
    development = manifest[manifest.role == 'development'].copy()
    if development.empty:
        raise RuntimeError('No development images available for calibration')
    if any((output / name).exists() for name in (
            'calibration_pairs.jsonl', 'calibration_positive_evidence.jsonl')):
        raise RuntimeError('Calibration evidence already exists; use a fresh output directory')
    print('Calibration: validating development source files', flush=True)
    validate_source_files(development, data_root, workers)
    output.mkdir(parents=True, exist_ok=True)
    incomplete = output / 'CALIBRATION_INCOMPLETE'
    incomplete.write_text(identity(calibration_binding(manifest, config)) + '\n', encoding='utf-8')
    (output / 'calibration.json').unlink(missing_ok=True)
    selected = development.sample(n=min(sample_images, len(development)),
                                  random_state=int(config.calibration['seed']))
    positive_distances = []
    positive_evidence = []
    fingerprints = {}
    positive_tasks = [(row.path, str(data_root), config.values)
                      for row in selected.itertuples()]
    print(f'Calibration: processing {len(positive_tasks)} synthetic-image groups', flush=True)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for image_number, (relative, base_hash, records) in enumerate(
                pool.map(_positive_calibration_task, positive_tasks, chunksize=1), 1):
            fingerprints[relative] = base_hash
            for record in records:
                positive_distances.append(record['hamming_distance'])
                positive_evidence.append(PairEvidence(**record['evidence']))
                _append_jsonl(output / 'calibration_positive_evidence.jsonl', {
                    'path': relative, **record,
                })
            if image_number % 10 == 0 or image_number == len(selected):
                print(f'Calibration synthetic images: {image_number}/{len(selected)}', flush=True)
    save_json(output / 'calibration_positive_distances.json', positive_distances)
    target = int(config.calibration['hard_negative_pairs'])
    possible = select_hard_negative_pairs(selected, fingerprints, target)
    tasks = [(left, right, str(data_root), config.values) for _, left, right in possible]
    negatives = []
    binding = calibration_binding(manifest, config)
    save_json(output / 'calibration_evidence_binding.json', binding)
    print(f'Calibration: comparing {len(tasks)} presumed-negative pairs', flush=True)
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for number, (pair, result) in enumerate(zip(
                possible, pool.map(_confirmation_task, tasks, chunksize=1)), 1):
            distance, left, right = pair
            evidence, classification = result
            negatives.append(PairEvidence(**evidence))
            _append_jsonl(output / 'calibration_pairs.jsonl', {
                'left': left, 'right': right, 'hamming_distance': distance,
                'evidence': evidence, 'classification': classification,
                'negative_status': 'presumed_cross_patient_not_adjudicated',
            })
            if number % 100 == 0 or number == len(tasks):
                print(f'Calibration pairs: {number}/{len(tasks)}', flush=True)
    radius = int(config.phash['candidate_hamming_radius'])
    confirmed = sum(
        distance <= radius and classify_pair(item, config.confirmation) == 'confirmed'
        for (distance, _, _), item in zip(possible, negatives))
    confirmation_recall = sum(
        distance <= radius and classify_pair(item, config.confirmation) == 'confirmed'
        for distance, item in zip(positive_distances, positive_evidence)
    ) / len(positive_distances)
    save_json(output / 'calibration_diagnostics.json', {
        **binding, 'pairs': len(negatives),
        'confirmed_presumed_negative_pairs': confirmed,
        'nominal_wilson_upper': wilson_upper(confirmed, len(negatives)),
        'positive_transformations': len(positive_distances),
        'retrieval_recall': sum(d <= int(config.phash['candidate_hamming_radius'])
                                for d in positive_distances) / len(positive_distances),
        'confirmation_recall': confirmation_recall,
        'limitations': ['Cross-patient pairs are not adjudicated negatives',
                        'Pairs share images; nominal Wilson bound is not a validated population bound'],
    })
    record = lock_calibration(positive_distances, negatives, config,
                              output / 'calibration.json',
                              calibration_binding(manifest, config),
                              positive_evidence=positive_evidence,
                              negative_distances=[distance for distance, _, _ in possible])
    incomplete.unlink()
    return record


def run_dino(data_root: Path, prepared: Path, output: Path, config_path: Path,
             weights: Path, weights_sha256: str, device: str,
             workers: int = 32) -> dict:
    data_root, prepared, output = map(Path, (data_root, prepared, output))
    config = AuditConfig.from_json(config_path)
    manifest_path = prepared / 'manifest.csv'
    manifest = pd.read_csv(manifest_path)
    parent_hash = digest(manifest_path)
    cpu = _verify_stage(output, 'CPU_COMPLETE.json', parent_hash)
    require_locked_calibration_before_external(manifest, output / 'calibration.json', config)
    validate_source_files(manifest, data_root, workers)
    if (output / 'DINO_COMPLETE.json').exists() and not (output / 'DINO_INCOMPLETE').exists():
        completed = _verify_stage(output, 'DINO_COMPLETE.json', parent_hash)
        if completed.get('cpu_audit_id') != cpu.get('audit_id'):
            raise RuntimeError('DINO completion is bound to another CPU audit')
        return {key: value for key, value in completed.items() if key != 'artifacts'}
    (output / 'DINO_INCOMPLETE').write_text(cpu['audit_id'] + '\n', encoding='utf-8')
    (output / 'DINO_COMPLETE.json').unlink(missing_ok=True)
    model = load_dino(config, weights, weights_sha256, device)

    def images():
        for row in manifest.itertuples():
            with Image.open(data_root / row.path) as image:
                yield row.path, normalize_retina(image, config, row.path).rgb

    records = list(embed_retinas(images(), model, config, device))
    keys = [record.key for record in records]
    embeddings = np.stack([record.vector for record in records]).astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise RuntimeError('Non-finite DINOv2 embedding')
    with (output / 'dinov2_embeddings.npy.tmp').open('wb') as handle:
        np.save(handle, embeddings, allow_pickle=False)
    os.replace(output / 'dinov2_embeddings.npy.tmp', output / 'dinov2_embeddings.npy')
    save_json(output / 'dinov2_keys.json', keys)
    neighbours = exact_cosine_neighbors(
        embeddings, keys, int(config.dino['top_k']), chunk_size=512)
    candidates = {}
    minimum = float(config.dino['minimum_cosine_candidate'])
    for row in neighbours:
        for neighbour in row['neighbors']:
            if neighbour['cosine'] < minimum:
                continue
            candidate = dino_candidate_record(row['key'], neighbour['key'], neighbour['cosine'])
            pair_id = candidate['pair_id']
            if pair_id not in candidates or candidate['cosine'] > candidates[pair_id]['cosine']:
                candidates[pair_id] = candidate
    dino_identity = identity({
        'cpu_audit_id': cpu['audit_id'],
        'weights_sha256': weights_sha256.lower(),
        'model': config.dino,
        'config_sha256': identity(config.values),
    })
    path = output / 'dino_candidates.jsonl'
    if path.exists():
        path.unlink()
    _touch(path)
    for candidate in sorted(candidates.values(), key=lambda row: row['pair_id']):
        _append_jsonl(path, {**candidate, 'audit_id': dino_identity})
    dino_pairs_path = output / 'dino_pairs.jsonl'
    if dino_pairs_path.exists():
        dino_pairs_path.unlink()
    _touch(dino_pairs_path)
    candidate_list = sorted(candidates.values(), key=lambda row: row['pair_id'])
    tasks = [(row['left'], row['right'], str(data_root), config.values)
             for row in candidate_list]
    classifications = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for candidate, (evidence, classification) in zip(
                candidate_list, pool.map(_confirmation_task, tasks, chunksize=1)):
            record = {**candidate, 'audit_id': dino_identity,
                      'evidence': evidence, 'classification': classification}
            _append_jsonl(dino_pairs_path, record)
            classifications.append(classification)
    summary = {
        'dino_audit_id': dino_identity,
        'cpu_audit_id': cpu['audit_id'],
        'parent_manifest_sha256': parent_hash,
        'weights_sha256': weights_sha256.lower(),
        'embeddings': len(records),
        'candidate_pairs': len(candidates),
        'classifications': pd.Series(classifications).value_counts().to_dict(),
    }
    validate_source_files(manifest, data_root, workers)
    completion = {**summary, 'artifacts': {
        'dinov2_embeddings.npy': _artifact(output / 'dinov2_embeddings.npy'),
        'dinov2_keys.json': _artifact(output / 'dinov2_keys.json'),
        'dino_candidates.jsonl': _artifact(path, lines=True),
        'dino_pairs.jsonl': _artifact(dino_pairs_path, lines=True),
    }}
    save_json(output / 'DINO_COMPLETE.json', completion)
    (output / 'DINO_INCOMPLETE').unlink()
    return summary


def _components(pairs: list[dict]) -> list[set[str]]:
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    for pair in pairs:
        union(pair['left'], pair['right'])
    groups = {}
    for value in parent:
        groups.setdefault(find(value), set()).add(value)
    return list(groups.values())


def _assign_global_splits(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame['split'] = 'external'
    indices = frame.index[frame.role == 'development']
    subset = frame.loc[indices]
    if subset.empty:
        return frame
    strata = subset.source.astype(str) + ':' + subset.label.astype(str)
    if subset.patient.nunique() < 10:
        raise RuntimeError('Too few linked development groups for ten-fold split regeneration')
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (_, positions) in enumerate(splitter.split(subset, strata, subset.patient)):
        split = 'train' if fold < 7 else {7: 'val', 8: 'calibration', 9: 'internal'}[fold]
        frame.loc[indices[positions], 'split'] = split
    if frame.loc[indices].groupby('patient').split.nunique().max() != 1:
        raise RuntimeError('Linked patient group crossed regenerated split boundaries')
    return frame


def disposition_manifest(manifest: pd.DataFrame, confirmed_pairs: list[dict],
                         seed: int) -> tuple[pd.DataFrame, list[dict]]:
    frame = manifest.copy()
    if 'original_patient' not in frame:
        frame['original_patient'] = frame.patient
    by_path = frame.set_index('path')
    exclusions = []
    remove = set()
    parent = {patient: patient for patient in frame.patient}
    def find(value):
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    components = _components(confirmed_pairs)
    for component in components:
        missing = sorted(component - set(by_path.index))
        if missing:
            raise RuntimeError('Confirmed pair path missing from manifest: ' + ', '.join(missing))
    regenerate = False
    # Build the complete patient graph before deciding any exclusions so that
    # external contamination propagates transitively through every linked eye.
    for pair in confirmed_pairs:
        left, right = by_path.loc[pair['left']], by_path.loc[pair['right']]
        union(left.patient, right.patient)
        if left.role == right.role == 'development' and left.patient != right.patient:
            regenerate = True
    external_roots = set()
    paired_paths = {path for component in components for path in component}
    for path in paired_paths:
        row = by_path.loc[path]
        if row.role == 'external':
            external_roots.add(find(row.patient))
    for row in frame.itertuples():
        if row.role == 'development' and find(row.patient) in external_roots:
            remove.add(row.path)
            exclusions.append({'path': row.path, 'reason': 'confirmed_external_overlap'})
    for component in components:
        rows = by_path.loc[sorted(component)]
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        if rows.label.nunique() > 1:
            for path in component:
                if path not in remove:
                    remove.add(path)
                    exclusions.append({'path': path,
                                       'reason': 'confirmed_near_duplicate_label_conflict'})
            continue
        external = rows[rows.role == 'external']
        if not external.empty:
            retained = min(external.index)
            for path in external.index:
                if path != retained:
                    remove.add(path)
                    exclusions.append({'path': path, 'reason': 'confirmed_external_duplicate',
                                       'retained': retained})
    frame['patient'] = frame.patient.map(find)
    frame = frame[~frame.path.isin(remove)].copy().reset_index(drop=True)
    if regenerate and (frame.role == 'development').any():
        frame = _assign_global_splits(frame, seed=seed)
    return frame, exclusions


def _blind_token(pair_id: str, side: str) -> str:
    return hashlib.sha256(f'{pair_id}:{side}'.encode()).hexdigest()[:16]


def build_review_records(pairs: list[dict], sample_seed: int,
                         sample_each: int) -> list[dict]:
    rng = random.Random(sample_seed)
    selected = [pair for pair in pairs if pair['classification'] == 'borderline']
    for classification in ('confirmed', 'rejected'):
        strata = {}
        for pair in pairs:
            if pair['classification'] != classification:
                continue
            key = (tuple(pair.get('sources', [])), tuple(pair.get('splits', [])))
            strata.setdefault(key, []).append(pair)
        for population in strata.values():
            selected.extend(rng.sample(population, min(sample_each, len(population))))
    unique = {pair['pair_id']: pair for pair in selected}
    return [{'pair_id': pair_id,
             'left_token': _blind_token(pair_id, 'left'),
             'right_token': _blind_token(pair_id, 'right')}
            for pair_id in sorted(unique)]


def export_review_pack(pairs: list[dict], data_root: Path, audit: Path,
                       public_output: Path, sample_seed: int,
                       sample_each: int) -> list[dict]:
    data_root, audit, public_output = map(Path, (data_root, audit, public_output))
    public_output.mkdir(parents=True, exist_ok=False)
    sheets = public_output / 'contact_sheets'
    sheets.mkdir()
    review = build_review_records(pairs, sample_seed, sample_each)
    pair_by_id = {pair['pair_id']: pair for pair in pairs}
    private = []
    for record in review:
        pair = pair_by_id[record['pair_id']]
        images = []
        for side in ('left', 'right'):
            with Image.open(data_root / pair[side]) as image:
                copy = image.convert('RGB')
                copy.thumbnail((512, 512), Image.Resampling.LANCZOS)
                canvas = Image.new('RGB', (512, 512), 'black')
                canvas.paste(copy, ((512 - copy.width) // 2, (512 - copy.height) // 2))
                images.append(canvas)
        sheet = Image.new('RGB', (1024, 548), 'white')
        sheet.paste(images[0], (0, 36)); sheet.paste(images[1], (512, 36))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), record['left_token'], fill='black')
        draw.text((520, 8), record['right_token'], fill='black')
        sheet.save(sheets / f"{record['pair_id']}.jpg", quality=95)
        private.append({'pair_id': record['pair_id'], 'left': pair['left'],
                        'right': pair['right'], 'left_token': record['left_token'],
                        'right_token': record['right_token']})
    save_json(public_output / 'review_pairs.json', review)
    save_json(audit / 'review_required.json', [row['pair_id'] for row in review])
    save_json(audit / 'review_private_mapping.json', private)
    return review


def validate_manual_review(pairs: list[dict], decisions: list[dict],
                           required_pair_ids: set[str] | None = None) -> None:
    required = (set(required_pair_ids) if required_pair_ids is not None else
                {pair['pair_id'] for pair in pairs if pair['classification'] == 'borderline'})
    indexed = {}
    for decision in decisions:
        pair_id = decision.get('pair_id')
        if not pair_id or pair_id in indexed:
            raise RuntimeError('Manual review has a missing or duplicate pair_id')
        if decision.get('decision') not in {'confirmed', 'rejected'}:
            raise RuntimeError('Manual review decision must be confirmed or rejected')
        if not decision.get('reviewer') or not decision.get('timestamp'):
            raise RuntimeError('Manual review requires reviewer and timestamp')
        indexed[pair_id] = decision
    missing = sorted(required - indexed.keys())
    if missing:
        raise RuntimeError(f'Missing manual decisions for {len(missing)} borderline pairs')


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Corrupt JSONL at {path}:{number}') from exc
    return records


def _link_cache(parent: Path, child: Path, manifest: pd.DataFrame) -> None:
    for relative in sorted(set(manifest.cache)):
        source = parent / relative
        target = child / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def finalize_audit(parent_prepared: Path, audit: Path, manual_review: Path,
                   child_prepared: Path, seed: int) -> dict:
    parent_prepared, audit, manual_review, child_prepared = map(
        Path, (parent_prepared, audit, manual_review, child_prepared))
    if child_prepared.exists():
        raise FileExistsError(f'Child preparation already exists: {child_prepared}')
    parent_manifest = parent_prepared / 'manifest.csv'
    parent_hash = digest(parent_manifest)
    cpu = _verify_stage(audit, 'CPU_COMPLETE.json', parent_hash)
    dino = _verify_stage(audit, 'DINO_COMPLETE.json', parent_hash)
    if dino.get('cpu_audit_id') != cpu.get('audit_id'):
        raise RuntimeError('DINO and CPU audit identities do not match')
    pairs = _read_jsonl(audit / 'candidate_pairs.jsonl') + _read_jsonl(audit / 'dino_pairs.jsonl')
    by_pair = {}
    rank = {'rejected': 0, 'borderline': 1, 'confirmed': 2}
    for pair in pairs:
        if pair.get('classification') not in rank:
            raise RuntimeError('Unverified candidate present during finalization')
        existing = by_pair.get(pair['pair_id'])
        if existing is None or rank[pair['classification']] > rank[existing['classification']]:
            by_pair[pair['pair_id']] = pair
    pairs = list(by_pair.values())
    decisions = _read_jsonl(manual_review)
    required_path = audit / 'review_required.json'
    if not required_path.is_file():
        raise RuntimeError('Blinded review pack has not been generated')
    required = set(json.loads(required_path.read_text(encoding='utf-8')))
    validate_manual_review(pairs, decisions, required)
    decision_by_pair = {row['pair_id']: row['decision'] for row in decisions}
    confirmed = []
    for pair in pairs:
        classification = decision_by_pair.get(pair['pair_id'], pair['classification'])
        if classification == 'confirmed':
            confirmed.append(pair)
    manifest = pd.read_csv(parent_manifest)
    child, exclusions = disposition_manifest(manifest, confirmed, seed)
    inherited_path = parent_prepared / 'exclusions.json'
    inherited_exclusions = (json.loads(inherited_path.read_text(encoding='utf-8'))
                            if inherited_path.is_file() else [])
    exclusions = inherited_exclusions + exclusions
    child_prepared.mkdir(parents=True)
    child.to_csv(child_prepared / 'manifest.csv', index=False)
    _link_cache(parent_prepared, child_prepared, child)
    audit_id = identity({
        'cpu_audit_id': cpu['audit_id'],
        'dino_audit_id': dino['dino_audit_id'],
        'manual_review_sha256': digest(manual_review),
        'confirmed_pair_ids': sorted(pair['pair_id'] for pair in confirmed),
    })
    summary = {
        'manifest_sha256': digest(child_prepared / 'manifest.csv'),
        'parent_manifest_sha256': parent_hash,
        'audit_id': audit_id,
        'confirmed_pairs': len(confirmed),
        'exclusions': len(exclusions),
        'counts': {str(key): int(value) for key, value in
                   child.groupby(['source', 'split']).size().to_dict().items()},
    }
    save_json(child_prepared / 'exclusions.json', exclusions)
    save_json(child_prepared / 'prepared.json', summary)
    (child_prepared / 'AUDIT_REPORT.md').write_text(
        '# Near-duplicate audit\n\n'
        f"Parent manifest: `{summary['parent_manifest_sha256']}`\n\n"
        f"Child manifest: `{summary['manifest_sha256']}`\n\n"
        f"Confirmed pairs: {len(confirmed)}\n\nExclusions: {len(exclusions)}\n",
        encoding='utf-8')
    return summary


def run_review_pack(data_root: Path, prepared: Path, audit: Path,
                    public_output: Path, sample_seed: int,
                    sample_each: int) -> list[dict]:
    prepared, audit = Path(prepared), Path(audit)
    parent_hash = digest(prepared / 'manifest.csv')
    cpu = _verify_stage(audit, 'CPU_COMPLETE.json', parent_hash)
    dino = _verify_stage(audit, 'DINO_COMPLETE.json', parent_hash)
    if dino.get('cpu_audit_id') != cpu.get('audit_id'):
        raise RuntimeError('DINO and CPU audit identities do not match')
    pairs = _read_jsonl(audit / 'candidate_pairs.jsonl') + _read_jsonl(audit / 'dino_pairs.jsonl')
    unique = {}
    rank = {'rejected': 0, 'borderline': 1, 'confirmed': 2}
    for pair in pairs:
        current = unique.get(pair['pair_id'])
        if current is None or rank[pair['classification']] > rank[current['classification']]:
            unique[pair['pair_id']] = pair
    return export_review_pack(list(unique.values()), data_root, audit,
                              public_output, sample_seed, sample_each)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='stage', required=True)
    for stage in ('calibrate', 'cpu'):
        command = subparsers.add_parser(stage)
        command.add_argument('--data', type=Path, required=True)
        command.add_argument('--prepared', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--config', type=Path, default=Path(__file__).with_name('near_duplicate_config.json'))
        command.add_argument('--workers', type=int, default=min(160, os.cpu_count() or 1))
    dino = subparsers.add_parser('dino')
    dino.add_argument('--data', type=Path, required=True)
    dino.add_argument('--prepared', type=Path, required=True)
    dino.add_argument('--out', type=Path, required=True)
    dino.add_argument('--config', type=Path, default=Path(__file__).with_name('near_duplicate_config.json'))
    dino.add_argument('--weights', type=Path, required=True)
    dino.add_argument('--weights-sha256', required=True)
    dino.add_argument('--device', default='cuda')
    dino.add_argument('--workers', type=int, default=min(64, os.cpu_count() or 1))
    review = subparsers.add_parser('review-pack')
    review.add_argument('--data', type=Path, required=True)
    review.add_argument('--prepared', type=Path, required=True)
    review.add_argument('--audit', type=Path, required=True)
    review.add_argument('--out', type=Path, required=True)
    review.add_argument('--sample-seed', type=int, default=20261103)
    review.add_argument('--sample-each', type=int, default=25)
    finalize = subparsers.add_parser('finalize')
    finalize.add_argument('--prepared', type=Path, required=True)
    finalize.add_argument('--audit', type=Path, required=True)
    finalize.add_argument('--manual-review', type=Path, required=True)
    finalize.add_argument('--out', type=Path, required=True)
    finalize.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.stage == 'calibrate':
        print(run_calibration(args.data, args.prepared, args.out, args.config,
                              workers=args.workers))
    elif args.stage == 'cpu':
        print(run_cpu(args.data, args.prepared, args.out, args.config, args.workers))
    elif args.stage == 'dino':
        print(run_dino(args.data, args.prepared, args.out, args.config,
                       args.weights, args.weights_sha256, args.device, args.workers))
    elif args.stage == 'review-pack':
        print({'review_pairs': len(run_review_pack(
            args.data, args.prepared, args.audit, args.out,
            args.sample_seed, args.sample_each))})
    else:
        print(finalize_audit(args.prepared, args.audit, args.manual_review,
                             args.out, args.seed))


if __name__ == '__main__':
    main()
