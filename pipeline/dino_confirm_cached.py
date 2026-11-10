"""Resumable cached confirmation for an already-completed DINO retrieval stage."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

from audit_duplicates import _artifact, _verify_stage, validate_source_files
from common import digest, identity, save_json
from near_duplicates import (
    AuditConfig,
    NormalizedRetina,
    classify_pair,
    confirm_pair,
    normalize_retina,
)


_CACHE_GRAY = None
_CACHE_MASK = None
_CACHE_CONFIG = None


def _normalize_cache_task(task):
    index, relative, data_root, config_values = task
    config = AuditConfig(config_values)
    with Image.open(Path(data_root) / relative) as image:
        normalized = normalize_retina(image, config, relative)
    return index, relative, normalized.gray.copy(), normalized.mask.copy()


def _init_confirmation_worker(gray_path: str, mask_path: str, config_values: dict):
    global _CACHE_GRAY, _CACHE_MASK, _CACHE_CONFIG
    _CACHE_GRAY = np.load(gray_path, mmap_mode="r")
    _CACHE_MASK = np.load(mask_path, mmap_mode="r")
    _CACHE_CONFIG = AuditConfig(config_values)


def _cached_confirmation_task(task):
    pair_id, left_index, right_index = task
    empty_rgb = np.empty((0,), dtype=np.uint8)
    left = NormalizedRetina(
        empty_rgb, np.asarray(_CACHE_GRAY[left_index]),
        np.asarray(_CACHE_MASK[left_index]), (0, 0, 0, 0), 0.0)
    right = NormalizedRetina(
        empty_rgb, np.asarray(_CACHE_GRAY[right_index]),
        np.asarray(_CACHE_MASK[right_index]), (0, 0, 0, 0), 0.0)
    evidence = confirm_pair(left, right, _CACHE_CONFIG)
    return pair_id, asdict(evidence), classify_pair(evidence, _CACHE_CONFIG.confirmation)


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl_batch(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_cache(data_root: Path, keys: list[str], output: Path,
                config: AuditConfig, workers: int, cache_id: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    size = int(config.normalized_size)
    total = len(keys)
    gray_partial = output / "gray.partial.npy"
    mask_partial = output / "mask.partial.npy"
    gray_final = output / "gray.npy"
    mask_final = output / "mask.npy"
    progress_path = output / "progress.json"
    complete_path = output / "CACHE_COMPLETE.json"

    if complete_path.exists():
        record = json.loads(complete_path.read_text(encoding="utf-8"))
        if record.get("cache_id") != cache_id:
            raise RuntimeError("Normalized cache identity mismatch")
        for name in ("gray.npy", "mask.npy"):
            path = output / name
            expected = record["artifacts"][name]
            if (not path.is_file() or path.stat().st_size != expected["bytes"] or
                    digest(path) != expected["sha256"]):
                raise RuntimeError(f"Normalized cache artifact invalid: {name}")
        return record

    completed = 0
    if gray_partial.exists() and mask_partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("cache_id") != cache_id or progress.get("total") != total:
            raise RuntimeError("Normalized cache resume identity mismatch")
        completed = int(progress["completed"])
        gray = np.lib.format.open_memmap(gray_partial, mode="r+")
        mask = np.lib.format.open_memmap(mask_partial, mode="r+")
    else:
        for path in (gray_partial, mask_partial, progress_path):
            path.unlink(missing_ok=True)
        gray = np.lib.format.open_memmap(
            gray_partial, mode="w+", dtype=np.float32, shape=(total, size, size))
        mask = np.lib.format.open_memmap(
            mask_partial, mode="w+", dtype=np.bool_, shape=(total, size, size))

    tasks = ((index, keys[index], str(data_root), config.values)
             for index in range(completed, total))
    context = multiprocessing.get_context("spawn")
    started = time.monotonic()
    resume_start = completed
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context) as pool:
        for index, relative, gray_image, mask_image in pool.map(
                _normalize_cache_task, tasks, chunksize=1):
            if index != completed or relative != keys[index]:
                raise RuntimeError("Normalized cache producer order changed")
            gray[index] = gray_image
            mask[index] = mask_image
            completed += 1
            if completed % 32 == 0 or completed == total:
                gray.flush()
                mask.flush()
                elapsed = max(time.monotonic() - started, 1e-9)
                save_json(progress_path, {
                    "cache_id": cache_id, "completed": completed, "total": total,
                    "workers": workers,
                    "images_per_second_since_resume": (completed - resume_start) / elapsed,
                })
                print(f"Normalized cache: {completed}/{total}", flush=True)

    gray.flush()
    mask.flush()
    del gray, mask
    os.replace(gray_partial, gray_final)
    os.replace(mask_partial, mask_final)
    progress_path.unlink(missing_ok=True)
    record = {
        "cache_id": cache_id,
        "images": total,
        "shape": [total, size, size],
        "gray_dtype": "float32",
        "mask_dtype": "bool",
        "artifacts": {
            "gray.npy": {"bytes": gray_final.stat().st_size,
                         "sha256": digest(gray_final)},
            "mask.npy": {"bytes": mask_final.stat().st_size,
                         "sha256": digest(mask_final)},
        },
    }
    save_json(complete_path, record)
    return record


def _direct_confirmation(candidate: dict, data_root: Path,
                         config: AuditConfig) -> tuple[dict, str]:
    with Image.open(data_root / candidate["left"]) as image:
        left = normalize_retina(image, config, candidate["left"])
    with Image.open(data_root / candidate["right"]) as image:
        right = normalize_retina(image, config, candidate["right"])
    evidence = confirm_pair(left, right, config)
    return asdict(evidence), classify_pair(evidence, config.confirmation)


def validate_cache_equivalence(candidates: list[dict], data_root: Path,
                               keys: list[str], cache: Path,
                               config: AuditConfig, samples: int,
                               execution_id: str) -> dict:
    selected = candidates[:min(samples, len(candidates))]
    index = {key: number for number, key in enumerate(keys)}
    _init_confirmation_worker(
        str(cache / "gray.npy"), str(cache / "mask.npy"), config.values)
    records = []
    maximum = 0.0
    numeric = ("ssim", "ncc", "nmae", "mask_iou", "rotation", "scale",
               "vessel_ncc", "vessel_dice")
    for candidate in selected:
        direct_evidence, direct_class = _direct_confirmation(candidate, data_root, config)
        _, cached_evidence, cached_class = _cached_confirmation_task((
            candidate["pair_id"], index[candidate["left"]], index[candidate["right"]]))
        if direct_class != cached_class:
            raise RuntimeError("Cached confirmation classification mismatch")
        for name in numeric:
            maximum = max(maximum, abs(float(direct_evidence[name]) -
                                       float(cached_evidence[name])))
        if direct_evidence["orientation"] != cached_evidence["orientation"]:
            raise RuntimeError("Cached confirmation orientation mismatch")
        maximum = max(maximum, *(abs(float(a) - float(b)) for a, b in zip(
            direct_evidence["translation"], cached_evidence["translation"])))
        records.append({"pair_id": candidate["pair_id"],
                        "classification": direct_class})
    if maximum > 1e-7:
        raise RuntimeError(f"Cached confirmation numerical mismatch: {maximum}")
    return {"execution_id": execution_id, "samples": len(records),
            "maximum_absolute_difference": maximum, "pairs": records}


def run(args) -> dict:
    data_root = args.data.resolve()
    prepared = args.prepared.resolve()
    output = args.out.resolve()
    config = AuditConfig.from_json(args.config)
    manifest_path = prepared / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    parent_hash = digest(manifest_path)
    cpu = _verify_stage(output, "CPU_COMPLETE.json", parent_hash)
    keys = json.loads((output / "dinov2_keys.json").read_text(encoding="utf-8"))
    if keys != manifest.path.astype(str).tolist():
        raise RuntimeError("DINO keys do not match the frozen manifest")
    embeddings_path = output / "dinov2_embeddings.npy"
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.shape[0] != len(keys):
        raise RuntimeError("Embedding row count mismatch")
    candidates = _read_jsonl(args.candidate_source)
    if len({row.get("pair_id") for row in candidates}) != len(candidates):
        raise RuntimeError("Candidate source has missing or duplicate pair identities")
    for row in candidates:
        row.pop("audit_id", None)
    script_hash = digest(Path(__file__))
    source_hash = digest(args.candidate_source)
    execution_id = identity({
        "cpu_audit_id": cpu["audit_id"],
        "weights_sha256": args.weights_sha256.lower(),
        "config_sha256": identity(config.values),
        "embeddings_sha256": digest(embeddings_path),
        "candidate_source_sha256": source_hash,
        "cached_confirmation_code_sha256": script_hash,
        "workers": args.workers,
    })
    cache_id = identity({
        "parent_manifest_sha256": parent_hash,
        "normalization_code_sha256": digest(Path(__file__).with_name("near_duplicates.py")),
        "config_sha256": identity(config.values),
        "keys": keys,
    })
    (output / "DINO_INCOMPLETE").write_text(execution_id + "\n", encoding="utf-8")
    (output / "DINO_COMPLETE.json").unlink(missing_ok=True)
    cache_dir = output / "normalized_confirmation_cache_v1"
    cache_record = build_cache(
        data_root, keys, cache_dir, config, args.workers, cache_id)
    equivalence = validate_cache_equivalence(
        candidates, data_root, keys, cache_dir, config,
        args.equivalence_samples, execution_id)
    equivalence.update({"code_sha256": script_hash, "cache_id": cache_id})
    equivalence_path = output / "dino_cached_confirmation_equivalence.json"
    save_json(equivalence_path, equivalence)

    rewritten = [{**row, "audit_id": execution_id} for row in candidates]
    candidates_path = output / "dino_candidates.jsonl"
    _write_jsonl_atomic(candidates_path, rewritten)
    pairs_path = output / "dino_pairs.jsonl"
    completed_records = []
    if pairs_path.exists():
        completed_records = _read_jsonl(pairs_path)
        if any(row.get("audit_id") != execution_id for row in completed_records):
            raise RuntimeError("Cached confirmation resume identity mismatch")
        if len({row.get("pair_id") for row in completed_records}) != len(completed_records):
            raise RuntimeError("Cached confirmation resume contains duplicate pairs")
    else:
        pairs_path.touch()
    completed_ids = {row["pair_id"] for row in completed_records}
    index = {key: number for number, key in enumerate(keys)}
    pending_candidates = [row for row in candidates if row["pair_id"] not in completed_ids]
    tasks = [(row["pair_id"], index[row["left"]], index[row["right"]])
             for row in pending_candidates]
    by_pair = {row["pair_id"]: row for row in candidates}
    classifications = [row["classification"] for row in completed_records]
    buffer = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
            max_workers=max(1, args.workers), mp_context=context,
            initializer=_init_confirmation_worker,
            initargs=(str(cache_dir / "gray.npy"), str(cache_dir / "mask.npy"),
                      config.values)) as pool:
        for pending_number, (pair_id, evidence, classification) in enumerate(
                pool.map(_cached_confirmation_task, tasks, chunksize=1), 1):
            number = len(completed_records) + pending_number
            buffer.append({**by_pair[pair_id], "audit_id": execution_id,
                           "evidence": evidence, "classification": classification})
            classifications.append(classification)
            if len(buffer) >= 500 or pending_number == len(tasks):
                _append_jsonl_batch(pairs_path, buffer)
                buffer.clear()
                print(f"Cached confirmations: {number}/{len(candidates)}", flush=True)

    summary = {
        "dino_audit_id": execution_id,
        "cpu_audit_id": cpu["audit_id"],
        "parent_manifest_sha256": parent_hash,
        "weights_sha256": args.weights_sha256.lower(),
        "cached_confirmation_code_sha256": script_hash,
        "cache": cache_record,
        "embeddings": len(keys),
        "candidate_pairs": len(candidates),
        "classifications": pd.Series(classifications).value_counts().to_dict(),
        "equivalence": equivalence,
    }
    validate_source_files(manifest, data_root, args.workers)
    completion = {**summary, "artifacts": {
        "dino_cached_confirmation_equivalence.json": _artifact(equivalence_path),
        "dinov2_embeddings.npy": _artifact(embeddings_path),
        "dinov2_keys.json": _artifact(output / "dinov2_keys.json"),
        "dino_candidates.jsonl": _artifact(candidates_path, lines=True),
        "dino_pairs.jsonl": _artifact(pairs_path, lines=True),
    }}
    save_json(output / "DINO_COMPLETE.json", completion)
    (output / "DINO_INCOMPLETE").unlink()
    return summary


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--candidate-source", type=Path, required=True)
    result.add_argument("--weights-sha256", required=True)
    result.add_argument("--workers", type=int, default=160)
    result.add_argument("--equivalence-samples", type=int, default=16)
    return result


if __name__ == "__main__":
    print(run(parser().parse_args()), flush=True)
