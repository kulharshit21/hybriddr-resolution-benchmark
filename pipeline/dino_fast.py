"""High-throughput, resumable DINO audit using frozen calibrated primitives.

This module deliberately leaves audit_duplicates.py and near_duplicates.py unchanged.
It parallelizes retinal normalization, batches GPU inference, performs exact cosine
retrieval on CUDA, and records its own code hash in every stage identity.
"""
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

from audit_duplicates import (
    _artifact,
    _confirmation_task,
    _verify_stage,
    require_locked_calibration_before_external,
    validate_source_files,
)
from common import digest, identity, save_json
from near_duplicates import (
    AuditConfig,
    classify_pair,
    dino_candidate_record,
    load_dino,
    normalize_retina,
)


def _normalize_task(task: tuple[int, str, str, dict]) -> tuple[int, str, np.ndarray]:
    index, relative, data_root, config_values = task
    config = AuditConfig(config_values)
    with Image.open(Path(data_root) / relative) as image:
        # copy() guarantees writable storage and removes torch.from_numpy warnings.
        rgb = normalize_retina(image, config, relative).rgb.copy()
    return index, relative, rgb


def _atomic_progress(path: Path, value: dict) -> None:
    save_json(path, value)


def _write_jsonl(path: Path, records: list[dict], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_resume_jsonl(path: Path, audit_id: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("audit_id") != audit_id:
                raise RuntimeError(f"Resume identity mismatch at {path}:{number}")
            pair_id = row.get("pair_id")
            if not pair_id or pair_id in records:
                raise RuntimeError(f"Invalid or duplicate pair_id at {path}:{number}")
            records[pair_id] = row
    return records


def _batched_features(model, rgb_batch: list[np.ndarray], config: AuditConfig,
                      device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    size = int(config.dino["input_size"])
    pretrained = getattr(model, "pretrained_cfg", {}) or {}
    mean = torch.tensor(pretrained.get("mean", (0.485, 0.456, 0.406)),
                        dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(pretrained.get("std", (0.229, 0.224, 0.225)),
                       dtype=torch.float32, device=device).view(1, 3, 1, 1)
    array = np.stack(rgb_batch, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    tensor = tensor.pin_memory().to(device, non_blocking=True).div_(255.0)
    tensor = F.interpolate(tensor, (size, size), mode="bicubic",
                           align_corners=False, antialias=True)
    with torch.inference_mode():
        features = model((tensor - mean) / std)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features = F.normalize(features.flatten(1), dim=1)
    result = features.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite DINOv2 embedding")
    return result


def _serial_feature(model, rgb: np.ndarray, config: AuditConfig,
                    device: str) -> np.ndarray:
    return _batched_features(model, [rgb], config, device)[0]


def numerical_equivalence(data_root: Path, manifest: pd.DataFrame, model,
                          config: AuditConfig, device: str,
                          batch_size: int, samples: int) -> dict:
    selected = manifest.iloc[:min(samples, len(manifest))]
    images = []
    paths = []
    for row in selected.itertuples():
        with Image.open(data_root / row.path) as image:
            images.append(normalize_retina(image, config, row.path).rgb.copy())
        paths.append(row.path)
    serial = np.stack([_serial_feature(model, image, config, device)
                       for image in images])
    batched_parts = []
    for start in range(0, len(images), batch_size):
        batched_parts.append(_batched_features(
            model, images[start:start + batch_size], config, device))
    batched = np.concatenate(batched_parts, axis=0)
    absolute = np.abs(serial - batched)
    cosine = np.sum(serial * batched, axis=1)
    result = {
        "samples": len(images),
        "batch_size": batch_size,
        "maximum_absolute_difference": float(absolute.max(initial=0.0)),
        "minimum_cosine_similarity": float(cosine.min(initial=1.0)),
        "paths": paths,
    }
    if result["maximum_absolute_difference"] > 1e-5:
        raise RuntimeError(f"Batched equivalence absolute error failed: {result}")
    if result["minimum_cosine_similarity"] < 0.99999:
        raise RuntimeError(f"Batched equivalence cosine failed: {result}")
    return result


def embed_parallel(data_root: Path, manifest: pd.DataFrame, output: Path,
                   model, config: AuditConfig, device: str, workers: int,
                   batch_size: int, execution_id: str) -> tuple[np.ndarray, list[str], dict]:
    import torch

    total = len(manifest)
    dimensions = int(getattr(model, "num_features", 768))
    partial = output / "dinov2_embeddings.partial.npy"
    progress_path = output / "dino_embedding_progress.json"
    final = output / "dinov2_embeddings.npy"
    keys_path = output / "dinov2_keys.json"
    keys = manifest.path.astype(str).tolist()

    if final.exists() and keys_path.exists():
        saved_keys = json.loads(keys_path.read_text(encoding="utf-8"))
        matrix = np.load(final, mmap_mode="r")
        if saved_keys != keys or matrix.shape != (total, dimensions):
            raise RuntimeError("Completed embedding artifact does not match this manifest")
        return matrix, keys, {
            "embeddings": total,
            "batch_size": batch_size,
            "preprocess_workers": workers,
            "resumed_completed_embeddings": True,
        }

    completed = 0
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("execution_id") != execution_id or progress.get("total") != total:
            raise RuntimeError("Embedding checkpoint identity mismatch")
        completed = int(progress["completed"])
        matrix = np.lib.format.open_memmap(partial, mode="r+")
        if matrix.shape != (total, dimensions):
            raise RuntimeError("Embedding checkpoint shape mismatch")
    else:
        partial.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        matrix = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(total, dimensions))

    resume_start = completed
    tasks = ((index, keys[index], str(data_root), config.values)
             for index in range(completed, total))
    started = time.monotonic()
    pending_rgb: list[np.ndarray] = []
    pending_indices: list[int] = []
    next_expected = completed
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context) as pool:
        for index, relative, rgb in pool.map(_normalize_task, tasks, chunksize=1):
            if index != next_expected or relative != keys[index]:
                raise RuntimeError("Parallel preprocessing order changed")
            next_expected += 1
            pending_indices.append(index)
            pending_rgb.append(rgb)
            if len(pending_rgb) < batch_size and next_expected < total:
                continue
            features = _batched_features(model, pending_rgb, config, device)
            matrix[pending_indices] = features
            completed = pending_indices[-1] + 1
            matrix.flush()
            elapsed = max(time.monotonic() - started, 1e-9)
            _atomic_progress(progress_path, {
                "execution_id": execution_id,
                "completed": completed,
                "total": total,
                "batch_size": batch_size,
                "workers": workers,
                "images_per_second_since_resume": (completed - resume_start) / elapsed,
            })
            print(f"DINO embeddings: {completed}/{total}", flush=True)
            pending_rgb.clear()
            pending_indices.clear()

    matrix.flush()
    del matrix
    os.replace(partial, final)
    save_json(keys_path, keys)
    progress_path.unlink(missing_ok=True)
    embeddings = np.load(final, mmap_mode="r")
    return embeddings, keys, {
        "embeddings": total,
        "batch_size": batch_size,
        "preprocess_workers": workers,
    }


def exact_cuda_neighbors(embeddings: np.ndarray, keys: list[str], top_k: int,
                         device: str, chunk_size: int) -> list[dict]:
    """Exact float32 cosine with deterministic lexicographic tie-breaking."""
    import torch
    import torch.nn.functional as F

    order = np.argsort(np.asarray(keys, dtype=object), kind="stable")
    sorted_keys = [keys[int(index)] for index in order]
    sorted_matrix = np.asarray(embeddings[order], dtype=np.float32)
    corpus = torch.from_numpy(sorted_matrix.copy()).to(device)
    corpus = F.normalize(corpus, dim=1)
    rows = []
    for start in range(0, len(sorted_keys), chunk_size):
        stop = min(len(sorted_keys), start + chunk_size)
        scores = corpus[start:stop] @ corpus.T
        local = torch.arange(stop - start, device=device)
        absolute = torch.arange(start, stop, device=device)
        scores[local, absolute] = -torch.inf
        indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
        selected_scores = torch.gather(scores, 1, indices).cpu().numpy()
        selected_indices = indices.cpu().numpy()
        for local_index, key in enumerate(sorted_keys[start:stop]):
            rows.append({
                "key": key,
                "neighbors": [
                    {"key": sorted_keys[int(candidate)],
                     "cosine": float(selected_scores[local_index, column])}
                    for column, candidate in enumerate(selected_indices[local_index])
                ],
            })
        print(f"DINO exact retrieval: {stop}/{len(sorted_keys)}", flush=True)
    return rows


def run(args) -> dict:
    data_root = args.data.resolve()
    prepared = args.prepared.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = AuditConfig.from_json(args.config)
    manifest_path = prepared / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    parent_hash = digest(manifest_path)
    cpu = _verify_stage(output, "CPU_COMPLETE.json", parent_hash)
    require_locked_calibration_before_external(manifest, output / "calibration.json", config)
    script_hash = digest(Path(__file__))
    execution_id = identity({
        "cpu_audit_id": cpu["audit_id"],
        "weights_sha256": args.weights_sha256.lower(),
        "model": config.dino,
        "config_sha256": identity(config.values),
        "fast_execution_code_sha256": script_hash,
        "batch_size": args.batch_size,
        "preprocess_workers": args.preprocess_workers,
        "retrieval_chunk_size": args.retrieval_chunk_size,
    })
    incomplete = output / "DINO_INCOMPLETE"
    incomplete.write_text(execution_id + "\n", encoding="utf-8")
    (output / "DINO_COMPLETE.json").unlink(missing_ok=True)

    validate_source_files(manifest, data_root, args.preprocess_workers)
    model = load_dino(config, args.weights, args.weights_sha256, args.device)
    equivalence_path = output / "dino_fast_equivalence.json"
    if not equivalence_path.exists():
        equivalence = numerical_equivalence(
            data_root, manifest, model, config, args.device,
            args.batch_size, args.equivalence_samples)
        equivalence.update({"execution_id": execution_id, "code_sha256": script_hash})
        save_json(equivalence_path, equivalence)
    else:
        equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
        if equivalence.get("execution_id") != execution_id:
            raise RuntimeError("Equivalence artifact identity mismatch")

    embeddings, keys, embedding_summary = embed_parallel(
        data_root, manifest, output, model, config, args.device,
        args.preprocess_workers, args.batch_size, execution_id)
    neighbours = exact_cuda_neighbors(
        embeddings, keys, int(config.dino["top_k"]), args.device,
        args.retrieval_chunk_size)
    minimum = float(config.dino["minimum_cosine_candidate"])
    candidates: dict[str, dict] = {}
    for row in neighbours:
        for neighbour in row["neighbors"]:
            if neighbour["cosine"] < minimum:
                continue
            candidate = dino_candidate_record(
                row["key"], neighbour["key"], neighbour["cosine"])
            pair_id = candidate["pair_id"]
            if pair_id not in candidates or candidate["cosine"] > candidates[pair_id]["cosine"]:
                candidates[pair_id] = candidate

    candidate_list = sorted(candidates.values(), key=lambda row: row["pair_id"])
    candidates_path = output / "dino_candidates.jsonl"
    _write_jsonl(candidates_path, [
        {**row, "audit_id": execution_id} for row in candidate_list
    ])
    pairs_path = output / "dino_pairs.jsonl"
    completed_pairs = _load_resume_jsonl(pairs_path, execution_id)
    pending = [row for row in candidate_list if row["pair_id"] not in completed_pairs]
    tasks = [(row["left"], row["right"], str(data_root), config.values)
             for row in pending]
    buffer = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, args.confirmation_workers),
                             mp_context=context) as pool:
        for number, (candidate, (evidence, classification)) in enumerate(
                zip(pending, pool.map(_confirmation_task, tasks, chunksize=1)), 1):
            buffer.append({**candidate, "audit_id": execution_id,
                           "evidence": evidence, "classification": classification})
            if len(buffer) >= 100 or number == len(pending):
                _write_jsonl(pairs_path, buffer, mode="a")
                completed_pairs.update((row["pair_id"], row) for row in buffer)
                buffer.clear()
                print(f"DINO confirmations: {len(completed_pairs)}/{len(candidate_list)}",
                      flush=True)

    classifications = [row["classification"] for row in completed_pairs.values()]
    summary = {
        "dino_audit_id": execution_id,
        "cpu_audit_id": cpu["audit_id"],
        "parent_manifest_sha256": parent_hash,
        "weights_sha256": args.weights_sha256.lower(),
        "fast_execution_code_sha256": script_hash,
        **embedding_summary,
        "candidate_pairs": len(candidate_list),
        "classifications": pd.Series(classifications).value_counts().to_dict(),
        "equivalence": equivalence,
    }
    validate_source_files(manifest, data_root, args.preprocess_workers)
    completion = {**summary, "artifacts": {
        "dino_fast_equivalence.json": _artifact(equivalence_path),
        "dinov2_embeddings.npy": _artifact(output / "dinov2_embeddings.npy"),
        "dinov2_keys.json": _artifact(output / "dinov2_keys.json"),
        "dino_candidates.jsonl": _artifact(candidates_path, lines=True),
        "dino_pairs.jsonl": _artifact(pairs_path, lines=True),
    }}
    save_json(output / "DINO_COMPLETE.json", completion)
    incomplete.unlink()
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--weights-sha256", required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--preprocess-workers", type=int, default=160)
    result.add_argument("--confirmation-workers", type=int, default=160)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--retrieval-chunk-size", type=int, default=512)
    result.add_argument("--equivalence-samples", type=int, default=16)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    print(run(arguments), flush=True)
