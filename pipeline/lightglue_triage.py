#!/usr/bin/env python3
"""ALIKED + LightGlue evidence for frozen retinal review pairs.

This program writes machine evidence only. It never writes manual-review
decisions, changes a prepared manifest, or starts model training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


REQUIRED_FEATURE_KEYS = {"keypoints", "descriptors", "image_size"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def load_pairs(path: Path) -> list[dict]:
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"pair_id", "left", "right"} - row.keys()
            if missing:
                raise RuntimeError(f"Missing pair identity at line {number}: {sorted(missing)}")
            if row["pair_id"] in seen:
                raise RuntimeError(f"Duplicate pair_id at line {number}: {row['pair_id']}")
            seen.add(row["pair_id"])
            rows.append({key: row[key] for key in ("pair_id", "left", "right")})
    if not rows:
        raise RuntimeError("Private review-pair ledger is empty")
    return sorted(rows, key=lambda row: row["pair_id"])


def validate_resume(path: Path, expected: set[str]) -> set[str]:
    completed = set()
    path = Path(path)
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"Truncated resume ledger at line {number}")
            row = json.loads(line)
            pair_id = row.get("pair_id")
            if pair_id not in expected:
                raise RuntimeError(f"Unknown pair_id in resume ledger: {pair_id}")
            if pair_id in completed:
                raise RuntimeError(f"Duplicate resume pair_id: {pair_id}")
            completed.add(pair_id)
    return completed


def geometry_evidence(left: np.ndarray, right: np.ndarray,
                      scores: np.ndarray) -> dict:
    left = np.asarray(left, dtype=np.float32).reshape(-1, 2)
    right = np.asarray(right, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(left) != len(right) or len(left) != len(scores):
        raise ValueError("Matched point/score lengths differ")
    result = {
        "matches": int(len(left)),
        "score_minimum": float(scores.min()) if len(scores) else None,
        "score_median": float(np.median(scores)) if len(scores) else None,
        "score_mean": float(scores.mean()) if len(scores) else None,
        "score_maximum": float(scores.max()) if len(scores) else None,
        "homography_inliers": 0,
        "homography_inlier_ratio": 0.0,
        "median_reprojection_error": None,
    }
    if len(left) < 4:
        return result
    matrix, mask = cv2.findHomography(left, right, cv2.RANSAC, 3.0,
                                      maxIters=5000, confidence=0.999)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return result
    mask = mask.reshape(-1).astype(bool)
    homogeneous = np.concatenate(
        [left.astype(np.float64), np.ones((len(left), 1))], axis=1
    )
    projected = homogeneous @ matrix.T
    valid = np.abs(projected[:, 2]) > 1e-12
    projected[valid, :2] /= projected[valid, 2:3]
    errors = np.linalg.norm(projected[:, :2] - right, axis=1)
    usable = mask & valid & np.isfinite(errors)
    inliers = int(usable.sum())
    result.update({
        "homography_inliers": inliers,
        "homography_inlier_ratio": float(inliers / len(left)),
        "median_reprojection_error": (
            float(np.median(errors[usable])) if inliers else None
        ),
    })
    return result


def feature_path(cache: Path, key: str) -> Path:
    return cache / (hashlib.sha256(key.encode()).hexdigest() + ".pt")


def cpu_feature(record: dict) -> dict:
    import torch
    result = {}
    for key, value in record.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().contiguous()
    missing = REQUIRED_FEATURE_KEYS - result.keys()
    if missing:
        raise RuntimeError(f"Extractor result missing keys: {sorted(missing)}")
    return result


def load_cached_feature(path: Path, device: str) -> dict:
    import torch
    try:
        record = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        record = torch.load(path, map_location="cpu")
    missing = REQUIRED_FEATURE_KEYS - record.keys()
    if missing:
        raise RuntimeError(f"Corrupt cached feature {path}: {sorted(missing)}")
    return {key: value.to(device, non_blocking=True) for key, value in record.items()}


def extract_features(extractor, load_image, data: Path, keys: list[str],
                     cache: Path, device: str, resize: int, run_id: str) -> None:
    import torch
    cache.mkdir(parents=True, exist_ok=True)
    progress_path = cache / "FEATURE_PROGRESS.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("run_id") != run_id:
            raise RuntimeError("Feature-cache execution identity mismatch")
    started = time.monotonic()
    for index, key in enumerate(keys, 1):
        destination = feature_path(cache, key)
        if not destination.is_file():
            source = data / key
            if not source.is_file():
                raise FileNotFoundError(source)
            image = load_image(source).to(device)
            with torch.inference_mode():
                values = cpu_feature(extractor.extract(image, resize=resize))
            temporary = Path(str(destination) + ".tmp")
            torch.save(values, temporary)
            os.replace(temporary, destination)
        if index % 32 == 0 or index == len(keys):
            elapsed = max(time.monotonic() - started, 1e-9)
            atomic_json(progress_path, {
                "run_id": run_id,
                "completed": index,
                "total": len(keys),
                "images_per_second_since_resume": index / elapsed,
            })
            print(f"ALIKED features: {index}/{len(keys)}", flush=True)


def match_pairs(matcher, rbd, pairs: list[dict], cache: Path, output: Path,
                device: str, run_id: str, flush_every: int) -> None:
    import torch
    ledger = output / "lightglue_scores.jsonl"
    expected = {row["pair_id"] for row in pairs}
    completed = validate_resume(ledger, expected)
    mode = "a" if ledger.exists() else "w"
    started, added = time.monotonic(), 0
    with ledger.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in pairs:
            if row["pair_id"] in completed:
                continue
            left = load_cached_feature(feature_path(cache, row["left"]), device)
            right = load_cached_feature(feature_path(cache, row["right"]), device)
            with torch.inference_mode():
                matched = rbd(matcher({"image0": left, "image1": right}))
            left_unbatched, right_unbatched = rbd(left), rbd(right)
            match_indices = matched["matches"].detach().cpu().numpy()
            scores = matched["scores"].detach().float().cpu().numpy()
            points_left = left_unbatched["keypoints"].detach().cpu().numpy()[match_indices[:, 0]]
            points_right = right_unbatched["keypoints"].detach().cpu().numpy()[match_indices[:, 1]]
            evidence = geometry_evidence(points_left, points_right, scores)
            evidence.update({
                "pair_id": row["pair_id"],
                "left_keypoints": int(left_unbatched["keypoints"].shape[0]),
                "right_keypoints": int(right_unbatched["keypoints"].shape[0]),
            })
            handle.write(json.dumps(evidence, sort_keys=True) + "\n")
            completed.add(row["pair_id"])
            added += 1
            if added % flush_every == 0 or len(completed) == len(pairs):
                handle.flush()
                os.fsync(handle.fileno())
                elapsed = max(time.monotonic() - started, 1e-9)
                atomic_json(output / "LIGHTGLUE_PROGRESS.json", {
                    "run_id": run_id,
                    "completed": len(completed),
                    "total": len(pairs),
                    "pairs_per_second_since_resume": added / elapsed,
                })
                print(f"LightGlue pairs: {len(completed)}/{len(pairs)}", flush=True)


def vendor_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def run(args: argparse.Namespace) -> dict:
    import torch
    vendor = args.vendor.resolve()
    sys.path.insert(0, str(vendor))
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image, rbd

    commit = vendor_commit(vendor)
    if commit != args.vendor_commit:
        raise RuntimeError(f"LightGlue commit mismatch: {commit}")
    pairs = load_pairs(args.pairs)
    keys = sorted({row[key] for row in pairs for key in ("left", "right")})
    for key in keys:
        resolved = (args.data / key).resolve()
        try:
            resolved.relative_to(args.data.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Image path escapes dataset root: {key}") from exc
    config = {
        "vendor_commit": commit,
        "pairs_sha256": digest(args.pairs),
        "code_sha256": digest(Path(__file__)),
        "images": len(keys),
        "pairs": len(pairs),
        "resize": args.resize,
        "max_keypoints": args.max_keypoints,
        "matcher": "aliked",
        "filter_threshold": args.filter_threshold,
    }
    run_id = identity(config)
    args.out.mkdir(parents=True, exist_ok=True)
    state = args.out / "LIGHTGLUE_RUN_IDENTITY.json"
    if state.exists() and json.loads(state.read_text()).get("run_id") != run_id:
        raise RuntimeError("Output directory belongs to another execution identity")
    atomic_json(state, {"run_id": run_id, **config})
    incomplete = args.out / "LIGHTGLUE_EVIDENCE_INCOMPLETE"
    incomplete.write_text(run_id + "\n", encoding="utf-8")
    (args.out / "LIGHTGLUE_EVIDENCE_COMPLETE.json").unlink(missing_ok=True)

    torch.set_float32_matmul_precision("high")
    extractor = ALIKED(max_num_keypoints=args.max_keypoints).eval().to(args.device)
    matcher = LightGlue(features="aliked", filter_threshold=args.filter_threshold,
                        mp=True).eval().to(args.device)
    cache = args.out / "aliked_features"
    extract_features(extractor, load_image, args.data, keys, cache, args.device,
                     args.resize, run_id)
    del extractor
    torch.cuda.empty_cache()
    match_pairs(matcher, rbd, pairs, cache, args.out, args.device, run_id,
                args.flush_every)

    ledger = args.out / "lightglue_scores.jsonl"
    completed = validate_resume(ledger, {row["pair_id"] for row in pairs})
    if len(completed) != len(pairs):
        raise RuntimeError(f"Incomplete LightGlue ledger: {len(completed)}/{len(pairs)}")
    summary = {
        "purpose": "independent machine evidence; not human adjudication",
        "run_id": run_id,
        **config,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "artifacts": {"lightglue_scores.jsonl": digest(ledger)},
        "limitations": [
            "No thresholds or automatic decisions are produced by this stage",
            "Local-feature evidence requires separate target-domain calibration",
            "Machine evidence cannot be represented as human review",
        ],
    }
    atomic_json(args.out / "LIGHTGLUE_EVIDENCE_COMPLETE.json", summary)
    incomplete.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--vendor-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--filter-threshold", type=float, default=0.1)
    parser.add_argument("--flush-every", type=int, default=100)
    args = parser.parse_args()
    if not (256 <= args.resize <= 2048):
        parser.error("--resize must be in [256, 2048]")
    if not (128 <= args.max_keypoints <= 8192):
        parser.error("--max-keypoints must be in [128, 8192]")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
