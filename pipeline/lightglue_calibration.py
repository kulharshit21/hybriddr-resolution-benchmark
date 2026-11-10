#!/usr/bin/env python3
"""Collect LightGlue calibration evidence without selecting thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lightglue_triage import (
    atomic_json,
    cpu_feature,
    digest,
    feature_path,
    geometry_evidence,
    identity,
    load_cached_feature,
    load_pairs,
)
from sscd_triage import transformed


def deterministic_negatives(keys: list[str], count: int,
                            seed: int) -> list[tuple[str, str]]:
    maximum = len(keys) * (len(keys) - 1) // 2
    if count > maximum:
        raise ValueError(f"Requested {count} negatives exceeds possible {maximum}")
    rng, selected = random.Random(seed), set()
    while len(selected) < count:
        left, right = rng.sample(keys, 2)
        selected.add(tuple(sorted((left, right))))
    return sorted(selected)


def validate_calibration_resume(path: Path, expected: set[str]) -> set[str]:
    completed = set()
    if not Path(path).exists():
        return completed
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"Truncated calibration ledger at line {number}")
            row = json.loads(line)
            calibration_id = row.get("calibration_id")
            if calibration_id not in expected:
                raise RuntimeError(f"Unknown calibration_id: {calibration_id}")
            if calibration_id in completed:
                raise RuntimeError(f"Duplicate calibration_id: {calibration_id}")
            completed.add(calibration_id)
    return completed


def image_token(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def pil_tensor(image: Image.Image):
    import torch
    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)))


def match_evidence(matcher, rbd, left: dict, right: dict) -> dict:
    import torch
    with torch.inference_mode():
        matched = rbd(matcher({"image0": left, "image1": right}))
    left_u, right_u = rbd(left), rbd(right)
    indices = matched["matches"].detach().cpu().numpy()
    scores = matched["scores"].detach().float().cpu().numpy()
    left_points = left_u["keypoints"].detach().cpu().numpy()[indices[:, 0]]
    right_points = right_u["keypoints"].detach().cpu().numpy()[indices[:, 1]]
    result = geometry_evidence(left_points, right_points, scores)
    result["left_keypoints"] = int(left_u["keypoints"].shape[0])
    result["right_keypoints"] = int(right_u["keypoints"].shape[0])
    return result


def append_records(path: Path, tasks: list[tuple], expected: set[str],
                   worker, progress_path: Path, run_id: str,
                   label: str) -> None:
    completed = validate_calibration_resume(path, expected)
    mode, added, started = ("a" if path.exists() else "w"), 0, time.monotonic()
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for task in tasks:
            calibration_id = task[0]
            if calibration_id in completed:
                continue
            record = worker(task)
            record["calibration_id"] = calibration_id
            record["kind"] = label
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            completed.add(calibration_id)
            added += 1
            if added % 25 == 0 or len(completed) == len(tasks):
                handle.flush()
                os.fsync(handle.fileno())
                elapsed = max(time.monotonic() - started, 1e-9)
                atomic_json(progress_path, {
                    "run_id": run_id,
                    "stage": label,
                    "completed": len(completed),
                    "total": len(tasks),
                    "records_per_second_since_resume": added / elapsed,
                })
                print(f"LightGlue calibration {label}: {len(completed)}/{len(tasks)}",
                      flush=True)


def run(args: argparse.Namespace) -> dict:
    import torch
    sys.path.insert(0, str(args.vendor.resolve()))
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import rbd

    pairs = load_pairs(args.pairs)
    keys = sorted({row[field] for row in pairs for field in ("left", "right")})
    if len(keys) < args.positive_images:
        raise RuntimeError("Insufficient unique images for calibration")
    feature_complete = json.loads(
        (args.evidence / "LIGHTGLUE_EVIDENCE_COMPLETE.json").read_text()
    )
    rng = random.Random(args.seed)
    positive_keys = sorted(rng.sample(keys, args.positive_images))
    positives = [
        (f"positive:{image_token(key)}:{variant}", key, variant)
        for key in positive_keys for variant in range(5)
    ]
    negatives_raw = deterministic_negatives(keys, args.negative_pairs, args.seed)
    negatives = [
        (f"negative:{image_token(left)}:{image_token(right)}", left, right)
        for left, right in negatives_raw
    ]
    config = {
        "code_sha256": digest(Path(__file__)),
        "pairs_sha256": digest(args.pairs),
        "feature_run_id": feature_complete["run_id"],
        "positive_images": args.positive_images,
        "positive_transformations": len(positives),
        "negative_pairs": len(negatives),
        "seed": args.seed,
        "resize": args.resize,
        "max_keypoints": args.max_keypoints,
    }
    run_id = identity(config)
    args.out.mkdir(parents=True, exist_ok=True)
    identity_path = args.out / "LIGHTGLUE_CALIBRATION_IDENTITY.json"
    if identity_path.exists() and json.loads(identity_path.read_text()).get("run_id") != run_id:
        raise RuntimeError("Calibration output identity mismatch")
    atomic_json(identity_path, {"run_id": run_id, **config})
    incomplete = args.out / "LIGHTGLUE_CALIBRATION_INCOMPLETE"
    incomplete.write_text(run_id + "\n", encoding="utf-8")

    extractor = ALIKED(max_num_keypoints=args.max_keypoints).eval().to(args.device)
    matcher = LightGlue(features="aliked", filter_threshold=0.1,
                        mp=True).eval().to(args.device)
    cache = args.evidence / "aliked_features"

    def positive_worker(task):
        _, key, variant = task
        base = load_cached_feature(feature_path(cache, key), args.device)
        with Image.open(args.data / key) as image:
            changed = pil_tensor(transformed(image.convert("RGB"), variant)).to(args.device)
        with torch.inference_mode():
            changed_features = {
                name: value.to(args.device, non_blocking=True)
                for name, value in cpu_feature(
                    extractor.extract(changed, resize=args.resize)
                ).items()
            }
        record = match_evidence(matcher, rbd, base, changed_features)
        record["transform_variant"] = variant
        return record

    positive_path = args.out / "lightglue_calibration_positive.jsonl"
    append_records(positive_path, positives, {x[0] for x in positives},
                   positive_worker, args.out / "LIGHTGLUE_CALIBRATION_PROGRESS.json",
                   run_id, "transformed_positive")
    del extractor
    torch.cuda.empty_cache()

    def negative_worker(task):
        _, left, right = task
        return match_evidence(
            matcher, rbd,
            load_cached_feature(feature_path(cache, left), args.device),
            load_cached_feature(feature_path(cache, right), args.device),
        )

    negative_path = args.out / "lightglue_calibration_negative.jsonl"
    append_records(negative_path, negatives, {x[0] for x in negatives},
                   negative_worker, args.out / "LIGHTGLUE_CALIBRATION_PROGRESS.json",
                   run_id, "presumed_negative")
    if len(validate_calibration_resume(positive_path, {x[0] for x in positives})) != len(positives):
        raise RuntimeError("Positive calibration incomplete")
    if len(validate_calibration_resume(negative_path, {x[0] for x in negatives})) != len(negatives):
        raise RuntimeError("Negative calibration incomplete")
    summary = {
        "purpose": "calibration evidence only; no thresholds or decisions",
        "run_id": run_id,
        **config,
        "artifacts": {
            positive_path.name: digest(positive_path),
            negative_path.name: digest(negative_path),
        },
    }
    atomic_json(args.out / "LIGHTGLUE_CALIBRATION_COMPLETE.json", summary)
    incomplete.unlink()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--positive-images", type=int, default=250)
    parser.add_argument("--negative-pairs", type=int, default=10000)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
