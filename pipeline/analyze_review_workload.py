#!/usr/bin/env python3
"""Summarize the frozen blinded-review workload without changing audit state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


RANK = {"rejected": 0, "borderline": 1, "confirmed": 2}
METRICS = (
    "ssim", "ncc", "nmae", "mask_iou", "vessel_ncc", "vessel_dice",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["path"]: {"source": row["source"], "split": row["split"]}
        for row in rows
    }


def analyze(audit: Path, manifest_path: Path, output: Path) -> dict:
    audit, manifest_path, output = map(Path, (audit, manifest_path, output))
    required_path = audit / "review_required.json"
    required_rows = json.loads(required_path.read_text(encoding="utf-8"))
    required = set(required_rows)
    if len(required) != len(required_rows):
        raise RuntimeError("review_required.json contains duplicate IDs")
    manifest = load_manifest(manifest_path)
    records: dict[str, dict] = {}
    origins: dict[str, set[str]] = defaultdict(set)
    stage_classes: dict[str, dict[str, str]] = defaultdict(dict)
    inputs = {
        "cpu": audit / "candidate_pairs.jsonl",
        "dino": audit / "dino_pairs.jsonl",
    }
    for stage, path in inputs.items():
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                pair_id = row.get("pair_id")
                if pair_id not in required:
                    continue
                classification = row.get("classification")
                if classification not in RANK:
                    raise RuntimeError(f"Invalid classification at {path}:{number}")
                origins[pair_id].add(stage)
                stage_classes[pair_id][stage] = classification
                existing = records.get(pair_id)
                if existing is None or RANK[classification] > RANK[existing["classification"]]:
                    records[pair_id] = row
    missing = sorted(required - records.keys())
    if missing:
        raise RuntimeError(f"Required review IDs missing from evidence: {len(missing)}")

    classification_counts = Counter()
    origin_counts = Counter()
    source_pairs = Counter()
    split_pairs = Counter()
    protected = Counter()
    stage_agreement = Counter()
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    compact = []
    strict_existing = Counter()
    for pair_id in sorted(required):
        row = records[pair_id]
        classification = row["classification"]
        classification_counts[classification] += 1
        origin = "+".join(sorted(origins[pair_id]))
        origin_counts[origin] += 1
        left, right = row["left"], row["right"]
        if left not in manifest or right not in manifest:
            raise RuntimeError(f"Pair path absent from prepared manifest: {pair_id}")
        sources = tuple(sorted({manifest[left]["source"], manifest[right]["source"]}))
        splits = tuple(sorted({manifest[left]["split"], manifest[right]["split"]}))
        source_pairs[sources] += 1
        split_pairs[splits] += 1
        protected["cross_split" if len(splits) > 1 else "same_split"] += 1
        classes = stage_classes[pair_id]
        if len(classes) == 2:
            stage_agreement[
                "agree" if classes["cpu"] == classes["dino"] else
                f"cpu_{classes['cpu']}__dino_{classes['dino']}"
            ] += 1
        evidence = row.get("evidence", {})
        for metric in METRICS:
            value = evidence.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[classification][metric].append(float(value))
        for metric in ("cosine", "hamming_distance"):
            value = row.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[classification][metric].append(float(value))
        if (evidence.get("ssim", -1) >= 0.90 and
                evidence.get("ncc", -1) >= 0.95 and
                evidence.get("nmae", 1) <= 0.12 and
                evidence.get("vessel_dice", 0) >= 0.75):
            strict_existing["strict_positive_geometry"] += 1
        elif (evidence.get("ssim", 1) < 0.60 or
              evidence.get("ncc", 1) < 0.65 or
              evidence.get("nmae", 0) > 0.35 or
              evidence.get("vessel_dice", 1) < 0.35):
            strict_existing["strict_negative_geometry"] += 1
        else:
            strict_existing["unresolved_geometry"] += 1
        compact.append({
            "pair_id": pair_id,
            "left": left,
            "right": right,
            "selected_classification": classification,
            "origins": sorted(origins[pair_id]),
            "stage_classifications": classes,
            "sources": list(sources),
            "splits": list(splits),
            "retrieval": {
                key: row[key] for key in ("cosine", "hamming_distance") if key in row
            },
            "evidence": evidence,
        })

    output.mkdir(parents=True, exist_ok=False)
    compact_path = output / "required_pairs_private.jsonl"
    with compact_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in compact:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "purpose": "read-only diagnostic; not adjudication and not a training artifact",
        "required_pairs": len(required),
        "classification_counts": dict(sorted(classification_counts.items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "source_pair_counts": {" + ".join(key): value for key, value in source_pairs.items()},
        "split_pair_counts": {" + ".join(key): value for key, value in split_pairs.items()},
        "boundary_counts": dict(sorted(protected.items())),
        "cpu_dino_agreement": dict(sorted(stage_agreement.items())),
        "provisional_existing_geometry_buckets": dict(sorted(strict_existing.items())),
        "metric_distributions": {
            classification: {
                metric: distribution(metric_values)
                for metric, metric_values in sorted(group.items())
            }
            for classification, group in sorted(values.items())
        },
        "inputs": {
            "review_required_sha256": digest(required_path),
            "manifest_sha256": digest(manifest_path),
            "candidate_pairs_sha256": digest(inputs["cpu"]),
            "dino_pairs_sha256": digest(inputs["dino"]),
        },
        "private_pair_ledger": {
            "path": str(compact_path),
            "lines": len(compact),
            "sha256": digest(compact_path),
        },
        "limitations": [
            "Provisional geometry buckets are diagnostic only and must not be used as labels",
            "No independent SSCD or local-feature evidence is included yet",
            "No human decisions are inferred or fabricated",
        ],
    }
    (output / "review_workload_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.audit, args.manifest, args.out), indent=2))


if __name__ == "__main__":
    main()
