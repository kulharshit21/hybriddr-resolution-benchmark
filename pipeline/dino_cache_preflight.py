"""Real-file entry point for spawned real-pair cache equivalence workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import identity
from dino_confirm_cached import build_cache, validate_cache_equivalence
from near_duplicates import AuditConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    project = args.project.resolve()
    audit = args.audit.resolve()
    data = project / "Datasets"
    config = AuditConfig.from_json(project / "pipeline/near_duplicate_config.json")
    candidates = []
    with (audit / "dino_candidates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row.pop("audit_id", None)
                candidates.append(row)
            if len(candidates) == args.pairs:
                break
    if len(candidates) != args.pairs:
        raise RuntimeError(
            f"Needed {args.pairs} candidates; found {len(candidates)}")
    keys = sorted({row[key] for row in candidates for key in ("left", "right")})
    cache_id = identity({"purpose": "real-pair-cache-preflight-v1", "keys": keys})
    build_cache(data, keys, args.out.resolve(), config, args.workers, cache_id)
    result = validate_cache_equivalence(
        candidates, data, keys, args.out.resolve(), config,
        args.pairs, "preflight")
    print(json.dumps(result, indent=2), flush=True)
    if result["maximum_absolute_difference"] != 0.0:
        raise RuntimeError("Real-pair cache preflight was not bit-exact")
    print("REAL_PAIR_CACHE_EQUIVALENCE: PASS", flush=True)


if __name__ == "__main__":
    main()
