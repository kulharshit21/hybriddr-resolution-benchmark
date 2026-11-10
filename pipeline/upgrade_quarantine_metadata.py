#!/usr/bin/env python3
"""Upgrade an existing quarantine child to the trainer metadata contract."""
import argparse
import json
import os
from pathlib import Path

from build_quarantine_child import digest, quarantine_audit_id


def upgrade(prepared):
    prepared = Path(prepared)
    metadata_path = prepared / 'prepared.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    manifest_hash = digest(prepared / 'manifest.csv')
    recorded = metadata.get('child_manifest_sha256') or metadata.get('manifest_sha256')
    if recorded != manifest_hash:
        raise RuntimeError('Existing child manifest identity does not match manifest.csv')
    parent_hash = metadata.get('parent_manifest_sha256')
    candidate_hash = metadata.get('candidate_ledger_sha256')
    if not parent_hash or not candidate_hash:
        raise RuntimeError('Parent or candidate-ledger identity is missing')
    metadata['manifest_sha256'] = manifest_hash
    metadata['child_manifest_sha256'] = manifest_hash
    metadata['audit_id'] = quarantine_audit_id(parent_hash, candidate_hash)
    temporary = metadata_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, metadata_path)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.prepared), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
