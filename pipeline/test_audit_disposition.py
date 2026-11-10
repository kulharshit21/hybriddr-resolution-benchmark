import unittest
import json
import tempfile
from pathlib import Path

# Windows CUDA wheels require Torch DLLs to load before SciPy's native stack.
import torch  # noqa: F401
import pandas as pd
import numpy as np

from audit_duplicates import (
    build_review_records, disposition_manifest, finalize_audit,
    validate_manual_review,
)
from run import validate_data_review
from common import digest


class AuditDispositionTests(unittest.TestCase):
    def test_development_external_overlap_removes_linked_development_patient(self):
        manifest = pd.DataFrame([
            {'path': 'dev-a', 'source': 'EyePACS', 'role': 'development', 'split': 'train',
             'patient': 'p1', 'original_patient': 'p1', 'label': 1},
            {'path': 'dev-c', 'source': 'EyePACS', 'role': 'development', 'split': 'train',
             'patient': 'p1', 'original_patient': 'p1', 'label': 1},
            {'path': 'ext-b', 'source': 'MESSIDOR2', 'role': 'external', 'split': 'external',
             'patient': 'm1', 'original_patient': 'm1', 'label': 1},
        ])
        child, exclusions = disposition_manifest(
            manifest, [{'left': 'dev-a', 'right': 'ext-b'}], seed=2026)
        self.assertEqual(set(child.path), {'ext-b'})
        self.assertEqual({row['path'] for row in exclusions}, {'dev-a', 'dev-c'})

    def test_conflicting_labels_exclude_every_member(self):
        manifest = pd.DataFrame([
            {'path': 'a', 'source': 'APTOS', 'role': 'development', 'split': 'train',
             'patient': 'a', 'original_patient': 'a', 'label': 0},
            {'path': 'b', 'source': 'APTOS', 'role': 'development', 'split': 'val',
             'patient': 'b', 'original_patient': 'b', 'label': 4},
        ])
        child, exclusions = disposition_manifest(manifest, [{'left': 'a', 'right': 'b'}], seed=2026)
        self.assertTrue(child.empty)
        self.assertEqual({row['reason'] for row in exclusions},
                         {'confirmed_near_duplicate_label_conflict'})

    def test_external_duplicates_keep_one_deterministic_representative(self):
        manifest = pd.DataFrame([
            {'path': 'z', 'source': 'MESSIDOR2', 'role': 'external', 'split': 'external',
             'patient': 'z', 'original_patient': 'z', 'label': 2},
            {'path': 'a', 'source': 'MESSIDOR2', 'role': 'external', 'split': 'external',
             'patient': 'a', 'original_patient': 'a', 'label': 2},
        ])
        child, exclusions = disposition_manifest(manifest, [{'left': 'z', 'right': 'a'}], seed=2026)
        self.assertEqual(child.path.tolist(), ['a'])
        self.assertEqual(exclusions[0]['path'], 'z')

    def test_development_duplicates_are_grouped_before_split_regeneration(self):
        rows = []
        for label in range(5):
            for index in range(20):
                patient = f'p{label}-{index}'
                rows.append({'path': patient, 'source': 'EyePACS', 'role': 'development',
                             'split': 'train' if index < 14 else 'val', 'patient': patient,
                             'original_patient': patient, 'label': label})
        manifest = pd.DataFrame(rows)
        child, _ = disposition_manifest(
            manifest, [{'left': 'p0-0', 'right': 'p0-15'}], seed=2026)
        linked = child[child.path.isin(['p0-0', 'p0-15'])]
        self.assertEqual(linked.patient.nunique(), 1)
        self.assertEqual(linked.split.nunique(), 1)

    def test_cross_source_linked_development_group_has_one_split(self):
        rows = []
        for source in ('EyePACS', 'APTOS'):
            for label in range(5):
                for index in range(12):
                    patient = f'{source}-{label}-{index}'
                    rows.append({'path': patient, 'source': source, 'role': 'development',
                                 'split': 'train', 'patient': patient,
                                 'original_patient': patient, 'label': label})
        manifest = pd.DataFrame(rows)
        pair = {'left': 'EyePACS-0-0', 'right': 'APTOS-0-0'}
        child, _ = disposition_manifest(manifest, [pair], seed=2026)
        linked = child[child.path.isin([pair['left'], pair['right']])]
        self.assertEqual(linked.patient.nunique(), 1)
        self.assertEqual(linked.split.nunique(), 1)

    def test_external_contamination_propagates_through_patient_graph(self):
        manifest = pd.DataFrame([
            {'path': 'p1-a', 'source': 'EyePACS', 'role': 'development', 'split': 'train',
             'patient': 'p1', 'original_patient': 'p1', 'label': 1},
            {'path': 'p2-b', 'source': 'APTOS', 'role': 'development', 'split': 'val',
             'patient': 'p2', 'original_patient': 'p2', 'label': 1},
            {'path': 'p2-c', 'source': 'APTOS', 'role': 'development', 'split': 'val',
             'patient': 'p2', 'original_patient': 'p2', 'label': 1},
            {'path': 'ext', 'source': 'MESSIDOR2', 'role': 'external', 'split': 'external',
             'patient': 'm1', 'original_patient': 'm1', 'label': 1},
        ])
        pairs = [{'left': 'p1-a', 'right': 'p2-b'}, {'left': 'p2-c', 'right': 'ext'}]
        child, _ = disposition_manifest(manifest, pairs, seed=2026)
        self.assertEqual(child.path.tolist(), ['ext'])

    def test_review_pack_is_blinded_and_all_borderlines_are_required(self):
        pairs = [
            {'pair_id': '1', 'left': 'secret-a', 'right': 'secret-b',
             'classification': 'borderline', 'label': 4, 'split': 'external'},
            {'pair_id': '2', 'left': 'secret-c', 'right': 'secret-d',
             'classification': 'rejected', 'label': 0, 'split': 'train'},
        ]
        review = build_review_records(pairs, sample_seed=7, sample_each=1)
        self.assertTrue(all(set(row) <= {'pair_id', 'left_token', 'right_token'} for row in review))
        with self.assertRaises(RuntimeError):
            validate_manual_review(pairs, [])
        decisions = [{'pair_id': '1', 'decision': 'confirmed', 'reviewer': 'r1',
                      'timestamp': '2026-11-03T10:00:00+05:30'}]
        validate_manual_review(pairs, decisions)

    def test_training_review_must_match_manifest_parent_and_audit(self):
        summary = {'manifest_sha256': 'child', 'parent_manifest_sha256': 'parent',
                   'audit_id': 'audit'}
        valid = {'approved': True, **summary}
        validate_data_review(summary, valid)
        for key in ('manifest_sha256', 'parent_manifest_sha256', 'audit_id'):
            invalid = dict(valid)
            invalid[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_data_review(summary, invalid)

    def test_finalization_writes_child_manifest_without_mutating_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root / 'parent'
            audit = root / 'audit'
            child = root / 'child'
            parent.mkdir()
            audit.mkdir()
            (parent / 'cache').mkdir()
            (parent / 'cache/a.png').write_bytes(b'a')
            (parent / 'cache/b.png').write_bytes(b'b')
            manifest = pd.DataFrame([
                {'path': 'dev-a', 'source': 'EyePACS', 'role': 'development',
                 'split': 'train', 'patient': 'p1', 'original_patient': 'p1',
                 'label': 1, 'cache': 'cache/a.png'},
                {'path': 'ext-b', 'source': 'MESSIDOR2', 'role': 'external',
                 'split': 'external', 'patient': 'm1', 'original_patient': 'm1',
                 'label': 1, 'cache': 'cache/b.png'},
            ])
            manifest.to_csv(parent / 'manifest.csv', index=False)
            original = (parent / 'manifest.csv').read_bytes()
            pair = {'pair_id': 'pair', 'left': 'dev-a', 'right': 'ext-b',
                    'classification': 'confirmed'}
            (audit / 'candidate_pairs.jsonl').write_text(json.dumps(pair) + '\n')
            (audit / 'dino_pairs.jsonl').write_text('')
            (audit / 'fingerprints.jsonl').write_text('{}\n{}\n')
            (audit / 'calibration.json').write_text(json.dumps({'locked': True}))
            np.save(audit / 'dinov2_embeddings.npy', np.ones((2, 3), dtype=np.float32))
            (audit / 'dinov2_keys.json').write_text(json.dumps(['dev-a', 'ext-b']))
            (audit / 'dino_candidates.jsonl').write_text('')
            parent_hash = digest(parent / 'manifest.csv')
            cpu_artifacts = {
                name: {'sha256': digest(audit / name),
                       **({'lines': sum(1 for line in (audit / name).read_text().splitlines() if line.strip())}
                          if name.endswith('.jsonl') else {})}
                for name in ('fingerprints.jsonl', 'candidate_pairs.jsonl', 'calibration.json')
            }
            dino_artifacts = {
                name: {'sha256': digest(audit / name),
                       **({'lines': sum(1 for line in (audit / name).read_text().splitlines() if line.strip())}
                          if name.endswith('.jsonl') else {})}
                for name in ('dinov2_embeddings.npy', 'dinov2_keys.json',
                             'dino_candidates.jsonl', 'dino_pairs.jsonl')
            }
            (audit / 'CPU_COMPLETE.json').write_text(json.dumps({
                'audit_id': 'cpu', 'parent_manifest_sha256': parent_hash,
                'artifacts': cpu_artifacts}))
            (audit / 'DINO_COMPLETE.json').write_text(json.dumps({
                'dino_audit_id': 'dino', 'cpu_audit_id': 'cpu',
                'parent_manifest_sha256': parent_hash, 'artifacts': dino_artifacts}))
            (audit / 'review_required.json').write_text(json.dumps(['pair']))
            manual = root / 'manual.jsonl'
            manual.write_text(json.dumps({'pair_id': 'pair', 'decision': 'confirmed',
                'reviewer': 'r1', 'timestamp': '2026-11-03T10:00:00+05:30'}) + '\n')
            summary = finalize_audit(parent, audit, manual, child, seed=2026)
            result = pd.read_csv(child / 'manifest.csv')
            self.assertEqual(result.path.tolist(), ['ext-b'])
            self.assertEqual((parent / 'manifest.csv').read_bytes(), original)
            self.assertTrue((child / 'cache/b.png').is_file())
            self.assertEqual(summary['parent_manifest_sha256'],
                             json.loads((child / 'prepared.json').read_text())['parent_manifest_sha256'])

    def test_finalization_rejects_incomplete_or_mismatched_stage_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent, audit = root / 'parent', root / 'audit'
            parent.mkdir(); audit.mkdir()
            pd.DataFrame([{'path': 'a', 'source': 'EyePACS', 'role': 'development',
                'split': 'train', 'patient': 'p', 'label': 1, 'cache': 'cache/a'}]).to_csv(
                    parent / 'manifest.csv', index=False)
            (audit / 'CPU_COMPLETE.json').write_text(json.dumps({
                'audit_id': 'cpu', 'parent_manifest_sha256': 'wrong', 'artifacts': {}}))
            (audit / 'DINO_COMPLETE.json').write_text(json.dumps({
                'dino_audit_id': 'dino', 'cpu_audit_id': 'cpu',
                'parent_manifest_sha256': 'wrong', 'artifacts': {}}))
            (audit / 'INCOMPLETE').write_text('stale')
            manual = root / 'manual.jsonl'; manual.write_text('')
            with self.assertRaises(RuntimeError):
                finalize_audit(parent, audit, manual, root / 'child', seed=2026)


if __name__ == '__main__':
    unittest.main(verbosity=2)
