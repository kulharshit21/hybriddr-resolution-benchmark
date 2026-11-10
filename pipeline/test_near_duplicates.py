import json
from io import BytesIO
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from scipy import ndimage

from near_duplicates import (
    AuditConfig, AuditState, HammingIndex, NormalizedRetina, PairEvidence, build_audit_identity,
    classify_pair, confirm_pair, dino_candidate_record, embed_retinas,
    exact_cosine_neighbors, fingerprint_one, load_dino, normalize_retina,
    perceptual_hash,
)
from audit_duplicates import (
    _candidate_pairs, _validate_bounded_positive_retrieval,
    calibration_binding, load_resume_records, lock_calibration,
    require_locked_calibration_before_external, run_cpu,
    select_hard_negative_pairs, wilson_upper,
)
from common import digest


class AuditIdentityStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.csv'
        self.manifest.write_text('path,sha256\na.png,aaa\n', encoding='utf-8')
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps({'schema_version': 1, 'hash_bits': 256}), encoding='utf-8')
        self.code = self.root / 'code.py'
        self.code.write_text('VERSION = 1\n', encoding='utf-8')
        self.lock = self.root / 'requirements.txt'
        self.lock.write_text('numpy==2.2.6\n', encoding='utf-8')
        self.weights = self.root / 'weights.bin'
        self.weights.write_bytes(b'weights-v1')

    def tearDown(self):
        self.temp.cleanup()

    def identity(self):
        return build_audit_identity(
            self.manifest,
            self.root,
            AuditConfig.from_json(self.config_path),
            [self.code],
            self.lock,
            self.weights,
        )

    def test_identity_is_stable_and_covers_every_frozen_input(self):
        first = self.identity()
        self.assertEqual(first, self.identity())
        cases = [
            (self.manifest, 'path,sha256\na.png,bbb\n'),
            (self.config_path, json.dumps({'schema_version': 2, 'hash_bits': 256})),
            (self.code, 'VERSION = 2\n'),
            (self.lock, 'numpy==2.2.7\n'),
        ]
        for path, content in cases:
            with self.subTest(path=path.name):
                original = path.read_text(encoding='utf-8')
                path.write_text(content, encoding='utf-8')
                self.assertNotEqual(first['audit_id'], self.identity()['audit_id'])
                path.write_text(original, encoding='utf-8')
        self.weights.write_bytes(b'weights-v2')
        self.assertNotEqual(first['audit_id'], self.identity()['audit_id'])

    def test_state_fails_closed_until_expected_artifacts_exist(self):
        state = AuditState(self.root / 'audit')
        identity = self.identity()
        state.start(identity)
        self.assertTrue((state.output / 'INCOMPLETE').exists())
        with self.assertRaises(RuntimeError):
            state.require_complete()
        with self.assertRaises(FileNotFoundError):
            state.complete({'expected_artifacts': {'rows.jsonl': 2}})
        (state.output / 'rows.jsonl').write_text('{}\n{}\n', encoding='utf-8')
        state.complete({'expected_artifacts': {'rows.jsonl': 2}})
        self.assertFalse((state.output / 'INCOMPLETE').exists())
        self.assertEqual(state.require_complete()['audit_id'], identity['audit_id'])

    def test_restarting_with_another_identity_is_rejected(self):
        state = AuditState(self.root / 'audit')
        state.start(self.identity())
        self.code.write_text('VERSION = 99\n', encoding='utf-8')
        with self.assertRaises(RuntimeError):
            state.start(self.identity())


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AuditConfig.from_json(Path(__file__).parent / 'near_duplicate_config.json')

    @staticmethod
    def retina(size=320, offset=(0, 0), variant=0):
        image = Image.new('RGB', (size, size), 'black')
        draw = ImageDraw.Draw(image)
        cx, cy = size // 2 + offset[0], size // 2 + offset[1]
        radius = int(size * 0.4)
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(135, 48, 28))
        draw.ellipse((cx-18-variant*3, cy-14, cx+18-variant*3, cy+22), fill=(230, 150, 90))
        for angle in range(0, 360, 30):
            radians = np.deg2rad(angle + variant * 7)
            end = (cx + int(radius * .85 * np.cos(radians)), cy + int(radius * .85 * np.sin(radians)))
            draw.line((cx-variant*3, cy, *end), fill=(90, 20 + variant * 12, 18), width=3)
        draw.ellipse((cx + 55, cy - 45 - variant * 8, cx + 66, cy - 34 - variant * 8), fill=(245, 210, 100))
        return image

    def distance(self, left, right):
        return (perceptual_hash(normalize_retina(left, self.config).gray, self.config) ^
                perceptual_hash(normalize_retina(right, self.config).gray, self.config)).bit_count()

    def test_roi_is_stable_under_border_resize_and_jpeg(self):
        base = self.retina()
        bordered = Image.new('RGB', (420, 380), 'black')
        bordered.paste(base, (50, 30))
        resized = base.resize((510, 510), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        base.save(buffer, format='JPEG', quality=82)
        jpeg = Image.open(BytesIO(buffer.getvalue())).convert('RGB')
        radius = self.config.phash['candidate_hamming_radius']
        for transformed in (bordered, resized, jpeg):
            with self.subTest(size=transformed.size):
                self.assertLessEqual(self.distance(base, transformed), radius)

    def test_flip_and_small_rotation_have_a_close_orientation(self):
        base = self.retina()
        normalized = normalize_retina(base, self.config)
        base_hashes = {
            perceptual_hash(normalized.gray, self.config),
            perceptual_hash(np.fliplr(normalized.gray), self.config),
        }
        for transformed in (
            base.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            base.rotate(3, resample=Image.Resampling.BICUBIC, fillcolor='black'),
        ):
            other = normalize_retina(transformed, self.config)
            hashes = {
                perceptual_hash(other.gray, self.config),
                perceptual_hash(np.fliplr(other.gray), self.config),
            }
            distance = min((a ^ b).bit_count() for a in base_hashes for b in hashes)
            self.assertLessEqual(distance, self.config.phash['candidate_hamming_radius'])

    def test_different_retinas_do_not_share_the_same_fingerprint(self):
        left = perceptual_hash(normalize_retina(self.retina(variant=0), self.config).gray, self.config)
        right = perceptual_hash(normalize_retina(self.retina(variant=3), self.config).gray, self.config)
        self.assertNotEqual(left, right)

    def test_invalid_or_tiny_foreground_fails_closed(self):
        black = Image.new('RGB', (320, 320), 'black')
        tiny = black.copy()
        ImageDraw.Draw(tiny).ellipse((150, 150, 170, 170), fill=(120, 40, 20))
        near_black = Image.new('RGB', (320, 320), (8, 1, 1))
        for image in (black, tiny, near_black):
            with self.assertRaisesRegex(ValueError, 'fixture-path'):
                normalize_retina(image, self.config, 'fixture-path')

    def test_border_filling_retina_is_flagged_instead_of_rejected(self):
        cropped = self.retina().crop((80, 80, 240, 240))
        normalized = normalize_retina(cropped, self.config, 'border-filled-fixture')
        self.assertTrue(normalized.edge_truncated)

    def test_fingerprint_records_input_and_both_orientations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'retina.png'
            self.retina().save(path)
            record = fingerprint_one(path, self.config)
            self.assertEqual(record.path, path.as_posix())
            self.assertEqual(len(record.input_sha256), 64)
            self.assertIsInstance(record.phash, int)
            self.assertIsInstance(record.phash_flipped, int)


class RetrievalConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AuditConfig.from_json(Path(__file__).parent / 'near_duplicate_config.json')

    def test_metric_index_matches_brute_force_and_excludes_self(self):
        values = {'a': 0b0000, 'b': 0b0011, 'c': 0b1111, 'd': 0b0101}
        index = HammingIndex()
        for key, value in values.items():
            index.add(key, value)
        for key, value in values.items():
            expected = sorted((other, (value ^ candidate).bit_count())
                              for other, candidate in values.items()
                              if other != key and (value ^ candidate).bit_count() <= 2)
            self.assertEqual(sorted(index.query(value, 2, exclude_key=key)), expected)

    def test_transformed_copy_confirms_and_unrelated_image_rejects(self):
        base = NormalizationTests.retina()
        buffer = BytesIO()
        base.rotate(1, resample=Image.Resampling.BICUBIC, fillcolor='black').save(
            buffer, format='JPEG', quality=90)
        copy = Image.open(BytesIO(buffer.getvalue())).convert('RGB')
        left = normalize_retina(base, self.config)
        right = normalize_retina(copy, self.config)
        unrelated = normalize_retina(NormalizationTests.retina(variant=4), self.config)
        match = confirm_pair(left, right, self.config)
        negative = confirm_pair(left, unrelated, self.config)
        self.assertEqual(classify_pair(match, self.config.confirmation), 'confirmed')
        self.assertEqual(classify_pair(negative, self.config.confirmation), 'rejected')
        self.assertGreater(match.ssim, negative.ssim)

    def test_no_valid_overlap_is_rejection_not_fatal_error(self):
        size = 512
        left_gray = np.zeros((size, size), dtype=np.float32)
        right_gray = np.zeros((size, size), dtype=np.float32)
        left_mask = np.zeros_like(left_gray, dtype=bool)
        right_mask = np.zeros_like(right_gray, dtype=bool)
        left_mask[16:80, 224:288] = True
        right_mask[432:496, 224:288] = True
        left_gray[16:80, 224:288] = np.linspace(0.2, 0.9, 64)[:, None]
        right_gray[432:496, 224:288] = np.linspace(0.9, 0.2, 64)[:, None]
        left = NormalizedRetina(
            rgb=np.zeros((size, size, 3), dtype=np.uint8), gray=left_gray,
            mask=left_mask, bbox=(224, 16, 288, 80),
            foreground_fraction=float(left_mask.mean()),
        )
        right = NormalizedRetina(
            rgb=np.zeros((size, size, 3), dtype=np.uint8), gray=right_gray,
            mask=right_mask, bbox=(224, 432, 288, 496),
            foreground_fraction=float(right_mask.mean()),
        )

        evidence = confirm_pair(left, right, self.config)

        self.assertEqual(classify_pair(evidence, self.config.confirmation), 'rejected')
        self.assertEqual(evidence.orientation, 'none')
        self.assertEqual(evidence.mask_iou, 0.0)
        self.assertEqual(evidence.ssim, -1.0)

    def test_horizontal_flip_selects_flipped_orientation(self):
        base = NormalizationTests.retina()
        flipped = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        evidence = confirm_pair(normalize_retina(base, self.config),
                                normalize_retina(flipped, self.config), self.config)
        self.assertEqual(evidence.orientation, 'flipped')
        self.assertEqual(classify_pair(evidence, self.config.confirmation), 'confirmed')

    def test_borderline_evidence_is_not_confirmed(self):
        evidence = PairEvidence(ssim=0.80, ncc=0.85, nmae=0.22,
                                mask_iou=0.99, orientation='normal',
                                rotation=0.0, scale=1.0, translation=(0.0, 0.0),
                                vessel_dice=0.60)
        self.assertEqual(classify_pair(evidence, self.config.confirmation), 'borderline')

    def test_small_translation_is_estimated_and_confirmed(self):
        base = normalize_retina(NormalizationTests.retina(), self.config)
        shift = (3.0, -4.0)
        moved = type(base)(
            rgb=base.rgb,
            gray=ndimage.shift(base.gray, shift, order=1, mode='constant', cval=0),
            mask=ndimage.shift(base.mask.astype(float), shift, order=0,
                               mode='constant', cval=0) > .5,
            bbox=base.bbox,
            foreground_fraction=base.foreground_fraction,
        )
        evidence = confirm_pair(base, moved, self.config)
        self.assertNotEqual(evidence.translation, (0.0, 0.0))
        self.assertEqual(classify_pair(evidence, self.config.confirmation), 'confirmed')


class AuditOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AuditConfig.from_json(Path(__file__).parent / 'near_duplicate_config.json')

    def test_calibration_locks_only_after_numeric_targets_pass(self):
        positive_distances = [10] * 999 + [80]
        negatives = [PairEvidence(0.3, 0.1, 0.4, 0.9, 'normal', 0, 1, (0, 0))
                     for _ in range(10000)]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'calibration.json'
            record = lock_calibration(positive_distances, negatives, self.config, output)
            self.assertTrue(record['locked'])
            self.assertEqual(record['retrieval_recall'], 0.999)
            self.assertLess(record['false_confirmation_upper_95'], 0.001)
            self.assertEqual(json.loads(output.read_text())['config_sha256'], record['config_sha256'])

    def test_calibration_rejects_low_recall_or_false_confirmations(self):
        safe = PairEvidence(0.3, 0.1, 0.4, 0.9, 'normal', 0, 1, (0, 0))
        false_match = PairEvidence(0.99, 0.999, 0.001, 0.99, 'normal', 0, 1,
                                   (0, 0), vessel_dice=0.9)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'calibration.json'
            with self.assertRaises(RuntimeError):
                lock_calibration([100] * 1000, [safe] * 10000, self.config, output)
            with self.assertRaises(RuntimeError):
                lock_calibration([10] * 1000, [false_match] * 20 + [safe] * 9980,
                                 self.config, output)
            self.assertFalse(output.exists())

    def test_calibration_counts_only_end_to_end_false_confirmations(self):
        safe = PairEvidence(0.3, 0.1, 0.4, 0.9, 'normal', 0, 1, (0, 0))
        apparent = PairEvidence(0.99, 0.999, 0.001, 0.99, 'normal', 0, 1,
                                (0, 0), vessel_dice=0.9)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'calibration.json'
            record = lock_calibration(
                [10] * 1000, [apparent] * 100 + [safe] * 9900,
                self.config, output,
                negative_distances=[65] * 100 + [10] * 9900)
            self.assertEqual(record['false_confirmations'], 0)

    def test_external_manifest_is_blocked_until_calibration_is_locked(self):
        manifest = pd.DataFrame([
            {'path': 'EyePACS/a.jpeg', 'role': 'development'},
            {'path': 'MESSIDOR2/b.tif', 'role': 'external'},
        ])
        with tempfile.TemporaryDirectory() as temp:
            calibration = Path(temp) / 'calibration.json'
            with self.assertRaises(RuntimeError):
                require_locked_calibration_before_external(manifest, calibration, self.config)
            binding = calibration_binding(manifest, self.config)
            calibration.write_text(json.dumps({'locked': True, **binding}), encoding='utf-8')
            require_locked_calibration_before_external(manifest, calibration, self.config)
            bad = {'locked': True, **binding, 'config_sha256': 'wrong'}
            calibration.write_text(json.dumps(bad), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                require_locked_calibration_before_external(manifest, calibration, self.config)

    def test_resume_rejects_corruption_identity_mismatch_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'records.jsonl'
            path.write_text('{bad json}\n', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                load_resume_records(path, 'path', 'audit-1')
            path.write_text(json.dumps({'path': 'a', 'audit_id': 'other'}) + '\n', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                load_resume_records(path, 'path', 'audit-1')
            row = json.dumps({'path': 'a', 'audit_id': 'audit-1'})
            path.write_text(row + '\n' + row + '\n', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                load_resume_records(path, 'path', 'audit-1')
            path.write_text(row + '\n', encoding='utf-8')
            self.assertEqual(set(load_resume_records(path, 'path', 'audit-1')), {'a'})

    def test_wilson_upper_is_conservative_for_zero_errors(self):
        self.assertGreater(wilson_upper(0, 10000), 0)
        self.assertLess(wilson_upper(0, 10000), 0.001)

    def test_hard_negatives_are_unique_and_cross_patient(self):
        frame = pd.DataFrame([
            {'path': 'a', 'patient': 'p1'}, {'path': 'b', 'patient': 'p1'},
            {'path': 'c', 'patient': 'p2'}, {'path': 'd', 'patient': 'p3'},
        ])
        fingerprints = {'a': 0, 'b': 1, 'c': 3, 'd': 15}
        pairs = select_hard_negative_pairs(frame, fingerprints, 3)
        patient = dict(zip(frame.path, frame.patient))
        self.assertEqual(len(pairs), len(set((left, right) for _, left, right in pairs)))
        self.assertTrue(all(patient[left] != patient[right] for _, left, right in pairs))
        with self.assertRaises(RuntimeError):
            select_hard_negative_pairs(frame, fingerprints, 6)

    def test_bounded_candidates_match_deterministic_top_k(self):
        values = {'a': 0, 'b': 1, 'c': 3, 'd': 15}
        records = {
            key: {'phash': f'{value:064x}', 'phash_flipped': f'{value:064x}'}
            for key, value in values.items()
        }
        pairs = _candidate_pairs(records, radius=4, top_k=1, workers=2, chunk_size=2)
        observed = {(row['left'], row['right'], row['hamming_distance']) for row in pairs}
        self.assertEqual(observed, {('a', 'b', 1), ('b', 'c', 1), ('c', 'd', 2)})

    def test_bounded_full_corpus_positive_validation_is_fail_closed(self):
        records = {
            'a': {'phash': f'{0:064x}', 'phash_flipped': f'{0:064x}'},
            'b': {'phash': f'{1:064x}', 'phash_flipped': f'{1:064x}'},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            positives = root / 'positives.jsonl'
            positives.write_text(json.dumps({
                'path': 'a', 'transform_number': 0,
                'phash': f'{0:064x}', 'phash_flipped': f'{0:064x}',
                'classification': 'confirmed',
            }) + '\n', encoding='utf-8')
            result = _validate_bounded_positive_retrieval(
                records, positives, self.config, root)
            self.assertEqual(result['bounded_end_to_end_recall'], 1.0)
            row = json.loads((root / 'bounded_positive_ranks.json').read_text())[0]
            self.assertEqual(row['source_rank'], 1)

            positives.write_text(json.dumps({
                'path': 'a', 'transform_number': 0,
                'phash': f'{0:064x}', 'phash_flipped': f'{0:064x}',
                'classification': 'rejected',
            }) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'full-corpus retrieval recall failed'):
                _validate_bounded_positive_retrieval(records, positives, self.config, root)

    def test_cpu_audit_is_resumable_and_preserves_parent_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            prepared = root / 'prepared'
            audit = root / 'audit'
            (data / 'EyePACS').mkdir(parents=True)
            (data / 'MESSIDOR2').mkdir(parents=True)
            prepared.mkdir()
            image = NormalizationTests.retina()
            image.save(data / 'EyePACS/a.png')
            image.save(data / 'MESSIDOR2/b.png')
            manifest = pd.DataFrame([
                {'path': 'EyePACS/a.png', 'source': 'EyePACS', 'role': 'development',
                 'split': 'train', 'patient': 'p1', 'label': 1,
                 'sha256': digest(data / 'EyePACS/a.png')},
                {'path': 'MESSIDOR2/b.png', 'source': 'MESSIDOR2', 'role': 'external',
                 'split': 'external', 'patient': 'm1', 'label': 1,
                 'sha256': digest(data / 'MESSIDOR2/b.png')},
            ])
            manifest.to_csv(prepared / 'manifest.csv', index=False)
            original = (prepared / 'manifest.csv').read_bytes()
            audit.mkdir()
            binding = calibration_binding(manifest, self.config)
            (audit / 'calibration.json').write_text(
                json.dumps({'locked': True, **binding}), encoding='utf-8')
            source_fingerprint = fingerprint_one(data / 'EyePACS/a.png', self.config)
            (audit / 'calibration_positive_evidence.jsonl').write_text(
                json.dumps({
                    'path': 'EyePACS/a.png', 'transform_number': 0,
                    'phash': f'{source_fingerprint.phash:064x}',
                    'phash_flipped': f'{source_fingerprint.phash_flipped:064x}',
                    'classification': 'confirmed',
                }) + '\n', encoding='utf-8')
            first = run_cpu(data, prepared, audit,
                            Path(__file__).parent / 'near_duplicate_config.json', 1)
            second = run_cpu(data, prepared, audit,
                             Path(__file__).parent / 'near_duplicate_config.json', 1)
            self.assertEqual(first, second)
            self.assertEqual(first['fingerprints'], 2)
            self.assertEqual(first['protected_boundary_candidates'], 1)
            self.assertTrue((audit / 'CPU_COMPLETE.json').exists())
            self.assertEqual((prepared / 'manifest.csv').read_bytes(), original)
            (data / 'EyePACS/a.png').unlink()
            with self.assertRaises(FileNotFoundError):
                run_cpu(data, prepared, audit,
                        Path(__file__).parent / 'near_duplicate_config.json', 1)

    def test_zero_candidate_cpu_audit_still_writes_empty_pair_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, prepared, audit = root / 'data', root / 'prepared', root / 'audit'
            (data / 'EyePACS').mkdir(parents=True)
            prepared.mkdir()
            audit.mkdir()
            image_path = data / 'EyePACS/a.png'
            NormalizationTests.retina().save(image_path)
            manifest = pd.DataFrame([{'path': 'EyePACS/a.png', 'source': 'EyePACS',
                'role': 'development', 'split': 'train', 'patient': 'p1', 'label': 1,
                'sha256': digest(image_path)}])
            manifest.to_csv(prepared / 'manifest.csv', index=False)
            run_cpu(data, prepared, audit,
                    Path(__file__).parent / 'near_duplicate_config.json', 1)
            self.assertTrue((audit / 'candidate_pairs.jsonl').is_file())
            self.assertEqual((audit / 'candidate_pairs.jsonl').read_text(), '')


class DinoAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = AuditConfig.from_json(Path(__file__).parent / 'near_duplicate_config.json')

    def test_weight_hash_mismatch_fails_before_model_construction(self):
        with tempfile.TemporaryDirectory() as temp:
            weights = Path(temp) / 'weights.pt'
            weights.write_bytes(b'not the approved weights')
            with self.assertRaises(RuntimeError):
                load_dino(self.config, weights, '0' * 64, 'cpu')

    def test_tiny_embeddings_are_l2_normalized(self):
        class TinyEncoder(torch.nn.Module):
            def forward(self, tensor):
                return tensor.mean(dim=(2, 3))

        images = [
            ('a', np.full((32, 32, 3), (200, 50, 20), dtype=np.uint8)),
            ('b', np.full((32, 32, 3), (20, 150, 80), dtype=np.uint8)),
        ]
        records = list(embed_retinas(images, TinyEncoder(), self.config, 'cpu'))
        self.assertEqual([record.key for record in records], ['a', 'b'])
        for record in records:
            self.assertAlmostEqual(float(np.linalg.norm(record.vector)), 1.0, places=6)

    def test_chunked_exact_neighbors_match_brute_force_without_self(self):
        rng = np.random.default_rng(42)
        embeddings = rng.normal(size=(13, 8)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        keys = [f'k{i}' for i in range(len(embeddings))]
        result = exact_cosine_neighbors(embeddings, keys, top_k=3, chunk_size=4)
        by_key = {row['key']: row['neighbors'] for row in result}
        cosine = embeddings @ embeddings.T
        for i, key in enumerate(keys):
            expected = [keys[j] for j in np.argsort(-cosine[i]) if j != i][:3]
            self.assertEqual([item['key'] for item in by_key[key]], expected)
            self.assertNotIn(key, expected)

    def test_dino_similarity_alone_is_never_confirmed(self):
        record = dino_candidate_record('a.png', 'b.png', 0.9999)
        self.assertEqual(record['classification'], 'unverified')
        self.assertEqual(record['candidate_source'], 'dinov2')


if __name__ == '__main__':
    unittest.main(verbosity=2)
