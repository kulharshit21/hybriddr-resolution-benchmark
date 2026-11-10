import unittest
import numpy as np
import pandas as pd
import torch
import timm
from common import read_json
from data import assign_splits, resolve_exact_duplicates
from models import Exchange, HybridDR, objective, probabilities
from metrics import measure, fit_temperature, paired_bootstrap
from pathlib import Path
from run import configure_smoke, jobs_for, parse_gpu_ids


class PipelineTests(unittest.TestCase):
    def test_smoke_profile_targets_explicit_physical_gpu(self):
        cfg = read_json(Path(__file__).parent / 'config.json')
        configured = configure_smoke(cfg, 2)
        self.assertEqual(configured['gpu_ids'], [2])
        self.assertEqual(configured['seeds'], [42])
        self.assertEqual([r['name'] for r in configured['recipes']],
                         ['concat_ce', 'aligned_ce'])
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            configure_smoke(read_json(Path(__file__).parent / 'config.json'), -1)

    def test_explicit_publication_gpu_mapping(self):
        self.assertEqual(parse_gpu_ids('1,3,5,6,7'), [1, 3, 5, 6, 7])
        for invalid in ('', '-1,2', '1,1', 'one,2'):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    parse_gpu_ids(invalid)

    def test_job_identity_is_independent_of_physical_gpu_ids(self):
        cfg = read_json(Path(__file__).parent / 'config.json')
        hardware = {'cpu_ids': list(range(20))}
        first = jobs_for(cfg, hardware, slots=5)
        cfg['gpu_ids'] = [1, 3, 5, 6, 7]
        second = jobs_for(cfg, hardware, slots=5)
        self.assertEqual(first, second)
        self.assertNotIn('gpu_ids', first[0]['config'])

    def test_primary_and_scale_ablation_encoder_contract(self):
        root = Path(__file__).parent
        primary = read_json(root / 'config.json')
        scale = read_json(root / 'config_base_small.json')
        self.assertEqual(primary['profile'], 'paper_faithful_large_large_primary')
        self.assertEqual(primary['vit'], 'vit_large_patch14_dinov2.lvd142m')
        self.assertEqual(primary['cnn'], 'convnext_large.fb_in22k_ft_in1k')
        self.assertEqual(scale['profile'], 'base_small_scale_ablation')
        self.assertEqual(scale['vit'], 'vit_base_patch14_dinov2.lvd142m')
        self.assertEqual(scale['cnn'], 'convnext_small.fb_in22k_ft_in1k')
        for model_name in (primary['vit'], primary['cnn'], scale['vit'], scale['cnn']):
            self.assertTrue(timm.is_model(model_name), model_name)
        for key in ('recipes', 'seeds', 'image_size', 'effective_batch_size'):
            self.assertEqual(primary[key], scale[key])

    def test_conflicting_exact_duplicates_are_quarantined(self):
        frame = pd.DataFrame([
            {'path': 'a.png', 'source': 'APTOS', 'role': 'development', 'label': 0,
             'pixel_sha256': 'conflict'},
            {'path': 'b.png', 'source': 'APTOS', 'role': 'development', 'label': 1,
             'pixel_sha256': 'conflict'},
            {'path': 'c.png', 'source': 'EyePACS', 'role': 'development', 'label': 2,
             'pixel_sha256': 'consistent'},
            {'path': 'd.png', 'source': 'MESSIDOR2', 'role': 'external', 'label': 2,
             'pixel_sha256': 'consistent'},
        ])
        keep, exclusions, conflicts = resolve_exact_duplicates(frame)
        self.assertEqual(keep, [3])
        self.assertEqual({item['path'] for item in exclusions}, {'a.png', 'b.png', 'c.png'})
        self.assertEqual([item['reason'] for item in exclusions].count(
            'conflicting_exact_duplicate_label'), 2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(
            {(member['path'], member['label']) for member in conflicts[0]['members']},
            {('a.png', 0), ('b.png', 1)},
        )

    def test_attention_query_key_gradients(self):
        block = Exchange(16, 4, True)
        q, k = torch.randn(2, 9, 16), torch.randn(2, 4, 16)
        out = block(q, k, torch.rand(9, 2), torch.rand(4, 2))
        out.square().mean().backward()
        grad = block.attention.in_proj_weight.grad
        self.assertGreater(float(grad[:16].norm()), 0)
        self.assertGreater(float(grad[16:32].norm()), 0)
        self.assertGreater(float(block.raw_scale.grad.abs()), 0)

    def test_all_recipes_forward_backward(self):
        cfg = read_json(Path(__file__).parent / 'config.json')
        cfg.update(tiny=True, fusion_width=32, fusion_heads=4)
        for recipe in cfg['recipes']:
            with self.subTest(recipe=recipe['name']):
                model = HybridDR(cfg, recipe)
                logits = model(torch.rand(2, 3, 56, 56))
                loss = objective(logits, torch.tensor([1, 4]), recipe['loss'], 2, 30, torch.ones(5))
                loss.backward()
                p = probabilities(logits, recipe['loss'] == 'ordinal')
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue((p >= 0).all())
                self.assertTrue(torch.allclose(p.sum(1), torch.ones(2), atol=1e-6))

    def test_grouped_splits(self):
        frame = pd.DataFrame([{'source': 'EyePACS', 'role': 'development', 'label': i % 5,
                               'patient': f'p{i}', 'eye': eye} for i in range(100) for eye in (0, 1)])
        split = assign_splits(frame)
        self.assertEqual(split.groupby('patient').split.nunique().max(), 1)
        self.assertEqual(set(split.split), {'train', 'val', 'calibration', 'internal'})

    def test_metrics_perfect(self):
        y = np.arange(5).repeat(4)
        score = measure(y, np.eye(5)[y])
        self.assertEqual(score['qwk'], 1)
        self.assertEqual(score['two_grade_error'], 0)
        self.assertEqual(score['ece_15_equal_width'], 0)

    def test_calibration_and_bootstrap(self):
        y = np.arange(5).repeat(4)
        logits = np.eye(5)[y] * 2
        t = fit_temperature(logits, y, False)
        self.assertGreater(t, 0)
        frame = pd.DataFrame({'path': list(map(str, range(20))), 'patient': list(map(str, range(20))), 'label': y, 'pred': y})
        self.assertEqual(paired_bootstrap(frame, frame, 20)['delta_qwk_ci95'], [0., 0.])

    def test_five_gpu_cpu_budget(self):
        cfg = read_json(Path(__file__).parent / 'config.json')
        jobs = jobs_for(cfg, {'cpu_ids': list(range(179))})
        self.assertEqual(len(jobs), 65)
        first = [set(j['cpu_ids']) for j in jobs[:5]]
        self.assertEqual(len(set.union(*first)), 179)
        for i in range(5):
            for j in range(i):
                self.assertFalse(first[i] & first[j])

    def test_matched_focal_exposure(self):
        logits = torch.randn(5, 5)
        labels = torch.arange(5)
        forward = torch.stack([objective(logits, labels, 'pfl', e, 30, torch.ones(5)) for e in range(30)]).mean()
        reverse = torch.stack([objective(logits, labels, 'reverse_pfl', e, 30, torch.ones(5)) for e in range(30)]).mean()
        matched = objective(logits, labels, 'matched_focal', 0, 30, torch.ones(5))
        self.assertTrue(torch.allclose(forward, matched, atol=1e-6))
        self.assertTrue(torch.allclose(reverse, matched, atol=1e-6))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
