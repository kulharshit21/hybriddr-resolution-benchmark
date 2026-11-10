import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from base_models import Exchange, HybridDR as OriginalModel, objective as original_loss
from models import HybridDR, ScaledExchange, class_weights, training_objective
from train import build_scheduler
from smoke_gpu import check_shared_gpu


class V2Tests(unittest.TestCase):
    def test_shared_smoke_rejects_v1_gpu_and_low_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory)
            (prior / 'selected-gpus.txt').write_text('1,3,5,6,2')
            with self.assertRaisesRegex(ValueError, 'V1 pool'):
                check_shared_gpu(prior, 2)
            with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'GPU-test'}):
                with patch('smoke_gpu.subprocess.check_output', return_value='GPU-test, 15999'):
                    with self.assertRaisesRegex(ValueError, '16000'):
                        check_shared_gpu(prior, 0)
                with patch('smoke_gpu.subprocess.check_output', return_value='GPU-test, 20000'):
                    self.assertEqual(check_shared_gpu(prior, 0), 'GPU-test')
    def test_zero_scale_preserves_input_and_small_scale_has_gradients(self):
        torch.manual_seed(12)
        block = ScaledExchange(Exchange(16, 4, True), 0.)
        q, k = torch.randn(2, 9, 16), torch.randn(2, 4, 16)
        qp, kp = torch.rand(9, 2), torch.rand(4, 2)
        self.assertTrue(torch.equal(block(q, k, qp, kp), q))
        block.gain.data.fill_(.001)
        block(q, k, qp, kp).square().mean().backward()
        self.assertGreater(float(block.gain.grad.abs()), 0)
        self.assertGreater(float(block.block.attention.in_proj_weight.grad.norm()), 0)

    def test_unmodified_recipe_preserves_legacy_predictions(self):
        cfg = dict(tiny=True, fusion_width=32, fusion_heads=4)
        recipe = dict(architecture='spatial', spatial_bias=True, loss='ce')
        torch.manual_seed(4)
        old = OriginalModel(cfg, recipe).eval()
        torch.manual_seed(4)
        new = HybridDR(cfg, recipe).eval()
        x = torch.rand(2, 3, 56, 56)
        self.assertTrue(torch.equal(old(x), new(x)))

    def test_blend_starts_as_ce_and_retains_confidence_penalty(self):
        counts = torch.tensor([100., 20., 10., 5., 2.])
        logits = torch.randn(5, 5)
        labels = torch.arange(5)
        for kind in ['blend_uniform', 'blend_tempered']:
            recipe = dict(loss=kind)
            actual = training_objective(logits, labels, recipe, 0, 30, counts, .1)
            expected = original_loss(logits, labels, 'ce', 0, 30, torch.ones(5), .1)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
            confident = torch.tensor([[20., 0., 0., 0., 0.]], requires_grad=True)
            loss = training_objective(confident, torch.tensor([0]), recipe, 29, 30, counts, .1)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(confident.grad.norm()), .001)
        weights = class_weights(counts, tempered=True)
        self.assertAlmostEqual(float((weights * counts / counts.sum()).sum()), 1., places=6)

    def test_original_pfl_and_matched_loss_preserved(self):
        counts = torch.tensor([100., 20., 10., 5., 2.])
        logits, labels = torch.randn(5, 5), torch.arange(5)
        for kind in ['ce', 'focal', 'pfl', 'matched_focal', 'reverse_pfl', 'ordinal']:
            x = logits[:, :4] if kind == 'ordinal' else logits
            alpha = class_weights(counts)
            actual = training_objective(x, labels, dict(loss=kind), 12, 30, counts, .1)
            expected = original_loss(x, labels, kind, 12, 30, alpha, .1)
            self.assertTrue(torch.equal(actual, expected))

    def test_scaled_fusion_trains_with_both_blend_losses(self):
        config = dict(tiny=True, fusion_width=32, fusion_heads=4)
        for kind in ['blend_uniform', 'blend_tempered']:
            recipe = dict(architecture='spatial', spatial_bias=True, loss=kind, residual_gain_init=.001)
            model = HybridDR(config, recipe)
            logits = model(torch.rand(2, 3, 56, 56))
            loss = training_objective(logits, torch.tensor([0, 4]), recipe, 20, 30,
                                      torch.tensor([100., 20., 10., 5., 2.]), .1)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.vexchange.gain.grad.abs()), 0)
            self.assertGreater(float(model.cexchange.gain.grad.abs()), 0)

    def test_warmup_scheduler_resumes_exactly(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=.0001)
        scheduler = build_scheduler(optimizer, 10, 3)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .0001 / 3)
        for _ in range(4):
            optimizer.step()
            scheduler.step()
        saved_optimizer, saved_scheduler = optimizer.state_dict(), scheduler.state_dict()
        second = torch.optim.AdamW([nn.Parameter(torch.ones(1))], lr=.0001)
        resumed = build_scheduler(second, 10, 3)
        second.load_state_dict(saved_optimizer)
        resumed.load_state_dict(saved_scheduler)
        for _ in range(5):
            optimizer.step(); scheduler.step()
            second.step(); resumed.step()
            self.assertEqual(optimizer.param_groups[0]['lr'], second.param_groups[0]['lr'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
