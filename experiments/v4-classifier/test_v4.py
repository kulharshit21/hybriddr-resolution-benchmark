import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from analysis import scores,write_report,figures
from worker import reconstruct_check,fit_heads

class Tests(unittest.TestCase):
    def test_plot_artifacts_are_created_for_each_source(self):
        y=np.tile(np.arange(5),4);p=np.eye(5)[y]*.9+.02
        with tempfile.TemporaryDirectory() as d:
            figures(y,p,np.array(['A']*10+['B']*10),Path(d),'synthetic')
            self.assertEqual(len(list(Path(d).glob('*.png'))),2)
            self.assertEqual(len(list(Path(d).glob('*.pdf'))),2)

    def test_metrics_match_perfect_classifier_and_pairing(self):
        y=np.tile(np.arange(5),4);p=np.eye(5)[y]*.9+.02
        source=np.array(['APTOS']*10+['EyePACS']*10)
        result=scores(y,p,source)
        self.assertEqual(result['macro']['qwk'],1)
        self.assertEqual(result['macro']['grade1_f1'],1)
        self.assertEqual(result['macro']['grade1_ap'],1)
        with tempfile.TemporaryDirectory() as d:
            write_report({'unchanged_ce_seed11':result,'balanced_head_seed11':result},Path(d))
            self.assertIn('+0.0000',(Path(d)/'REPORT.md').read_text())

    def test_invalid_predictions_and_reconstruction_fail_closed(self):
        with self.assertRaises(ValueError):scores(np.arange(5),np.ones((5,5)),np.array(['A']*5))
        with self.assertRaises(ValueError):reconstruct_check(np.eye(5),np.zeros((5,5)))
        reconstruct_check(np.eye(5),np.eye(5))

    def test_frozen_extractor_feature_hook_and_head_gradients(self):
        sys.path.insert(0,str(Path(__file__).parent/'baseline'))
        from models import HybridDR
        cfg=dict(tiny=True,fusion_width=16,fusion_heads=2)
        recipe=dict(name='aligned_ce',architecture='spatial',loss='ce',spatial_bias=True)
        model=HybridDR(cfg,recipe,initialize=False).eval()
        for p in model.parameters():p.requires_grad_(False)
        capture=[]
        hook=model.head.register_forward_pre_hook(lambda module,inputs:capture.append(inputs[0].detach()))
        x=torch.rand(5,3,32,32)
        with torch.no_grad():logits=model(x)
        hook.remove()
        self.assertEqual(len(capture),1)
        torch.testing.assert_close(model.head(capture[0]),logits)
        frozen={n:p.clone() for n,p in model.named_parameters()}
        for p in model.head.parameters():p.requires_grad_(True)
        opt=torch.optim.SGD(model.head.parameters(),lr=.1)
        torch.nn.functional.cross_entropy(model.head(capture[0]),torch.arange(5)).backward();opt.step()
        for n,p in model.named_parameters():
            if not n.startswith('head.'):
                torch.testing.assert_close(p,frozen[n]);self.assertIsNone(p.grad)
        self.assertFalse(torch.equal(model.head.weight,frozen['head.weight']))

    def test_two_adaptations_and_baseline_are_reported(self):
        torch.set_num_threads(2);torch.manual_seed(11)
        x=torch.rand(50,8);y=torch.arange(50)%5
        with tempfile.TemporaryDirectory() as d,patch('worker.figures'):
            out=Path(d)
            result=fit_heads(x,y,x,y.numpy(),np.array(['A']*25+['B']*25),
                torch.rand(5,8),torch.rand(5),11,out,epochs=2)
            self.assertEqual(set(result),{'unchanged_ce_seed11','natural_head_seed11','balanced_head_seed11'})
            for mode in ('natural','balanced'):
                self.assertEqual(len(json.loads((out/(mode+'_history.json')).read_text())),2)
                self.assertTrue((out/(mode+'_head.pt')).exists())
                self.assertGreaterEqual(result[mode+'_head_seed11']['macro']['qwk'],result['unchanged_ce_seed11']['macro']['qwk'])

if __name__=='__main__':unittest.main(verbosity=2)
