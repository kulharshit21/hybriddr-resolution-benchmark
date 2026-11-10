import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from common import read_json,save_json,digest
from base_models import objective as legacy
from models import training_objective,class_weights
from train import train,resolve_device
from controller import eligible,jobs
from diagnostics import verify_development_labels

HERE=Path(__file__).resolve().parent


class TargetedTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_gpu_allowlist_and_fixed_registry(self):
        rows=[dict(index=i,free=40000,total=40960) for i in range(8)]
        self.assertEqual({r['index'] for r in eligible(rows,{2})},{0,1,5,6})
        cfg=read_json(HERE/'config.json')
        self.assertEqual(len(jobs(cfg)),10)
        self.assertEqual(cfg['epochs']*len(jobs(cfg)),300)

    def test_unindexed_cuda_resolves_before_memory_cap(self):
        self.assertEqual(resolve_device('cuda'),torch.device('cuda:0'))
        self.assertEqual(resolve_device('cuda:0'),torch.device('cuda:0'))
        self.assertEqual(resolve_device('cpu'),torch.device('cpu'))

    def test_balanced_softmax_formula_and_uniform_prior(self):
        x=torch.randn(6,5,requires_grad=True);y=torch.tensor([0,1,2,3,4,1])
        counts=torch.tensor([100.,20.,10.,5.,2.])
        actual=training_objective(x,y,dict(loss='balanced_softmax'),0,30,counts,.1)
        expected=F.cross_entropy(x+(counts/counts.sum()).log(),y,label_smoothing=.1)
        self.assertTrue(torch.equal(actual,expected))
        uniform=training_objective(x,y,dict(loss='balanced_softmax'),0,30,torch.ones(5),.1)
        self.assertTrue(torch.allclose(uniform,F.cross_entropy(x,y,label_smoothing=.1)))
        actual.backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_original_loss_preserved_and_blends_keep_smoothing_gradient(self):
        counts=torch.tensor([16295.,1699.,3642.,632.,597.]);x=torch.randn(5,5);y=torch.arange(5)
        self.assertTrue(torch.equal(training_objective(x,y,dict(loss='pfl'),15,30,counts,.1),legacy(x,y,'pfl',15,30,class_weights(counts),.1)))
        for kind in ['blend_uniform','blend_tempered']:
            recipe=dict(loss=kind,blend_max=.5,blend_ramp_epochs=10,focal_gamma=1.)
            self.assertTrue(torch.allclose(training_objective(x,y,recipe,0,30,counts,.1),F.cross_entropy(x,y,label_smoothing=.1),atol=1e-6))
            confident=torch.tensor([[20.,0.,0.,0.,0.]],requires_grad=True)
            training_objective(confident,torch.tensor([0]),recipe,29,30,counts,.1).backward()
            self.assertGreater(float(confident.grad.norm()),.001)
        weights=class_weights(counts,tempered=True)
        self.assertAlmostEqual(float((weights*counts/counts.sum()).sum()),1.,places=6)

    def test_label_join_rejects_changed_grade(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'EyePACS').mkdir();(root/'APTOS2019').mkdir()
            pd.DataFrame(dict(image=['a_left'],level=[1])).to_csv(root/'EyePACS/trainLabels.csv',index=False)
            pd.DataFrame(dict(id_code=['b'],diagnosis=[2])).to_csv(root/'APTOS2019/train.csv',index=False)
            frame=pd.DataFrame(dict(path=['EyePACS/train/a_left.jpeg','APTOS2019/train_images/b.png'],source=['EyePACS','APTOS'],label=[1,2]))
            self.assertEqual(verify_development_labels(frame,root),2)
            frame.loc[0,'label']=0
            with self.assertRaisesRegex(ValueError,'mismatch'):verify_development_labels(frame,root)


    def test_all_five_arms_train_and_reuse_completed_checkpoint_cpu(self):
        config=read_json(HERE/'config.json')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);prepared=root/'prepared';prepared.mkdir()
            rows=[]
            for split in ['train','val']:
                for source in ['APTOS','EyePACS']:
                    for grade in range(5):
                        name=f'{split}-{source}-{grade}.png'
                        arr=np.random.default_rng(grade).integers(0,256,(56,56,3),dtype=np.uint8)
                        Image.fromarray(arr).save(prepared/name)
                        rows.append(dict(path=name,cache=name,source=source,split=split,label=grade,patient=name))
            pd.DataFrame(rows).to_csv(prepared/'manifest.csv',index=False)
            for recipe in config['recipes']:
                cfg=dict(config,tiny=True,pretrained=False,epochs=2,image_size=56,fusion_width=16,fusion_heads=2,
                         gradient_checkpointing=False,auto_batch=False,batch_size=5,effective_batch_size=5,
                         workers_per_gpu=0,threads_per_gpu=1)
                import psutil
                try:cpu_ids=psutil.Process().cpu_affinity()[:1]
                except AttributeError:cpu_ids=[0]
                job=dict(config=cfg,recipe=recipe,seed=11,cpu_ids=cpu_ids)
                jobfile=root/(recipe['name']+'.json');save_json(jobfile,job)
                out=root/'runs'/f"{recipe['name']}_seed11"
                train(jobfile,prepared,out,'cpu')
                history=read_json(out/'history.json')
                self.assertEqual(len(history),2)
                self.assertEqual(len(history[-1]['validation_per_grade']['EyePACS']['recall']),5)
                self.assertIn('mean_source_grade1_recall',history[-1])
                previous=digest(out/'best.pt')
                train(jobfile,prepared,out,'cpu')
                self.assertEqual(digest(out/'best.pt'),previous)
                self.assertTrue((out/'latest_validation_private.npz').is_file())
                self.assertTrue((out/'best_validation_private.npz').is_file())
                shutil.copytree(out,root/'runs'/f"{recipe['name']}_seed23")
            save_json(root/'STUDY_CONTRACT.json',dict(config=config))
            from summarize import main as summarize
            with patch('sys.argv',['summarize','--out',str(root)]):summarize()
            self.assertEqual(read_json(root/'VALIDATION_REPORT_COMPLETE.json')['rows'],20)
            self.assertTrue((root/'validation_tradeoffs.pdf').is_file())


if __name__=='__main__':unittest.main(verbosity=2)
