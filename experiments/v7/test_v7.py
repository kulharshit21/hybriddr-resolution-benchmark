import tempfile,unittest,json
from pathlib import Path
import numpy as np
from PIL import Image
from audit import inspect
from launch import ALLOWED
class Tests(unittest.TestCase):
    def test_gpu_admission_and_five_slots(self):
        from launch import available_cards
        raw='\n'.join(f'{i}, GPU-{i}, 40000' for i in range(8))
        self.assertEqual([r[0] for r in available_cards(raw,{},6)],[0,1,2,3,5])
        self.assertEqual([r[0] for r in available_cards(raw,{0:None,1:None,2:None,3:None},2)],[5])
        self.assertEqual(available_cards('5,GPU-5,19999',{},6),[])
        parts=[set(range(256)[slot::5]) for slot in range(5)]
        self.assertEqual(len(set.union(*parts)),256)
        self.assertEqual(sum(map(len,parts)),256)
    def test_completion_rejects_corruption(self):
        import hashlib
        from launch import verify_completion,digest
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);job={'seed':11};mh='manifest'
            (r/'best.pt').write_bytes(b'checkpoint');(r/'history.json').write_text(json.dumps([{}]*30))
            record=dict(run_hash=hashlib.sha256(json.dumps({'job':job,'manifest':mh},sort_keys=True).encode()).hexdigest(),smoke_only=False,checkpoint_sha256=digest(r/'best.pt'))
            (r/'complete.json').write_text(json.dumps(record));verify_completion(r,job,mh)
            (r/'best.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_completion(r,job,mh)
    def test_cache_fidelity_and_aspect_distortion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);x=Image.fromarray(np.random.default_rng(1).integers(0,256,(128,256,3),dtype=np.uint8))
            x.save(p/'original.png');x.resize((448,448),Image.Resampling.LANCZOS).save(p/'cache.png')
            row=dict(path='original.png',cache='cache.png',source='EyePACS',label='1')
            r=inspect((row,d,d));self.assertEqual(r['aspect_ratio'],2);self.assertEqual(r['cache_reference_mae'],0)
            Image.new('RGB',(448,448)).save(p/'cache.png')
            self.assertGreater(inspect((row,d,d))['cache_reference_mae'],1)
    def test_scope_and_matched_training(self):
        self.assertEqual(ALLOWED,{0,1,2,3,5})
        c=json.loads((Path(__file__).parent/'config.json').read_text())
        self.assertEqual(c['effective_batch_size'],64);self.assertTrue(c['pretrained'])
        self.assertEqual(c['seeds'],[11,23,42,67,101]);self.assertEqual(c['epochs'],30)
    def test_paired_prediction_comparison(self):
        from launch import recipes
        self.assertEqual([(r['architecture'],r['loss']) for r in recipes()],[('cnn','ce'),('gated','ce')])
        self.assertEqual(len({f"{r['name']}_seed{s}" for r in recipes() for s in [11,23,42,67,101]}),10)
if __name__=='__main__':unittest.main()
