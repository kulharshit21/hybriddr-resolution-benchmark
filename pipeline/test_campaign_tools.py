import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from campaign_tools import expected_runs, inspect


class CampaignToolsTests(unittest.TestCase):
    def test_expected_run_registry(self):
        cfg = {'recipes': [{'name': 'a'}, {'name': 'b'}], 'seeds': [1, 2]}
        self.assertEqual(expected_runs(cfg), ['a_seed1', 'a_seed2', 'b_seed1', 'b_seed2'])

    def test_inspection_requires_exact_epoch_count_and_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'profile': 'pilot', 'recipes': [{'name': 'a'}],
                      'seeds': [1], 'epochs': 1}
            config_path = root / 'config-source.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            run = root / 'runs/a_seed1'
            run.mkdir(parents=True)
            checkpoint = run / 'best.pt'
            checkpoint.write_bytes(b'model')
            sha = hashlib.sha256(b'model').hexdigest()
            (run / 'history.json').write_text(json.dumps([{
                'microbatch_size': 2, 'training_images_per_second': 3.0
            }]), encoding='utf-8')
            (run / 'complete.json').write_text(json.dumps({
                'checkpoint_sha256': sha, 'best_validation_qwk': 0.1,
                'wall_seconds_this_invocation': 4, 'peak_cuda_bytes': 5
            }), encoding='utf-8')
            result = inspect(root, config_path)
            self.assertTrue(result['all_training_complete'])
            self.assertEqual(result['runs'][0]['last_epoch_images_per_second'], 3.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
