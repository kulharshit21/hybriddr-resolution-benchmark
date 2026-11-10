import tempfile
import unittest
import json
from pathlib import Path
from review import paired, predictions, review
from watch_review import campaigns_ready

class Tests(unittest.TestCase):
    def test_gate_requires_both_committed_result_hashes(self):
        from review import digest
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertFalse(campaigns_ready(root))
            for version, name in [('V6', 'v6-resolution-final-01'), ('V7', 'v7-controls-448-01')]:
                r = root/'runs'/name
                r.mkdir(parents=True)
                (r/'RESULTS.json').write_text('{}')
                (r/(version+'_COMPLETE.json')).write_text(json.dumps({'runs':10,'epochs':300,'results_sha256':digest(r/'RESULTS.json')}))
            self.assertTrue(campaigns_ready(root))
            (r/'RESULTS.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): campaigns_ready(root)

    def test_complete_registry_and_label_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for arch in ('aligned', 'cnn', 'gated'):
                campaign = 'v6-resolution-final-01' if arch == 'aligned' else 'v7-controls-448-01'
                for seed in (11, 23, 42, 67, 101):
                    run = root/'runs'/campaign/'runs'/f'{arch}_ce_r448_seed{seed}'
                    run.mkdir(parents=True)
                    (run/'history.json').write_text(json.dumps([{}]*30))
                    (run/'complete.json').write_text(json.dumps({'checkpoint_sha256': 'abc'}))
                    (run/'evaluation.json').write_text(json.dumps({'checkpoint_sha256': 'abc', 'scores': {'EyePACS_internal': {'raw': {'qwk': .8 if arch == 'aligned' else .7}}}}))
                    (run/'EyePACS_internal_predictions.csv').write_text('path,source,patient,label,p0,p1,p2,p3,p4\na,EyePACS,x,1,0,1,0,0,0\n')
            result = review(root)
            self.assertEqual(result['status'], 'READY_FOR_PROVENANCE_REVIEW')
            self.assertEqual(len(result['contrasts']), 2)
            self.assertAlmostEqual(result['contrasts'][0]['mean_delta'], .1)
            path = root/'runs/v7-controls-448-01/runs/gated_ce_r448_seed101/EyePACS_internal_predictions.csv'
            path.write_text(path.read_text().replace(',x,1,', ',x,2,'))
            with self.assertRaisesRegex(ValueError, 'Unpaired'): review(root)

    def test_missing_runs_wait_without_training(self):
        with tempfile.TemporaryDirectory() as d:
            x = review(Path(d))
            self.assertEqual(x['status'], 'WAITING')
            self.assertEqual(len(x['missing_runs']), 15)

    def test_paired_difference_interval(self):
        x = paired([.02]*5)
        self.assertEqual(x['positive_seeds'], 5)
        self.assertEqual(x['seed_t95_assumption_sensitive'], [.02, .02])
        with self.assertRaises(ValueError):
            paired([.02]*4)

    def test_invalid_probability_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'p.csv'
            header = 'path,source,patient,label,p0,p1,p2,p3,p4\n'
            row = 'a,EyePACS,x,1,0,1,0,0,0\n'
            p.write_text(header+row+row)
            with self.assertRaises(ValueError): predictions(p)
            p.write_text(header+'a,EyePACS,x,1,0,nan,0,0,0\n')
            with self.assertRaises(ValueError): predictions(p)

if __name__ == '__main__': unittest.main(verbosity=2)
