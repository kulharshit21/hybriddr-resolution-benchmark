import json
import unittest
from pathlib import Path


class FinalCampaignContractTests(unittest.TestCase):
    def load(self, name):
        return json.loads((Path(__file__).parent / name).read_text(encoding='utf-8'))

    def test_frozen_run_counts_and_profiles(self):
        pilot = self.load('config_full_data_pilot.json')
        primary = self.load('config_final.json')
        scale = self.load('config_scale_ablation_final.json')
        self.assertEqual(len(pilot['recipes']) * len(pilot['seeds']), 5)
        self.assertEqual(pilot['epochs'], 1)
        self.assertEqual(len(primary['recipes']) * len(primary['seeds']), 65)
        self.assertEqual(len(scale['recipes']) * len(scale['seeds']), 10)
        self.assertEqual(primary['epochs'], scale['epochs'])
        self.assertEqual(primary['seeds'], scale['seeds'])
        self.assertEqual(primary['image_size'], scale['image_size'])
        self.assertEqual(primary['effective_batch_size'], scale['effective_batch_size'])

    def test_primary_roster_contains_prespecified_ablation_names(self):
        primary = self.load('config_final.json')
        names = {item['name'] for item in primary['recipes']}
        required = {'vit_ce', 'cnn_ce', 'concat_ce', 'gated_ce', 'spatial_ce',
                    'aligned_ce', 'aligned_oneway_ce', 'concat_pfl', 'aligned_pfl',
                    'aligned_focal', 'aligned_matched', 'aligned_reverse',
                    'aligned_ordinal'}
        self.assertEqual(names, required)


if __name__ == '__main__':
    unittest.main(verbosity=2)
