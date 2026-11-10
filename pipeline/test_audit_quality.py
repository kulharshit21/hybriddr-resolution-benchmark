import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from audit_quality import run_quality
from common import digest
import test_near_duplicates


class QualityAuditTests(unittest.TestCase):
    def test_unusable_retina_is_excluded_with_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            prepared = root / 'prepared'
            output = root / 'quality'
            data.mkdir(); prepared.mkdir()
            valid = data / 'valid.png'
            invalid = data / 'invalid.png'
            test_near_duplicates.NormalizationTests.retina().save(valid)
            Image.new('RGB', (320, 320), 'black').save(invalid)
            pd.DataFrame([
                {'path': 'valid.png', 'sha256': digest(valid)},
                {'path': 'invalid.png', 'sha256': digest(invalid)},
            ]).to_csv(prepared / 'manifest.csv', index=False)

            summary = run_quality(
                data, prepared, output,
                Path(__file__).parent / 'near_duplicate_config.json', workers=1)

            child = pd.read_csv(output / 'manifest.csv')
            self.assertEqual(child.path.tolist(), ['valid.png'])
            self.assertEqual(summary['quality_exclusions'], 1)
            evidence = (output / 'quality_evidence.jsonl').read_text(encoding='utf-8')
            self.assertIn('retinal_quality_failure', evidence)
            self.assertTrue((output / 'QUALITY_COMPLETE').is_file())
            self.assertFalse((output / 'QUALITY_INCOMPLETE').exists())


if __name__ == '__main__':
    unittest.main()
