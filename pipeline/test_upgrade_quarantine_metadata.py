import json
import tempfile
import unittest
from pathlib import Path

from build_quarantine_child import digest, quarantine_audit_id
from upgrade_quarantine_metadata import upgrade


class UpgradeTests(unittest.TestCase):
    def test_upgrade_is_identity_bound_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            (prepared / 'manifest.csv').write_text('path,label\na.png,0\n', encoding='utf-8')
            manifest_hash = digest(prepared / 'manifest.csv')
            original = {
                'child_manifest_sha256': manifest_hash,
                'parent_manifest_sha256': 'a' * 64,
                'candidate_ledger_sha256': 'b' * 64,
            }
            (prepared / 'prepared.json').write_text(json.dumps(original), encoding='utf-8')
            first = upgrade(prepared)
            second = upgrade(prepared)
            self.assertEqual(first, second)
            self.assertEqual(first['manifest_sha256'], manifest_hash)
            self.assertEqual(first['audit_id'], quarantine_audit_id('a' * 64, 'b' * 64))

    def test_upgrade_rejects_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory)
            (prepared / 'manifest.csv').write_text('path,label\na.png,0\n', encoding='utf-8')
            metadata = {
                'child_manifest_sha256': '0' * 64,
                'parent_manifest_sha256': 'a' * 64,
                'candidate_ledger_sha256': 'b' * 64,
            }
            (prepared / 'prepared.json').write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                upgrade(prepared)


if __name__ == '__main__':
    unittest.main(verbosity=2)
