import tempfile
import unittest
from pathlib import Path

from lightglue_calibration import deterministic_negatives, validate_calibration_resume


class LightGlueCalibrationTests(unittest.TestCase):
    def test_deterministic_negatives_are_unique_and_not_self_pairs(self):
        keys = [f"image-{i}" for i in range(20)]
        first = deterministic_negatives(keys, 100, 20261104)
        second = deterministic_negatives(keys, 100, 20261104)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first)), 100)
        self.assertTrue(all(left != right for left, right in first))

    def test_negative_request_rejects_impossible_count(self):
        with self.assertRaisesRegex(ValueError, "exceeds possible"):
            deterministic_negatives(["a", "b", "c"], 4, 1)

    def test_resume_rejects_duplicate_and_unknown_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.jsonl"
            path.write_text('{"calibration_id":"a"}\n{"calibration_id":"a"}\n')
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                validate_calibration_resume(path, {"a", "b"})
            path.write_text('{"calibration_id":"z"}\n')
            with self.assertRaisesRegex(RuntimeError, "Unknown"):
                validate_calibration_resume(path, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
