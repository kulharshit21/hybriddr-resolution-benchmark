import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lightglue_triage import geometry_evidence, load_pairs, validate_resume


class LightGlueTriageTests(unittest.TestCase):
    def test_load_pairs_is_sorted_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            rows = [
                {"pair_id": "b", "left": "l2", "right": "r2"},
                {"pair_id": "a", "left": "l1", "right": "r1"},
            ]
            path.write_text("".join(json.dumps(x) + "\n" for x in rows))
            self.assertEqual([x["pair_id"] for x in load_pairs(path)], ["a", "b"])
            path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "Duplicate pair_id"):
                load_pairs(path)

    def test_validate_resume_rejects_unknown_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "scores.jsonl"
            ledger.write_text(
                json.dumps({"pair_id": "a"}) + "\n" +
                json.dumps({"pair_id": "a"}) + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "Duplicate resume"):
                validate_resume(ledger, {"a", "b"})
            ledger.write_text(json.dumps({"pair_id": "z"}) + "\n")
            with self.assertRaisesRegex(RuntimeError, "Unknown pair_id"):
                validate_resume(ledger, {"a", "b"})

    def test_geometry_handles_too_few_matches(self):
        points = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        result = geometry_evidence(points, points.copy(), np.asarray([0.9, 0.8, 0.7]))
        self.assertEqual(result["matches"], 3)
        self.assertEqual(result["homography_inliers"], 0)
        self.assertEqual(result["homography_inlier_ratio"], 0.0)

    def test_geometry_recovers_exact_translation(self):
        rng = np.random.default_rng(7)
        left = rng.uniform(0, 500, size=(40, 2)).astype(np.float32)
        right = left + np.asarray([7.0, -4.0], dtype=np.float32)
        scores = np.linspace(0.5, 0.99, 40)
        result = geometry_evidence(left, right, scores)
        self.assertEqual(result["matches"], 40)
        self.assertGreaterEqual(result["homography_inliers"], 39)
        self.assertGreater(result["homography_inlier_ratio"], 0.97)
        self.assertLess(result["median_reprojection_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
