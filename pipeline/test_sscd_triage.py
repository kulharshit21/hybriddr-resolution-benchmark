import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .sscd_triage import (
        CalibrationDataset, calibrate, load_pairs, preprocess, transformed,
    )
except ImportError:
    from sscd_triage import (
        CalibrationDataset, calibrate, load_pairs, preprocess, transformed,
    )


class SSCDTriageTests(unittest.TestCase):
    def test_preprocessing_and_transformations_are_deterministic(self):
        values = np.zeros((80, 120, 3), dtype=np.uint8)
        values[10:70, 20:100, 1] = 180
        image = Image.fromarray(values)
        first = preprocess(image)
        second = preprocess(image)
        self.assertEqual(first.shape, (3, 320, 320))
        np.testing.assert_array_equal(first, second)
        for variant in range(5):
            result = transformed(image, variant)
            self.assertEqual(result.mode, "RGB")

    def test_calibration_locks_only_with_positive_negative_separation(self):
        embeddings = np.asarray([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        ], dtype=np.float32)
        positive_a = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        positive_b = np.asarray([[.999, .001], [.001, .999]], dtype=np.float32)
        record = calibrate(embeddings, positive_a, positive_b,
                           seed=4, negative_pairs=4)
        self.assertTrue(record["locked"])
        self.assertGreater(record["separation_margin"], 0)

        failed = calibrate(
            np.asarray([[1.0, 0.0], [.99, .01]], dtype=np.float32),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
            seed=2, negative_pairs=1,
        )
        self.assertFalse(failed["locked"])
        self.assertIsNone(failed["provisional_confirmation_threshold"])

    def test_pair_ledger_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.jsonl"
            row = {"pair_id": "a" * 64, "left": "a.jpg", "right": "b.jpg"}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(RuntimeError):
                load_pairs(path)

    def test_calibration_dataset_returns_five_deterministic_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = np.zeros((80, 120, 3), dtype=np.uint8)
            values[10:70, 20:100, 1] = 180
            Image.fromarray(values).save(root / "retina.png")
            dataset = CalibrationDataset(root, ["retina.png"])
            index, bases, variants = dataset[0]
            self.assertEqual(index, 0)
            self.assertEqual(tuple(bases.shape), (5, 3, 320, 320))
            self.assertEqual(tuple(variants.shape), (5, 3, 320, 320))
            np.testing.assert_array_equal(bases[0].numpy(), bases[4].numpy())
            self.assertFalse(np.array_equal(bases[0].numpy(), variants[0].numpy()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
