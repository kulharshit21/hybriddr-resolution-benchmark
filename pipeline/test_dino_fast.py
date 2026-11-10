import tempfile
import unittest
from pathlib import Path

import numpy as np

from dino_fast import _load_resume_jsonl, _write_jsonl, exact_cuda_neighbors
from near_duplicates import exact_cosine_neighbors


class FastDinoArtifactTests(unittest.TestCase):
    def test_jsonl_resume_requires_identity_and_unique_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            records = [
                {"pair_id": "a", "audit_id": "run", "classification": "rejected"},
                {"pair_id": "b", "audit_id": "run", "classification": "confirmed"},
            ]
            _write_jsonl(path, records)
            loaded = _load_resume_jsonl(path, "run")
            self.assertEqual(set(loaded), {"a", "b"})
            with self.assertRaises(RuntimeError):
                _load_resume_jsonl(path, "another-run")

    def test_jsonl_writer_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.jsonl"
            right = Path(directory) / "right.jsonl"
            records = [{"z": 2, "a": 1, "pair_id": "x", "audit_id": "run"}]
            _write_jsonl(left, records)
            _write_jsonl(right, records)
            self.assertEqual(left.read_bytes(), right.read_bytes())

    def test_vectorized_exact_retrieval_matches_reference(self):
        rng = np.random.default_rng(20261104)
        embeddings = rng.normal(size=(19, 32)).astype(np.float32)
        keys = [f"image-{index:02d}" for index in reversed(range(len(embeddings)))]
        reference = exact_cosine_neighbors(embeddings, keys, top_k=5, chunk_size=4)
        accelerated = exact_cuda_neighbors(
            embeddings, keys, top_k=5, device="cpu", chunk_size=7)
        reference_by_key = {row["key"]: row["neighbors"] for row in reference}
        accelerated_by_key = {row["key"]: row["neighbors"] for row in accelerated}
        self.assertEqual(set(reference_by_key), set(accelerated_by_key))
        for key in keys:
            self.assertEqual(
                [row["key"] for row in reference_by_key[key]],
                [row["key"] for row in accelerated_by_key[key]],
            )
            np.testing.assert_allclose(
                [row["cosine"] for row in reference_by_key[key]],
                [row["cosine"] for row in accelerated_by_key[key]],
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
