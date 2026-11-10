import json
import tempfile
import unittest
from pathlib import Path

try:
    from .analyze_review_workload import analyze
except ImportError:
    from analyze_review_workload import analyze


class ReviewWorkloadAnalysisTests(unittest.TestCase):
    def test_analysis_binds_inputs_and_selects_strongest_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit"
            audit.mkdir()
            manifest = root / "manifest.csv"
            manifest.write_text(
                "path,source,split\n"
                "a,EyePACS,train\n"
                "b,APTOS,val\n"
                "c,MESSIDOR2,external\n",
                encoding="utf-8",
            )
            ids = ["1" * 64, "2" * 64]
            (audit / "review_required.json").write_text(json.dumps(ids))
            cpu = [
                {"pair_id": ids[0], "left": "a", "right": "b",
                 "classification": "borderline", "hamming_distance": 30,
                 "evidence": {"ssim": .80, "ncc": .85, "nmae": .2,
                              "vessel_dice": .6}},
                {"pair_id": ids[1], "left": "a", "right": "c",
                 "classification": "rejected", "hamming_distance": 45,
                 "evidence": {"ssim": .3, "ncc": .4, "nmae": .5,
                              "vessel_dice": .2}},
            ]
            dino = [
                {"pair_id": ids[0], "left": "a", "right": "b",
                 "classification": "confirmed", "cosine": .99,
                 "evidence": {"ssim": .95, "ncc": .97, "nmae": .08,
                              "vessel_dice": .82}},
            ]
            for name, rows in (("candidate_pairs.jsonl", cpu),
                               ("dino_pairs.jsonl", dino)):
                (audit / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            output = root / "output"
            summary = analyze(audit, manifest, output)
            self.assertEqual(summary["required_pairs"], 2)
            self.assertEqual(summary["classification_counts"], {
                "confirmed": 1, "rejected": 1,
            })
            self.assertEqual(summary["origin_counts"], {"cpu": 1, "cpu+dino": 1})
            self.assertEqual(
                summary["provisional_existing_geometry_buckets"],
                {"strict_negative_geometry": 1, "strict_positive_geometry": 1},
            )
            self.assertEqual(
                len((output / "required_pairs_private.jsonl").read_text().splitlines()), 2
            )

    def test_missing_required_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit"
            audit.mkdir()
            manifest = root / "manifest.csv"
            manifest.write_text("path,source,split\na,EyePACS,train\n")
            (audit / "review_required.json").write_text(json.dumps(["1" * 64]))
            (audit / "candidate_pairs.jsonl").write_text("")
            (audit / "dino_pairs.jsonl").write_text("")
            with self.assertRaises(RuntimeError):
                analyze(audit, manifest, root / "output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
