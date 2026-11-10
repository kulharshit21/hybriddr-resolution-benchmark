import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

try:
    from .review_server import (
        DecisionStore, ReviewApplication, ThreadingHTTPServer,
        load_review_items, make_handler,
    )
except ImportError:  # Direct execution from the pipeline directory.
    from review_server import (
        DecisionStore, ReviewApplication, ThreadingHTTPServer,
        load_review_items, make_handler,
    )


def pair_id(number: int) -> str:
    return f"{number:064x}"


class ReviewServerTests(unittest.TestCase):
    def make_pack(self, root: Path, count: int = 3) -> tuple[Path, list[dict]]:
        pack = root / "pack"
        sheets = pack / "contact_sheets"
        sheets.mkdir(parents=True)
        rows = []
        for number in range(count):
            row = {
                "pair_id": pair_id(number),
                "left_token": f"{number:016x}",
                "right_token": f"{number + 10:016x}",
            }
            rows.append(row)
            (sheets / f"{row['pair_id']}.jpg").write_bytes(b"jpeg")
        (pack / "review_pairs.json").write_text(json.dumps(rows), encoding="utf-8")
        return pack, rows

    def test_pack_validation_rejects_duplicate_and_missing_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack, rows = self.make_pack(root)
            self.assertEqual(len(load_review_items(pack)), 3)
            (pack / "contact_sheets" / f"{rows[0]['pair_id']}.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                load_review_items(pack)
            (pack / "contact_sheets" / f"{rows[0]['pair_id']}.jpg").write_bytes(b"jpeg")
            rows.append(dict(rows[0]))
            (pack / "review_pairs.json").write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_review_items(pack)

    def test_transactional_resume_history_and_identity_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [pair_id(1), pair_id(2)]
            database = root / "review.sqlite3"
            store = DecisionStore(database, ids, "Reviewer A")
            store.record(ids[0], "uncertain", "needs adjudication")
            store.record(ids[0], "confirmed")
            resumed = DecisionStore(database, ids, "Reviewer A")
            self.assertEqual(resumed.get(ids[0])["decision"], "confirmed")
            self.assertEqual(resumed.counts()["pending"], 1)
            with sqlite_connection(database) as connection:
                history = connection.execute(
                    "SELECT previous_decision, decision FROM decision_history ORDER BY sequence"
                ).fetchall()
            self.assertEqual(history, [(None, "uncertain"), ("uncertain", "confirmed")])
            with self.assertRaises(RuntimeError):
                DecisionStore(database, list(reversed(ids)), "Reviewer A")
            with self.assertRaises(RuntimeError):
                DecisionStore(database, ids, "Reviewer B")

    def test_export_is_fail_closed_and_pipeline_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [pair_id(1), pair_id(2), pair_id(3)]
            store = DecisionStore(root / "review.sqlite3", ids, "Reviewer A")
            output = root / "manual-review.jsonl"
            store.record(ids[0], "confirmed")
            store.record(ids[1], "rejected", "visibly different vessels")
            store.record(ids[2], "uncertain")
            with self.assertRaises(RuntimeError):
                store.export(output)
            self.assertFalse(output.exists())
            store.record(ids[2], "rejected")
            result = store.export(output)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(result["lines"], 3)
            self.assertEqual([row["pair_id"] for row in records], sorted(ids))
            self.assertTrue(all(row["decision"] in {"confirmed", "rejected"} for row in records))
            self.assertTrue(all(row["reviewer"] == "Reviewer A" for row in records))
            self.assertTrue(all(row["timestamp"].endswith("+00:00") for row in records))

    def test_http_api_requires_token_and_records_decision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack, items = self.make_pack(root, count=1)
            store = DecisionStore(
                root / "review.sqlite3", [items[0]["pair_id"]], "Reviewer A"
            )
            application = ReviewApplication(
                pack, items, store, root / "manual.jsonl", "secret-token"
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(application))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with self.assertRaises(urllib.error.HTTPError) as failure:
                    urllib.request.urlopen(base + "/api/item?index=0")
                self.assertEqual(failure.exception.code, 403)

                request = urllib.request.Request(
                    base + "/api/decision",
                    data=json.dumps({
                        "pair_id": items[0]["pair_id"],
                        "decision": "confirmed",
                    }).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-Review-Token": "secret-token",
                    },
                    method="POST",
                )
                response = json.loads(urllib.request.urlopen(request).read())
                self.assertEqual(response["saved"]["decision"], "confirmed")
                self.assertEqual(response["counts"]["final"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


class sqlite_connection:
    def __init__(self, path: Path):
        import sqlite3
        self.connection = sqlite3.connect(path)

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, traceback):
        self.connection.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
