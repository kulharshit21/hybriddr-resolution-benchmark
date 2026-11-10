#!/usr/bin/env python3
"""Local-only, resumable blinded near-duplicate review interface.

The server never exposes private source paths, labels, dataset names, or split
names. Decisions are stored transactionally in SQLite. A complete, duplicate-
free JSONL ledger compatible with ``audit_duplicates.py finalize`` is exported
only after every required pair has a final confirmed/rejected decision.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import secrets
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


FINAL_DECISIONS = {"confirmed", "rejected"}
ALL_DECISIONS = FINAL_DECISIONS | {"uncertain"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_review_items(pack: Path, verify_images: bool = True) -> list[dict]:
    pack = Path(pack).resolve()
    control = pack / "review_pairs.json"
    sheets = pack / "contact_sheets"
    if not control.is_file():
        raise FileNotFoundError(f"Missing review control file: {control}")
    if not sheets.is_dir():
        raise FileNotFoundError(f"Missing contact-sheet directory: {sheets}")
    raw = json.loads(control.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("review_pairs.json must be a non-empty JSON list")
    expected_keys = {"pair_id", "left_token", "right_token"}
    seen: set[str] = set()
    items: list[dict] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError(f"Invalid blinded record at index {index}")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or len(pair_id) != 64:
            raise ValueError(f"Invalid pair_id at index {index}")
        try:
            int(pair_id, 16)
        except ValueError as exc:
            raise ValueError(f"Non-hex pair_id at index {index}") from exc
        if pair_id in seen:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        seen.add(pair_id)
        for key in ("left_token", "right_token"):
            value = row[key]
            if not isinstance(value, str) or len(value) != 16:
                raise ValueError(f"Invalid {key} for {pair_id}")
        image = sheets / f"{pair_id}.jpg"
        if verify_images and not image.is_file():
            raise FileNotFoundError(f"Missing review sheet: {image}")
        items.append(dict(row))
    return items


class DecisionStore:
    def __init__(self, database: Path, pair_ids: list[str], reviewer: str):
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.pair_ids = tuple(pair_ids)
        self.required = frozenset(pair_ids)
        self.reviewer = reviewer.strip()
        if not self.reviewer:
            raise ValueError("Reviewer identity must not be empty")
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    pair_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL CHECK(decision IN
                        ('confirmed', 'rejected', 'uncertain')),
                    reviewer TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS decision_history (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair_id TEXT NOT NULL,
                    previous_decision TEXT,
                    decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT ''
                );
                """
            )
            identity = hashlib.sha256("\n".join(self.pair_ids).encode()).hexdigest()
            existing = dict(connection.execute("SELECT key, value FROM metadata"))
            expected = {
                "review_identity": identity,
                "required_pairs": str(len(self.pair_ids)),
                "reviewer": self.reviewer,
            }
            if existing:
                for key, value in expected.items():
                    if existing.get(key) != value:
                        raise RuntimeError(
                            f"Review database {key} mismatch: "
                            f"{existing.get(key)!r} != {value!r}"
                        )
            else:
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    expected.items(),
                )
            unknown = [
                row[0]
                for row in connection.execute("SELECT pair_id FROM decisions")
                if row[0] not in self.required
            ]
            if unknown:
                raise RuntimeError(
                    f"Review database contains {len(unknown)} unknown pair IDs"
                )

    def record(self, pair_id: str, decision: str, notes: str = "") -> dict:
        if pair_id not in self.required:
            raise ValueError("Unknown pair_id")
        if decision not in ALL_DECISIONS:
            raise ValueError("Decision must be confirmed, rejected, or uncertain")
        notes = notes.strip()
        if len(notes) > 1000:
            raise ValueError("Notes must be at most 1000 characters")
        timestamp = utc_now()
        with self._lock, self._connect() as connection:
            previous_row = connection.execute(
                "SELECT decision FROM decisions WHERE pair_id=?", (pair_id,)
            ).fetchone()
            previous = previous_row[0] if previous_row else None
            connection.execute(
                """
                INSERT INTO decisions(pair_id, decision, reviewer, timestamp, notes)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pair_id) DO UPDATE SET
                    decision=excluded.decision,
                    reviewer=excluded.reviewer,
                    timestamp=excluded.timestamp,
                    notes=excluded.notes
                """,
                (pair_id, decision, self.reviewer, timestamp, notes),
            )
            connection.execute(
                """
                INSERT INTO decision_history(
                    pair_id, previous_decision, decision, reviewer, timestamp, notes
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (pair_id, previous, decision, self.reviewer, timestamp, notes),
            )
        return {
            "pair_id": pair_id,
            "decision": decision,
            "reviewer": self.reviewer,
            "timestamp": timestamp,
            "notes": notes,
        }

    def get(self, pair_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM decisions WHERE pair_id=?", (pair_id,)
            ).fetchone()
        return dict(row) if row else None

    def counts(self) -> dict[str, int]:
        counts = {key: 0 for key in ALL_DECISIONS}
        with self._connect() as connection:
            for row in connection.execute(
                "SELECT decision, COUNT(*) FROM decisions GROUP BY decision"
            ):
                counts[row[0]] = row[1]
        counts["total"] = len(self.pair_ids)
        counts["reviewed"] = sum(counts[key] for key in ALL_DECISIONS)
        counts["final"] = sum(counts[key] for key in FINAL_DECISIONS)
        counts["pending"] = counts["total"] - counts["reviewed"]
        return counts

    def next_pending_index(self, after_index: int = -1) -> int | None:
        with self._connect() as connection:
            decided = {row[0] for row in connection.execute("SELECT pair_id FROM decisions")}
        total = len(self.pair_ids)
        for offset in range(1, total + 1):
            index = (after_index + offset) % total
            if self.pair_ids[index] not in decided:
                return index
        return None

    def export(self, output: Path) -> dict:
        output = Path(output).resolve()
        with self._lock, self._connect() as connection:
            rows = {
                row["pair_id"]: dict(row)
                for row in connection.execute("SELECT * FROM decisions")
            }
        missing = self.required - rows.keys()
        uncertain = [
            pair_id for pair_id, row in rows.items() if row["decision"] == "uncertain"
        ]
        if missing or uncertain:
            raise RuntimeError(
                f"Cannot export: missing={len(missing)}, uncertain={len(uncertain)}"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", delete=False, dir=output.parent
        ) as handle:
            temporary = Path(handle.name)
            for pair_id in sorted(self.required):
                row = rows[pair_id]
                record = {
                    "pair_id": pair_id,
                    "decision": row["decision"],
                    "reviewer": row["reviewer"],
                    "timestamp": row["timestamp"],
                }
                if row["notes"]:
                    record["notes"] = row["notes"]
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        temporary.replace(output)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return {"path": str(output), "lines": len(rows), "sha256": digest}


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HybridDR Blinded Duplicate Review</title>
<style>
:root{font-family:system-ui,sans-serif;color-scheme:dark;background:#101418;color:#eef2f5}
body{margin:0} header{padding:10px 18px;background:#192129;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
#sheet{display:block;margin:12px auto;max-width:calc(100vw - 24px);max-height:calc(100vh - 245px);object-fit:contain;background:#000}
.controls{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;padding:8px}
button{font-size:16px;padding:10px 18px;border:0;border-radius:7px;cursor:pointer}
.yes{background:#167d45;color:white}.no{background:#a33131;color:white}.maybe{background:#9a6b12;color:white}
.nav{background:#405060;color:white}.export{background:#315ca8;color:white}
#notes{width:min(800px,calc(100vw - 40px));display:block;margin:5px auto;padding:8px}
#message{text-align:center;min-height:24px}.help{text-align:center;color:#b8c2ca;font-size:13px;padding:4px 18px}
</style></head><body>
<header><strong>HybridDR blinded review</strong><span id="position"></span><span id="counts"></span><span id="existing"></span></header>
<img id="sheet" alt="Blinded side-by-side retinal comparison">
<input id="notes" maxlength="1000" placeholder="Optional note (required only for uncertainty/adjudication context)">
<div class="controls">
 <button class="nav" onclick="move(-1)">← Previous</button>
 <button class="yes" onclick="decide('confirmed')">C / 1 — Same image</button>
 <button class="no" onclick="decide('rejected')">R / 2 — Different image</button>
 <button class="maybe" onclick="decide('uncertain')">U / 3 — Uncertain</button>
 <button class="nav" onclick="move(1)">Next →</button>
 <button class="nav" onclick="nextPending()">Next pending</button>
 <button class="export" onclick="exportLedger()">Export final JSONL</button>
</div>
<div id="message"></div>
<div class="help">Confirm only when both panels show the same underlying retinal photograph despite crop, rotation, mirror, compression, colour, or illumination changes. Similar disease appearance alone is not a duplicate. Uncertain items must later be adjudicated.</div>
<script>
const TOKEN='__TOKEN__'; let index=0,total=0,current=null;
async function api(url,options={}){options.headers={...(options.headers||{}),'X-Review-Token':TOKEN};const r=await fetch(url,options);const x=await r.json();if(!r.ok)throw new Error(x.error||r.statusText);return x}
async function load(i){try{const x=await api('/api/item?index='+i);index=x.index;total=x.total;current=x.item;document.getElementById('sheet').src=x.image_url;document.getElementById('position').textContent=`Item ${index+1} / ${total}`;document.getElementById('existing').textContent=x.decision?`Saved: ${x.decision.decision}`:'Pending';document.getElementById('notes').value=x.decision?.notes||'';showCounts(x.counts);msg('')}catch(e){msg(e.message,true)}}
function showCounts(c){document.getElementById('counts').textContent=`Final ${c.final} · uncertain ${c.uncertain} · pending ${c.pending}`}
function msg(s,bad=false){const e=document.getElementById('message');e.textContent=s;e.style.color=bad?'#ff8a8a':'#9ee6ad'}
async function decide(decision){if(!current)return;try{const x=await api('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pair_id:current.pair_id,decision,notes:document.getElementById('notes').value})});showCounts(x.counts);msg(`Saved ${decision}`);if(x.next_pending_index!==null)load(x.next_pending_index);else load(index)}catch(e){msg(e.message,true)}}
function move(d){load((index+d+total)%total)} async function nextPending(){try{const x=await api('/api/next-pending?after='+index);if(x.index===null)msg('No pending items; resolve uncertainties or export.');else load(x.index)}catch(e){msg(e.message,true)}}
async function exportLedger(){try{const x=await api('/api/export',{method:'POST'});msg(`Exported ${x.lines} rows; SHA-256 ${x.sha256}`)}catch(e){msg(e.message,true)}}
document.addEventListener('keydown',e=>{if(e.target.id==='notes')return;if(e.key==='c'||e.key==='1')decide('confirmed');else if(e.key==='r'||e.key==='2')decide('rejected');else if(e.key==='u'||e.key==='3')decide('uncertain');else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1)});load(0);
</script></body></html>"""


class ReviewApplication:
    def __init__(self, pack: Path, items: list[dict], store: DecisionStore,
                 output: Path, token: str):
        self.pack = Path(pack).resolve()
        self.sheets = self.pack / "contact_sheets"
        self.store = store
        self.output = Path(output).resolve()
        self.token = token
        self.items = [dict(row) for row in items]
        self.index_by_pair = {row["pair_id"]: i for i, row in enumerate(self.items)}


def make_handler(application: ReviewApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HybridDRReview/1.0"

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'",
            )
            self.end_headers()

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self._headers(status, "application/json; charset=utf-8", len(body))
            self.wfile.write(body)

        def _error(self, error: Exception, status: int = 400) -> None:
            self._json({"error": str(error)}, status)

        def _authorized(self) -> bool:
            return secrets.compare_digest(
                self.headers.get("X-Review-Token", ""), application.token
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = INDEX_HTML.replace("__TOKEN__", html.escape(application.token)).encode()
                self._headers(200, "text/html; charset=utf-8", len(body))
                self.wfile.write(body)
                return
            if parsed.path.startswith("/images/"):
                pair_id = parsed.path.removeprefix("/images/").removesuffix(".jpg")
                if pair_id not in application.index_by_pair:
                    self._error(ValueError("Unknown image"), HTTPStatus.NOT_FOUND)
                    return
                body = (application.sheets / f"{pair_id}.jpg").read_bytes()
                self._headers(200, "image/jpeg", len(body))
                self.wfile.write(body)
                return
            if not self._authorized():
                self._error(PermissionError("Invalid review token"), HTTPStatus.FORBIDDEN)
                return
            try:
                query = parse_qs(parsed.query)
                if parsed.path == "/api/item":
                    index = int(query.get("index", ["0"])[0])
                    if not 0 <= index < len(application.items):
                        raise ValueError("Index out of range")
                    item = application.items[index]
                    self._json({
                        "index": index,
                        "total": len(application.items),
                        "item": item,
                        "image_url": f"/images/{item['pair_id']}.jpg",
                        "decision": application.store.get(item["pair_id"]),
                        "counts": application.store.counts(),
                    })
                elif parsed.path == "/api/next-pending":
                    after = int(query.get("after", ["-1"])[0])
                    self._json({"index": application.store.next_pending_index(after)})
                else:
                    self._error(ValueError("Not found"), HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._error(exc)

        def do_POST(self) -> None:
            if not self._authorized():
                self._error(PermissionError("Invalid review token"), HTTPStatus.FORBIDDEN)
                return
            try:
                if self.path == "/api/decision":
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 8192:
                        raise ValueError("Request body too large")
                    payload = json.loads(self.rfile.read(length))
                    saved = application.store.record(
                        payload.get("pair_id", ""),
                        payload.get("decision", ""),
                        payload.get("notes", ""),
                    )
                    current = application.index_by_pair[saved["pair_id"]]
                    self._json({
                        "saved": saved,
                        "counts": application.store.counts(),
                        "next_pending_index": application.store.next_pending_index(current),
                    })
                elif self.path == "/api/export":
                    self._json(application.store.export(application.output))
                else:
                    self._error(ValueError("Not found"), HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._error(exc)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-image-verification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_review_items(args.pack, not args.skip_image_verification)
    store = DecisionStore(args.database, [row["pair_id"] for row in items], args.reviewer)
    token = secrets.token_urlsafe(32)
    application = ReviewApplication(args.pack, items, store, args.output, token)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(application))
    print(f"Verified blinded review items: {len(items)}", flush=True)
    print(f"Reviewer: {store.reviewer}", flush=True)
    print(f"Resume counts: {json.dumps(store.counts(), sort_keys=True)}", flush=True)
    print(f"Open: http://{args.host}:{args.port}/", flush=True)
    print("The server binds locally by default; use an SSH tunnel for remote access.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
