from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import asdict, dataclass


@dataclass
class LedgerEntry:
    trace_id: str
    session_id: str
    event_type: str
    threat_type: str
    timestamp: float
    hash: str


class Weilchain:
    def __init__(self, db_path: str = "weilchain.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._bootstrap()

    def _bootstrap(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                threat_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                hash TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def commit(self, trace_id: str, session_id: str, event_type: str, threat_type: str) -> LedgerEntry:
        timestamp = time.time()
        base = f"{trace_id}|{session_id}|{event_type}|{threat_type}|{timestamp}"
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
        entry = LedgerEntry(
            trace_id=trace_id,
            session_id=session_id,
            event_type=event_type,
            threat_type=threat_type,
            timestamp=timestamp,
            hash=digest,
        )
        self._conn.execute(
            "INSERT INTO ledger (trace_id, session_id, event_type, threat_type, timestamp, hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entry.trace_id, entry.session_id, entry.event_type, entry.threat_type, entry.timestamp, entry.hash),
        )
        self._conn.commit()
        return entry

    def get_all(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT trace_id, session_id, event_type, threat_type, timestamp, hash FROM ledger ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]

    def verify(self, entry: dict) -> bool:
        base = (
            f"{entry['trace_id']}|{entry['session_id']}|{entry['event_type']}|"
            f"{entry['threat_type']}|{entry['timestamp']}"
        )
        expected = hashlib.sha256(base.encode("utf-8")).hexdigest()
        return expected == entry.get("hash")
