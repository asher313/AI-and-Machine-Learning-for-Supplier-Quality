"""Scoped append-only notes and derived summaries with source provenance."""

import datetime as dt
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CarMemory:
    supplier_id: str
    user: object  # Trusted AssistantUser, constructed by the server.
    store_path: str
    summarizer: object  # Approved adapter; input/output classification stays its responsibility.
    policy_revision: str
    max_working: int = 20
    retriever: object | None = None

    def __post_init__(self):
        if self.max_working < 2:
            raise ValueError("max_working must be at least two")
        scope = {
            "supplier": self.supplier_id,
            "user": self.user.user_id,
            "programs": sorted(self.user.allowed_programs),
            "sources": sorted(self.user.allowed_source_types),
            "policy_revision": self.policy_revision,
        }
        self.scope = hashlib.sha256(
            json.dumps(scope, sort_keys=True).encode()
        ).hexdigest()
        path = Path(self.store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_events (
              id text PRIMARY KEY, scope text NOT NULL, kind text NOT NULL,
              recorded_at text NOT NULL, payload text NOT NULL);
            CREATE INDEX IF NOT EXISTS memory_scope ON memory_events(scope,kind,recorded_at);
            CREATE TABLE IF NOT EXISTS memory_coverage (
              note_id text PRIMARY KEY REFERENCES memory_events(id),
              summary_id text NOT NULL REFERENCES memory_events(id));
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.store_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def add(self, note, *, source_ids, review_state="proposal"):
        if (
            not note.strip()
            or len(note) > 10000
            or not source_ids
        ):
            raise ValueError(
                "bounded note and source IDs required"
            )
        if review_state not in {
            "proposal",
            "reviewed_note",
            "confirmed_outcome",
        }:
            raise ValueError("explicit note status required")
        # The caller must substantiate confirmed_outcome; a model cannot promote itself.
        stamp = dt.datetime.now(dt.UTC).isoformat()
        event_id = uuid.uuid4().hex
        payload = {
            "text": note,
            "source_ids": list(source_ids),
            "review_state": review_state,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_events VALUES (?,?,?,?,?)",
                (
                    event_id,
                    self.scope,
                    "note",
                    stamp,
                    json.dumps(payload),
                ),
            )
            working = conn.execute(
                """SELECT * FROM memory_events e WHERE scope=? AND kind='note'
                AND NOT EXISTS (SELECT 1 FROM memory_coverage c WHERE c.note_id=e.id)
                ORDER BY recorded_at,id""",
                (self.scope,),
            ).fetchall()
            if len(working) > self.max_working:
                oldest = working[: self.max_working // 2]
                notes = [
                    {
                        "event_id": r["id"],
                        **json.loads(r["payload"]),
                    }
                    for r in oldest
                ]
                summary = self.summarizer(notes)
                if (
                    not isinstance(summary, str)
                    or not summary.strip()
                    or len(summary) > 4000
                ):
                    raise ValueError(
                        "bounded nonempty summary required"
                    )
                sid = uuid.uuid4().hex
                value = {
                    "text": summary,
                    "status": "unverified_derived_summary",
                    "source_event_ids": [r["id"] for r in oldest],
                    "source_ids": sorted(
                        {
                            s
                            for n in notes
                            for s in n["source_ids"]
                        }
                    ),
                }
                conn.execute(
                    "INSERT INTO memory_events VALUES (?,?,?,?,?)",
                    (
                        sid,
                        self.scope,
                        "summary",
                        stamp,
                        json.dumps(value),
                    ),
                )
                conn.executemany(
                    "INSERT INTO memory_coverage VALUES (?,?)",
                    [(r["id"], sid) for r in oldest],
                )
        return event_id

    def recall(self, query, k=3, *, asof=None):
        if k < 1:
            raise ValueError("positive retrieval limit required")
        asof = asof or dt.datetime.now(dt.UTC).date()
        cutoff = dt.datetime.combine(
            asof + dt.timedelta(days=1), dt.time(), dt.UTC
        ).isoformat()
        with self._connect() as conn:
            # Only summaries already recorded by the cutoff suppress their notes.
            working = conn.execute(
                """SELECT e.* FROM memory_events e WHERE e.scope=? AND e.kind='note' AND e.recorded_at<?
                AND NOT EXISTS (SELECT 1 FROM memory_coverage c JOIN memory_events s ON s.id=c.summary_id
                                WHERE c.note_id=e.id AND s.recorded_at<?)
                ORDER BY e.recorded_at DESC,e.id DESC LIMIT ?""",
                (self.scope, cutoff, cutoff, self.max_working),
            ).fetchall()
            episodes = conn.execute(
                "SELECT * FROM memory_events WHERE scope=? AND kind='summary' AND recorded_at<? ORDER BY recorded_at DESC,id DESC LIMIT 3",
                (self.scope, cutoff),
            ).fetchall()
        retriever = self.retriever
        if retriever is None:
            from sqm_ai.assistant.retrieve import fetch_chunks

            retriever = fetch_chunks
        chunks = retriever(query, self.user, asof=asof, keep=k)
        return {
            "working": [
                {"event_id": r["id"], **json.loads(r["payload"])}
                for r in reversed(working)
            ],
            "episodic": [
                json.loads(r["payload"])
                for r in reversed(episodes)
            ],
            "semantic": chunks,
        }
