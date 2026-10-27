"""Append-only durable run state for long scientific research loops."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import EvidenceEntry, RunSnapshot, to_dict


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """SQLite event store.  Snapshots accelerate resume; trace rows remain immutable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A modest timeout lets concurrent readers wait for a short commit;
        # per-run controller leases below prevent two processes from advancing
        # the same durable state machine at once.
        self._connection = sqlite3.connect(self.path, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_runs (
              run_id TEXT PRIMARY KEY, question TEXT NOT NULL, created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL, stage TEXT NOT NULL, snapshot_json TEXT NOT NULL,
              controller_pid INTEGER, controller_token TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS research_trace (
              trace_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, timestamp TEXT NOT NULL,
              event_type TEXT NOT NULL, payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence_ledger (
              evidence_id TEXT NOT NULL, run_id TEXT NOT NULL, recorded_at TEXT NOT NULL,
              entry_json TEXT NOT NULL, PRIMARY KEY(run_id, evidence_id)
            );
            CREATE TABLE IF NOT EXISTS research_action_attempts (
              run_id TEXT NOT NULL,
              action_id TEXT NOT NULL,
              input_fingerprint TEXT NOT NULL,
              component_id TEXT NOT NULL,
              kind TEXT NOT NULL,
              tool_id TEXT NOT NULL,
              status TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              evidence_ids_json TEXT NOT NULL,
              error TEXT NOT NULL,
              started_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(run_id, action_id, input_fingerprint)
            );
            """
        )
        # Existing checkpoints predate the explicit idempotency column.  Keep
        # them resumable instead of requiring a destructive migration.
        attempt_columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(research_action_attempts)")}
        if "idempotency_key" not in attempt_columns:
            self._connection.execute("ALTER TABLE research_action_attempts ADD COLUMN idempotency_key TEXT NOT NULL DEFAULT ''")
        run_columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(research_runs)")}
        if "controller_pid" not in run_columns:
            self._connection.execute("ALTER TABLE research_runs ADD COLUMN controller_pid INTEGER")
        if "controller_token" not in run_columns:
            self._connection.execute("ALTER TABLE research_runs ADD COLUMN controller_token TEXT NOT NULL DEFAULT ''")
        self._connection.commit()

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def create(self, question: str) -> RunSnapshot:
        run_id = str(uuid.uuid4())
        snapshot = RunSnapshot(run_id=run_id, stage="created", question=question)
        now = utcnow()
        self._connection.execute(
            """INSERT INTO research_runs
               (run_id, question, created_at, updated_at, stage, snapshot_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, question, now, now, snapshot.stage, json.dumps(to_dict(snapshot), ensure_ascii=False)),
        )
        self._connection.commit()
        self.append_trace(run_id, "run_created", {"question": question})
        return snapshot

    def acquire_run(self, run_id: str) -> str:
        """Acquire the single durable controller lease for one run.

        A dead process's PID is recoverable; a live controller cannot be
        silently requeued by a concurrent ``resume`` process.
        """

        token = str(uuid.uuid4())
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT controller_pid, controller_token FROM research_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            owner_pid = row["controller_pid"]
            owner_token = row["controller_token"] or ""
            if owner_token and owner_pid and self._pid_is_alive(int(owner_pid)):
                raise RuntimeError(f"run {run_id} already has a live controller process")
            self._connection.execute(
                "UPDATE research_runs SET controller_pid=?, controller_token=?, updated_at=? WHERE run_id=?",
                (os.getpid(), token, utcnow(), run_id),
            )
            self._insert_action_trace(run_id, "run_controller_acquired", {
                "controller_pid": os.getpid(), "recovered_stale_lease": bool(owner_token),
            })
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise
        return token

    def release_run(self, run_id: str, token: str) -> None:
        with self._connection:
            self._connection.execute(
                """UPDATE research_runs SET controller_pid=NULL, controller_token='', updated_at=?
                   WHERE run_id=? AND controller_token=?""",
                (utcnow(), run_id, token),
            )

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def save(self, snapshot: RunSnapshot) -> None:
        self._connection.execute(
            "UPDATE research_runs SET updated_at=?, stage=?, snapshot_json=? WHERE run_id=?",
            (utcnow(), snapshot.stage, json.dumps(to_dict(snapshot), ensure_ascii=False), snapshot.run_id),
        )
        self._connection.commit()

    def checkpoint(self, snapshot: RunSnapshot, event_type: str, payload: dict[str, Any]) -> str:
        """Atomically persist a snapshot and its transition event."""

        trace_id = str(uuid.uuid4())
        with self._connection:
            current = self._connection.execute(
                "SELECT stage FROM research_runs WHERE run_id=?",
                (snapshot.run_id,),
            ).fetchone()
            # A completed run is a durable result, not an ordinary
            # intermediate checkpoint.  A stale controller or a bad resume
            # path must not move it back into research and make the same
            # local graph spend API calls again.  Failed runs remain resumable
            # because retrying a failed provider/action is an explicit
            # recovery operation.
            if (
                current is not None
                and current["stage"] == "completed"
                and snapshot.stage != "completed"
                and event_type in {
                    "research_round_started",
                    "planner_reopened_local_graph",
                    "planner_global_patch",
                    "planner_formalized_evidence_gap",
                    "resume_requeued_actions",
                }
            ):
                raise RuntimeError(
                    f"completed run {snapshot.run_id} cannot transition through {event_type}"
                )
            self._connection.execute(
                "UPDATE research_runs SET updated_at=?, stage=?, snapshot_json=? WHERE run_id=?",
                (utcnow(), snapshot.stage, json.dumps(to_dict(snapshot), ensure_ascii=False), snapshot.run_id),
            )
            self._connection.execute(
                "INSERT INTO research_trace VALUES (?, ?, ?, ?, ?)",
                (trace_id, snapshot.run_id, utcnow(), event_type, json.dumps(payload, ensure_ascii=False)),
            )
        return trace_id

    def raw_snapshot(self, run_id: str) -> dict[str, Any]:
        row = self._connection.execute("SELECT snapshot_json FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row["snapshot_json"])

    def append_trace(self, run_id: str, event_type: str, payload: dict[str, Any]) -> str:
        trace_id = str(uuid.uuid4())
        with self._connection:
            self._connection.execute(
                "INSERT INTO research_trace VALUES (?, ?, ?, ?, ?)",
                (trace_id, run_id, utcnow(), event_type, json.dumps(payload, ensure_ascii=False)),
            )
        return trace_id

    def _insert_action_trace(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT INTO research_trace VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), run_id, utcnow(), event_type, json.dumps(payload, ensure_ascii=False)),
        )

    def trace(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM research_trace WHERE run_id=? ORDER BY timestamp, rowid", (run_id,)).fetchall()
        return [{"trace_id": row["trace_id"], "timestamp": row["timestamp"], "event_type": row["event_type"], "payload": json.loads(row["payload_json"])} for row in rows]

    def append_evidence(self, run_id: str, entry: EvidenceEntry) -> None:
        recorded_at = entry.recorded_at or utcnow()
        with self._connection:
            # Evidence IDs are durable identities.  A repeated recovery write
            # must not replace an already committed result with a duplicate
            # attempt's payload.
            self._connection.execute(
                "INSERT OR IGNORE INTO evidence_ledger VALUES (?, ?, ?, ?)",
                (entry.evidence_id, run_id, recorded_at, json.dumps(to_dict(entry), ensure_ascii=False)),
            )

    def begin_action(self, run_id: str, action_id: str, input_fingerprint: str, *, component_id: str, kind: str, tool_id: str = "") -> dict[str, Any]:
        """Durably mark an action before its external/tool work starts.

        ``INSERT OR IGNORE`` makes this an idempotency boundary: a resumed
        process can inspect the same action key instead of creating a second
        attempt record.
        """
        now = utcnow()
        idempotency_key = f"research:{run_id}:{action_id}:{input_fingerprint}"
        with self._connection:
            cursor = self._connection.execute(
                """INSERT OR IGNORE INTO research_action_attempts
                   (run_id, action_id, input_fingerprint, component_id, kind, tool_id,
                    status, idempotency_key, evidence_ids_json, error, started_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'started', ?, '[]', '', ?, ?)""",
                (run_id, action_id, input_fingerprint, component_id, kind, tool_id, idempotency_key, now, now),
            )
            if cursor.rowcount:
                self._insert_action_trace(run_id, "action_attempt_started", {
                    "action_id": action_id, "component_id": component_id, "kind": kind,
                    "tool_id": tool_id, "input_fingerprint": input_fingerprint,
                    "idempotency_key": idempotency_key,
                })
            else:
                restarted = self._connection.execute(
                    """UPDATE research_action_attempts
                       SET status='started', error='', updated_at=?
                       WHERE run_id=? AND action_id=? AND input_fingerprint=? AND status='retryable'""",
                    (now, run_id, action_id, input_fingerprint),
                )
                if restarted.rowcount:
                    self._insert_action_trace(run_id, "action_attempt_restarted", {
                        "action_id": action_id, "component_id": component_id,
                        "input_fingerprint": input_fingerprint, "idempotency_key": idempotency_key,
                    })
        return self.action_attempt(run_id, action_id, input_fingerprint) or {}

    def complete_action(self, run_id: str, action_id: str, input_fingerprint: str, evidence_ids: list[str]) -> None:
        ids = list(dict.fromkeys(evidence_ids))
        with self._connection:
            self._connection.execute(
                """UPDATE research_action_attempts
                   SET status='completed', evidence_ids_json=?, error='', updated_at=?
                   WHERE run_id=? AND action_id=? AND input_fingerprint=?""",
                (json.dumps(ids), utcnow(), run_id, action_id, input_fingerprint),
            )
            self._insert_action_trace(run_id, "action_attempt_completed", {
                "action_id": action_id, "input_fingerprint": input_fingerprint, "evidence_ids": ids,
            })

    def fail_action(self, run_id: str, action_id: str, input_fingerprint: str, error: str) -> None:
        message = error[:2_000]
        with self._connection:
            self._connection.execute(
                """UPDATE research_action_attempts
                   SET status='failed', error=?, updated_at=?
                   WHERE run_id=? AND action_id=? AND input_fingerprint=?""",
                (message, utcnow(), run_id, action_id, input_fingerprint),
            )
            self._insert_action_trace(run_id, "action_attempt_failed", {
                "action_id": action_id, "input_fingerprint": input_fingerprint, "error": message,
            })

    def requeue_inflight_actions(self, run_id: str) -> list[dict[str, Any]]:
        """Convert abandoned in-flight attempts into resumable attempts."""

        rows = self._connection.execute(
            "SELECT * FROM research_action_attempts WHERE run_id=? AND status IN ('started', 'running')",
            (run_id,),
        ).fetchall()
        if not rows:
            return []
        with self._connection:
            for row in rows:
                self._connection.execute(
                    """UPDATE research_action_attempts SET status='retryable', updated_at=?
                       WHERE run_id=? AND action_id=? AND input_fingerprint=?""",
                    (utcnow(), run_id, row["action_id"], row["input_fingerprint"]),
                )
                self._insert_action_trace(run_id, "action_requeued_after_interruption", {
                    "action_id": row["action_id"], "component_id": row["component_id"],
                    "input_fingerprint": row["input_fingerprint"], "previous_status": row["status"],
                })
        return [dict(row) for row in rows]

    def action_attempt(self, run_id: str, action_id: str, input_fingerprint: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM research_action_attempts WHERE run_id=? AND action_id=? AND input_fingerprint=?",
            (run_id, action_id, input_fingerprint),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"], "action_id": row["action_id"],
            "input_fingerprint": row["input_fingerprint"], "component_id": row["component_id"],
            "kind": row["kind"], "tool_id": row["tool_id"], "status": row["status"],
            "idempotency_key": row["idempotency_key"],
            "evidence_ids": json.loads(row["evidence_ids_json"]), "error": row["error"],
            "started_at": row["started_at"], "updated_at": row["updated_at"],
        }

    def evidence(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT entry_json FROM evidence_ledger WHERE run_id=? ORDER BY evidence_id", (run_id,)).fetchall()
        return [json.loads(row["entry_json"]) for row in rows]

    def close(self) -> None:
        self._connection.close()
