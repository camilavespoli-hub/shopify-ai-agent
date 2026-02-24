import os
import sqlite3
from datetime import datetime


class DatabaseManager:
    """
    SQLite-backed task logger for the Glomend Blog AI pipeline.

    Tables:
      - task_log   : one row per agent action (agent, title, status, note)
      - run_log    : one row per full Orchestrator run (start, end, summary)

    Used by the Orchestrator and read by the Dashboard.
    """

    def __init__(self, db_path="pipeline.db"):
        self.db_path = db_path
        self._init_db()
        print(f"   🗄️ DatabaseManager connected: {db_path}")

    # ─────────────────────────────────────────────
    # SCHEMA INIT
    # ─────────────────────────────────────────────

    def _init_db(self):
        """Creates tables if they don't exist yet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS task_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    agent       TEXT    NOT NULL,
                    title       TEXT    NOT NULL,
                    status      TEXT    NOT NULL,
                    note        TEXT    DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS run_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at    TEXT    NOT NULL,
                    finished_at   TEXT,
                    total_tasks   INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    fail_count    INTEGER DEFAULT 0,
                    summary_note  TEXT    DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_task_agent     ON task_log(agent);
                CREATE INDEX IF NOT EXISTS idx_task_status    ON task_log(status);
                CREATE INDEX IF NOT EXISTS idx_task_timestamp ON task_log(timestamp);
            """)
            conn.commit()

    # ─────────────────────────────────────────────
    # WRITE
    # ─────────────────────────────────────────────

    def log_task(self, agent: str, title: str, status: str, note: str = ""):
        """
        Logs one agent action.
        Called by the Orchestrator after every agent step.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO task_log (timestamp, agent, title, status, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, agent, str(title), str(status), str(note))
            )
            conn.commit()

    def start_run(self) -> int:
        """
        Opens a new run record. Returns run_id for later closing.
        Call at the start of Orchestrator.run().
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO run_log (started_at) VALUES (?)", (ts,)
            )
            conn.commit()
            return cursor.lastrowid

    def finish_run(self, run_id: int, success: int, fail: int, note: str = ""):
        """Closes a run record with final counts."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE run_log
                   SET finished_at   = ?,
                       total_tasks   = ?,
                       success_count = ?,
                       fail_count    = ?,
                       summary_note  = ?
                   WHERE id = ?""",
                (ts, success + fail, success, fail, note, run_id)
            )
            conn.commit()

    # ─────────────────────────────────────────────
    # READ — used by Dashboard
    # ─────────────────────────────────────────────

    def get_recent_logs(self, limit: int = 100) -> list[dict]:
        """Returns the N most recent task_log rows, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM task_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_summary(self) -> list[dict]:
        """
        Per-agent counts: successes, failures, needs_review, total.
        Used by Dashboard bar chart.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    agent,
                    SUM(CASE WHEN status = 'SUCCESS'
                        THEN 1 ELSE 0 END)                      AS successes,
                    SUM(CASE WHEN status IN ('BLOCKED','FAILED','HARD_FAIL')
                        THEN 1 ELSE 0 END)                      AS failures,
                    SUM(CASE WHEN status IN ('NEEDS REVIEW','NEEDS_REVIEW')
                        THEN 1 ELSE 0 END)                      AS needs_review,
                    COUNT(*)                                     AS total
                FROM task_log
                GROUP BY agent
                ORDER BY agent
            """).fetchall()
        return [dict(r) for r in rows]

    def get_status_breakdown(self) -> list[dict]:
        """All distinct statuses and their total counts. Used for pie chart."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT status, COUNT(*) AS count
                FROM task_log
                GROUP BY status
                ORDER BY count DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        """Returns the N most recent pipeline runs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM run_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_totals(self) -> dict:
        """Summary counts for the Dashboard KPI cards."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            t = conn.execute("""
                SELECT
                    COUNT(*)                                              AS total,
                    SUM(CASE WHEN status = 'SUCCESS'           THEN 1 ELSE 0 END) AS successes,
                    SUM(CASE WHEN status IN ('BLOCKED','FAILED','HARD_FAIL')
                                                               THEN 1 ELSE 0 END) AS failures,
                    SUM(CASE WHEN status IN ('NEEDS REVIEW','NEEDS_REVIEW')
                                                               THEN 1 ELSE 0 END) AS needs_review
                FROM task_log
            """).fetchone()
        return dict(t) if t else {"total": 0, "successes": 0,
                                   "failures": 0, "needs_review": 0}

