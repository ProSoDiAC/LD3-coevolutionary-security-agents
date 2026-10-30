"""Optional PostgreSQL ingestion of trajectory JSONL.

JSONL is always written; Postgres is a convenience for querying at scale. This
module degrades gracefully if psycopg is not installed or no DSN is configured.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    id           BIGSERIAL PRIMARY KEY,
    run_name     TEXT,
    config_hash  TEXT,
    episode      INT,
    turn         INT,
    agent        TEXT,
    role         TEXT,
    action       TEXT,
    target       TEXT,
    reason       TEXT,
    result       TEXT,
    success      BOOLEAN,
    strategy     TEXT,
    cost         DOUBLE PRECISION,
    timestamp    DOUBLE PRECISION,
    record       JSONB
);
CREATE INDEX IF NOT EXISTS idx_traj_run ON trajectories (run_name, episode);
"""


def available() -> bool:
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("COEVSEC_PG_DSN"))


def ingest_run(run_dir: str | Path, dsn: str | None = None) -> int:
    """Ingest a run's trajectories.jsonl into Postgres. Returns rows inserted."""
    import psycopg

    dsn = dsn or os.environ.get("COEVSEC_PG_DSN")
    if not dsn:
        raise RuntimeError("no COEVSEC_PG_DSN configured")

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    run_name, config_hash = meta.get("run_name"), meta.get("config_hash")
    rows = 0
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(_SCHEMA)
        with (run_dir / "trajectories.jsonl").open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    # A truncated trailing line (e.g. from an interrupted run)
                    # should not roll back every row already ingested for it.
                    continue
                cur.execute(
                    """INSERT INTO trajectories
                       (run_name, config_hash, episode, turn, agent, role, action,
                        target, reason, result, success, strategy, cost, timestamp, record)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (run_name, config_hash, r["episode"], r["turn"], r["agent"], r["role"],
                     r["action"], r.get("target"), r.get("reason"), r.get("result"),
                     r.get("success"), r.get("strategy"), r.get("cost"), r.get("timestamp"),
                     json.dumps(r)),
                )
                rows += 1
        conn.commit()
    return rows
