"""SQLite persistence for jobs, items and size presets.

Item lifecycle:
  uploaded -> (start) -> queued -> processing -> done | failed
  uploaded -> (not selected at start) -> skipped
  invalid  (rejected at upload, never processed)
  queued   -> canceled
  failed   -> (retry) -> queued
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from . import config

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    state TEXT NOT NULL DEFAULT 'draft',          -- draft | started
    settings TEXT
);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    original_name TEXT NOT NULL,
    input_file TEXT,
    size_bytes INTEGER,
    width INTEGER, height INTEGER, format TEXT,
    status TEXT NOT NULL,
    error TEXT,
    output_file TEXT,
    output_name TEXT,
    report TEXT,
    est_seconds REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    queued_at REAL, started_at REAL, finished_at REAL
);
CREATE INDEX IF NOT EXISTS ix_items_job ON items(job_id, idx);
CREATE INDEX IF NOT EXISTS ix_items_status ON items(status, queued_at);
CREATE TABLE IF NOT EXISTS presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    ratio_w REAL, ratio_h REAL, height INTEGER,
    builtin INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""

BUILTIN_PRESETS = [
    ("方形主图 800（10:10）", 800, 10, 10),
    ("方形主图 1200（10:10）", 1200, 10, 10),
    ("竖版 750（10:13.3≈3:4）", 750, 3, 4),
    ("横版 1920（16:9）", 1920, 16, 9),
]


def connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(config.DATA_DIR / "app.db", check_same_thread=False, isolation_level=None)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA foreign_keys=ON")
            _conn.executescript(SCHEMA)
            if _conn.execute("SELECT COUNT(*) FROM presets").fetchone()[0] == 0:
                for name, w, rw, rh in BUILTIN_PRESETS:
                    _conn.execute("INSERT INTO presets(name,width,ratio_w,ratio_h,builtin,created_at) VALUES(?,?,?,?,1,?)",
                                  (name, w, rw, rh, time.time()))
        return _conn


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    c = connect()
    with _lock:
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise


def q(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    c = connect()
    with _lock:
        return c.execute(sql, args).fetchall()


def q1(sql: str, args: tuple = ()) -> sqlite3.Row | None:
    rows = q(sql, args)
    return rows[0] if rows else None


def ex(sql: str, args: tuple = ()) -> int:
    c = connect()
    with _lock:
        return c.execute(sql, args).rowcount


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def row_to_dict(r: sqlite3.Row | None) -> dict[str, Any] | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("settings", "report"):
        if d.get(k):
            d[k] = json.loads(d[k])
    return d


def kv_get(k: str, default: Any = None) -> Any:
    r = q1("SELECT v FROM kv WHERE k=?", (k,))
    return json.loads(r["v"]) if r else default


def kv_set(k: str, v: Any) -> None:
    ex("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))
