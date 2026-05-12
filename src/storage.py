"""SQLite 저장소.

스키마는 단순: (member_cd, fund_cd, std_ymd)가 PK인 fund_nav 한 테이블.
upsert는 ON CONFLICT DO UPDATE로 처리해서 동일 날짜 재호출해도 안전.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .scraper import NavRow


SCHEMA = """
CREATE TABLE IF NOT EXISTS fund_nav (
    member_cd   TEXT NOT NULL,
    fund_cd     TEXT NOT NULL,
    std_ymd     TEXT NOT NULL,
    nav         REAL NOT NULL,
    net_assets  REAL,
    change_pct  REAL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (member_cd, fund_cd, std_ymd)
);
CREATE INDEX IF NOT EXISTS idx_fund_nav_date ON fund_nav(std_ymd);
CREATE INDEX IF NOT EXISTS idx_fund_nav_fund ON fund_nav(member_cd, fund_cd);
"""


def get_conn(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.executescript(SCHEMA)
    return conn


def upsert_rows(
    conn: sqlite3.Connection,
    member_cd: str,
    fund_cd: str,
    rows: Iterable[NavRow],
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    sql = """
    INSERT INTO fund_nav (member_cd, fund_cd, std_ymd, nav, net_assets, change_pct, fetched_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(member_cd, fund_cd, std_ymd) DO UPDATE SET
        nav         = excluded.nav,
        net_assets  = excluded.net_assets,
        change_pct  = excluded.change_pct,
        fetched_at  = excluded.fetched_at
    """
    rows_list = list(rows)
    if not rows_list:
        return 0
    conn.executemany(
        sql,
        [
            (member_cd, fund_cd, r.std_ymd, r.nav, r.net_assets, r.change_pct, now)
            for r in rows_list
        ],
    )
    conn.commit()
    return len(rows_list)


def latest_date(conn: sqlite3.Connection, member_cd: str, fund_cd: str) -> Optional[date]:
    cur = conn.execute(
        "SELECT MAX(std_ymd) FROM fund_nav WHERE member_cd=? AND fund_cd=?",
        (member_cd, fund_cd),
    )
    row = cur.fetchone()
    if row and row[0]:
        return date.fromisoformat(row[0])
    return None
