"""벤치마크 ETF/지수 데이터 yfinance로 가져오기.

yfinance에서 일별 종가를 받아 fund_nav 테이블에 member_cd='BENCH'로 저장.
기존 보험 펀드 데이터와 동일한 스키마로 대시보드에서 함께 분석 가능.

사용 예:
    python -m src.benchmarks
    python -m src.benchmarks --start 2010-01-01
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import yfinance as yf

from .storage import get_conn

logger = logging.getLogger(__name__)

MEMBER_CD = "BENCH"
BENCHMARK_LIST = Path("benchmark_list.csv")
DB_PATH = Path("data/fund_history.db")


def load_benchmark_list(csv_path: str | Path) -> list[dict]:
    benchmarks: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ticker") and row.get("fundCd"):
                benchmarks.append(row)
    return benchmarks


def fetch_and_store(conn, ticker: str, fund_cd: str, name: str,
                    start: str, end: str) -> int:
    """yfinance에서 데이터를 받아 DB에 upsert."""
    logger.info("Fetching %s (%s) ...", name, ticker)
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        logger.warning("No data for %s (%s)", name, ticker)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    prev_close = None
    for dt, row in df.iterrows():
        close = row["Close"]
        if close is None or close != close or close <= 0:  # NaN check: x != x
            continue
        std_ymd = dt.strftime("%Y-%m-%d")
        change_pct = ((close / prev_close - 1) * 100) if prev_close else None
        rows.append((MEMBER_CD, fund_cd, std_ymd, round(close, 4), None, round(change_pct, 4) if change_pct is not None else None, now))
        prev_close = close

    if not rows:
        return 0

    conn.executemany(
        """INSERT INTO fund_nav (member_cd, fund_cd, std_ymd, nav, net_assets, change_pct, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(member_cd, fund_cd, std_ymd) DO UPDATE SET
               nav = excluded.nav,
               change_pct = excluded.change_pct,
               fetched_at = excluded.fetched_at""",
        rows,
    )
    conn.commit()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="벤치마크 ETF/지수 데이터 가져오기")
    ap.add_argument("--benchmark-list", default=str(BENCHMARK_LIST))
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    benchmarks = load_benchmark_list(args.benchmark_list)
    if not benchmarks:
        raise SystemExit(f"No benchmarks in {args.benchmark_list}")

    conn = get_conn(args.db)
    total = 0
    for b in benchmarks:
        name = b.get("name") or b["ticker"]
        n = fetch_and_store(conn, b["ticker"], b["fundCd"], name, args.start, args.end)
        total += n
        print(f"[{name}] {n} rows")

    # Post-processing: fix known stock splits not handled by yfinance
    _fix_splits(conn)

    print(f"\nDone. Total {total} rows across {len(benchmarks)} benchmark(s).")


# Known splits and bad data points in yfinance
SPLIT_FIXES = [
    # (fundCd, split_date, divisor) — divide all data before split_date by divisor
    ("1329", "2014-01-06", 10),
    ("1656", "2017-09-28", 10),
]
BAD_DATA_FIXES = [
    # (fundCd, date, multiplier) — multiply nav by multiplier to fix
    ("1656", "2022-10-07", 10),
    ("1656", "2022-10-11", 10),
]
SPIKE_CLEANUP = [
    # (fundCd, max_pct) — delete 1-day spikes exceeding max_pct
    ("1322", 15),
]


def _fix_splits(conn):
    """Apply known split adjustments and data fixes after fetch."""
    for fund_cd, split_date, divisor in SPLIT_FIXES:
        cur = conn.execute(
            "SELECT MAX(nav) FROM fund_nav WHERE fund_cd=? AND std_ymd<?",
            (fund_cd, split_date),
        )
        max_nav = cur.fetchone()[0]
        if max_nav and max_nav > 500 * divisor:
            conn.execute(
                "UPDATE fund_nav SET nav = nav / ? WHERE fund_cd=? AND std_ymd<?",
                (divisor, fund_cd, split_date),
            )
            logger.info("Fixed split for %s before %s (÷%d)", fund_cd, split_date, divisor)

    for fund_cd, bad_date, multiplier in BAD_DATA_FIXES:
        cur = conn.execute(
            "SELECT nav FROM fund_nav WHERE fund_cd=? AND std_ymd=?",
            (fund_cd, bad_date),
        )
        row = cur.fetchone()
        if row and row[0] < 100:
            conn.execute(
                "UPDATE fund_nav SET nav = nav * ? WHERE fund_cd=? AND std_ymd=?",
                (multiplier, fund_cd, bad_date),
            )
            logger.info("Fixed bad data for %s on %s (×%d)", fund_cd, bad_date, multiplier)

    for fund_cd, max_pct in SPIKE_CLEANUP:
        threshold = max_pct / 100
        cur = conn.execute(f"""
            DELETE FROM fund_nav WHERE rowid IN (
                SELECT a.rowid FROM fund_nav a
                JOIN fund_nav b ON a.fund_cd = b.fund_cd
                    AND b.std_ymd = (SELECT MAX(std_ymd) FROM fund_nav WHERE fund_cd=? AND std_ymd < a.std_ymd)
                JOIN fund_nav c ON a.fund_cd = c.fund_cd
                    AND c.std_ymd = (SELECT MIN(std_ymd) FROM fund_nav WHERE fund_cd=? AND std_ymd > a.std_ymd)
                WHERE a.fund_cd = ?
                AND ABS(a.nav / b.nav - 1) > ?
                AND ABS(c.nav / a.nav - 1) > ?
            )
        """, (fund_cd, fund_cd, fund_cd, threshold, threshold))
        if cur.rowcount > 0:
            logger.info("Removed %d spike rows from %s", cur.rowcount, fund_cd)

    conn.commit()


if __name__ == "__main__":
    main()
