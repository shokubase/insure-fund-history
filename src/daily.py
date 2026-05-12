"""매일 증분 업데이트.

각 펀드의 DB 마지막 날짜+1 부터 오늘까지만 호출. cron/GH Actions에서 매일 실행.

DB가 비어 있으면 fallback start (2018-01-01)부터.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, timedelta
from pathlib import Path

from .scraper import FundScraper
from .storage import get_conn, latest_date, upsert_rows


FUND_LIST = Path("fund_list.csv")
DB_PATH = Path("data/fund_history.db")
FALLBACK_START = date(2018, 1, 1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    today = date.today()

    with open(FUND_LIST, newline="", encoding="utf-8") as f:
        funds = [
            row for row in csv.DictReader(f)
            if row.get("memberCd") and row.get("fundCd")
        ]

    if not funds:
        raise SystemExit(f"No funds in {FUND_LIST}")

    conn = get_conn(DB_PATH)
    scraper = FundScraper()

    total_new = 0
    for f in funds:
        label = f.get("name") or f["fundCd"]
        last = latest_date(conn, f["memberCd"], f["fundCd"])
        start = (last + timedelta(days=1)) if last else FALLBACK_START

        if start > today:
            print(f"[{label}] up to date (last={last})")
            continue

        rows = scraper.fetch(f["memberCd"], f["fundCd"], start, today)
        n = upsert_rows(conn, f["memberCd"], f["fundCd"], rows)
        total_new += n
        print(f"[{label}] +{n} rows ({start} ~ {today})")

    print(f"Total upserted: {total_new}")


if __name__ == "__main__":
    main()
