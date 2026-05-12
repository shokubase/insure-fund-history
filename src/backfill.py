"""전체 기간 백필.

각 펀드에 대해 [start, end] 범위로 한 번에 호출. 응답이 8년치도 한 방에 되는
구조라 청크 분할은 일단 안 하지만, 만약 너무 큰 범위에서 서버가 잘라먹거나
타임아웃 나면 `--chunk-years` 로 쪼개기 옵션도 둠.
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import date, timedelta
from pathlib import Path

from .scraper import FundScraper
from .storage import get_conn, upsert_rows


def load_fund_list(csv_path: str | Path) -> list[dict]:
    funds: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # 빈 줄 / 주석 같은 거 스킵
            if not row.get("memberCd") or not row.get("fundCd"):
                continue
            funds.append(row)
    return funds


def _iter_chunks(start: date, end: date, chunk_years: int):
    if chunk_years <= 0:
        yield start, end
        return
    cur = start
    while cur <= end:
        chunk_end = min(cur.replace(year=cur.year + chunk_years) - timedelta(days=1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fund-list", default="fund_list.csv")
    ap.add_argument("--db", default="data/fund_history.db")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument(
        "--chunk-years", type=int, default=0,
        help="0=쪼개지 않음. 큰 범위에서 서버 응답이 잘릴 때 사용 (예: 2)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    funds = load_fund_list(args.fund_list)
    if not funds:
        raise SystemExit(f"No funds in {args.fund_list}")

    conn = get_conn(args.db)
    scraper = FundScraper()

    grand_total = 0
    for f in funds:
        label = f.get("name") or f["fundCd"]
        fund_total = 0
        for chunk_start, chunk_end in _iter_chunks(start, end, args.chunk_years):
            rows = scraper.fetch(f["memberCd"], f["fundCd"], chunk_start, chunk_end)
            n = upsert_rows(conn, f["memberCd"], f["fundCd"], rows)
            fund_total += n
        grand_total += fund_total
        print(f"[{label}] upserted {fund_total} rows ({start} ~ {end})")

    print(f"Done. Total {grand_total} rows across {len(funds)} fund(s).")


if __name__ == "__main__":
    main()
