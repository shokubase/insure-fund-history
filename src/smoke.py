"""스모크 테스트: 최초 1회 응답 형식 확인용.

응답 본문을 파일로 저장하고 파싱 결과를 stdout에 찍어준다.
파싱이 0행이면 응답 HTML 구조가 가정과 다르다는 뜻이니
저장된 파일을 열어보거나 공유해서 parser를 조정하면 됨.

사용 예:
    python -m src.smoke --member-cd L34 --fund-cd KLVL34001NQ --days 30
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from .scraper import FundScraper, parse_response_html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-cd", default="L34")
    ap.add_argument("--fund-cd", default="KLVL34001NQ")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--save", default="smoke_response.html")
    args = ap.parse_args()

    scraper = FundScraper()
    end = date.today()
    start = end - timedelta(days=args.days)

    print(f"Fetching {args.member_cd}/{args.fund_cd} from {start} to {end} ...")
    raw = scraper.fetch_raw(args.member_cd, args.fund_cd, start, end)

    Path(args.save).write_text(raw, encoding="utf-8")
    print(f"Saved raw response → {args.save} ({len(raw)} chars)")
    print()
    print("First 500 chars of response:")
    print("-" * 60)
    print(raw[:500])
    print("-" * 60)
    print()

    rows = parse_response_html(raw)
    print(f"Parsed {len(rows)} rows")
    for r in rows[:5]:
        print(f"  {r}")
    if len(rows) > 5:
        print(f"  ... and {len(rows) - 5} more")

    if len(rows) == 0:
        print()
        print("WARNING: 0 rows parsed. 응답 형식이 예상과 다를 수 있음.")
        print(f"         {args.save} 파일 열어서 HTML 구조 확인 필요.")


if __name__ == "__main__":
    main()
