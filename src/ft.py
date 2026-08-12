"""Financial Times 과거 시세 스크래퍼.

yfinance에 없는 해외 펀드(예: MS Global Opportunity, LU0552385535)를
markets.ft.com 에서 가져온다.

FT 티어시트 화면은 내부적으로 AJAX 엔드포인트를 호출하는데,
그 엔드포인트를 직접 부르면 로그인·쿠키·JS 렌더링 없이 requests 만으로
전체 이력을 받을 수 있다. Playwright 는 필요 없다.

    GET /data/equities/ajax/get-historical-prices
        ?startDate=YYYY/MM/DD&endDate=YYYY/MM/DD&symbol=<xid>

`symbol` 은 티커가 아니라 FT 내부 식별자(xid)다. 티어시트 HTML 의
data-mod-config 에서 뽑아낼 수 있고, 종목마다 고정이라 캐시해 둔다.

주의: 한 번에 16년치를 요청하면 JSON 대신 에러 HTML 이 돌아온다.
CHUNK_YEARS 단위로 쪼개서 요청한다.

사용 예:
    python -m src.ft LU0552385535:USD --start 2010-01-01
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

import requests

logger = logging.getLogger(__name__)

BASE = "https://markets.ft.com/data"
TEARSHEET_URL = f"{BASE}/funds/tearsheet/historical"
HISTORICAL_URL = f"{BASE}/equities/ajax/get-historical-prices"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# 요청 1건당 최대 기간. 너무 길면 FT 가 JSON 대신 에러 페이지를 준다.
CHUNK_YEARS = 5
TIMEOUT = 60

# fundCd → FT 심볼. xid 는 매 실행 시 티어시트에서 해석한다.
FT_SYMBOLS = {
    "MSGO": "LU0552385535:USD",  # MS INVF Global Opportunity Fund Z (USD)
}

_XID_RE = re.compile(r"(?:&quot;|\")xid(?:&quot;|\")\s*:\s*(?:&quot;|\")(\d+)")
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_DATE_RE = re.compile(r"hide-small-below\">([^<]+)<")
_NUM_RE = re.compile(r"<td>([\d.,-]+)</td>")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s


def resolve_xid(symbol: str, session: requests.Session | None = None) -> str | None:
    """티어시트 HTML 에서 FT 내부 식별자(xid)를 추출."""
    s = session or _session()
    r = s.get(TEARSHEET_URL, params={"s": symbol}, timeout=TIMEOUT)
    r.raise_for_status()
    m = _XID_RE.search(r.text)
    if not m:
        logger.warning("Could not resolve xid for %s", symbol)
        return None
    return m.group(1)


def _parse_rows(html: str) -> dict[str, float]:
    """AJAX 응답 HTML 조각에서 날짜→종가 dict 를 뽑는다.

    각 행은 Date / Open / High / Low / Close / Volume 순.
    펀드는 OHLC 가 모두 같은 값이지만 그래도 Close 를 쓴다.
    Volume 셀은 span 으로 감싸져 있어서 _NUM_RE 에 걸리지 않는다.
    """
    closes: dict[str, float] = {}
    for row in _ROW_RE.findall(html):
        dm = _DATE_RE.search(row)
        nums = _NUM_RE.findall(row)
        if not dm or len(nums) < 4:
            continue
        try:
            d = datetime.strptime(dm.group(1).strip(), "%A, %B %d, %Y").date()
            close = float(nums[3].replace(",", ""))
        except ValueError:
            continue
        if close > 0:
            closes[d.isoformat()] = close
    return closes


def _chunks(start: date, end: date):
    """[start, end] 를 CHUNK_YEARS 단위 구간으로 쪼갠다."""
    cur = start
    while cur <= end:
        try:
            nxt = cur.replace(year=cur.year + CHUNK_YEARS)
        except ValueError:  # 2/29
            nxt = cur.replace(year=cur.year + CHUNK_YEARS, day=28)
        chunk_end = min(nxt, end)
        yield cur, chunk_end
        if chunk_end >= end:
            break
        cur = chunk_end


def fetch_history(symbol: str, start: str, end: str,
                  session: requests.Session | None = None) -> dict[str, float]:
    """FT 에서 날짜→종가 dict 를 가져온다. 실패하면 빈 dict."""
    s = session or _session()
    xid = resolve_xid(symbol, s)
    if not xid:
        return {}

    s.headers["X-Requested-With"] = "XMLHttpRequest"
    s.headers["Referer"] = f"{TEARSHEET_URL}?s={symbol}"

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    closes: dict[str, float] = {}

    for a, b in _chunks(start_d, end_d):
        params = {
            "startDate": a.strftime("%Y/%m/%d"),
            "endDate": b.strftime("%Y/%m/%d"),
            "symbol": xid,
        }
        try:
            r = s.get(HISTORICAL_URL, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
        except (requests.RequestException, ValueError) as e:
            # JSON 이 아니면 FT 가 에러 페이지를 준 것 — 해당 청크만 건너뛴다.
            logger.warning("FT chunk %s~%s failed for %s: %s", a, b, symbol, e)
            continue
        chunk = _parse_rows(payload.get("html", ""))
        logger.info("FT %s %s~%s: %d rows", symbol, a, b, len(chunk))
        closes.update(chunk)

    return closes


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="FT 과거 시세 조회 (단독 실행용)")
    ap.add_argument("symbol", help="예: LU0552385535:USD")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    closes = fetch_history(args.symbol, args.start, args.end)
    if not closes:
        raise SystemExit("No data")
    keys = sorted(closes)
    print(f"{len(closes)} rows: {keys[0]} ~ {keys[-1]}")
    for d in keys[-5:]:
        print(f"  {d}  {closes[d]}")


if __name__ == "__main__":
    main()
