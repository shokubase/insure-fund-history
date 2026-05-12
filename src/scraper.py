"""펀드 기준가 스크래퍼

생명보험협회 공시실 `fundStdAmtAjax.do` 엔드포인트를 호출해서
펀드별 일별 기준가/순자산액/등락률 데이터를 받아온다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE = "https://pub.insure.or.kr"
POPUP_URL = f"{BASE}/compareDis/variableInsrn/fundDay/fundInfoViewPopup.do"
AJAX_URL = f"{BASE}/compareDis/variableInsrn/fundDay/fundStdAmtAjax.do"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

AJAX_HEADERS = {
    "AJAX": "true",
    "Accept": "text/html, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": BASE,
}


@dataclass
class NavRow:
    std_ymd: str            # 'YYYY-MM-DD'
    nav: float              # 기준가(원)
    net_assets: Optional[float]   # 순자산액(억원)
    change_pct: Optional[float]   # 기준가 등락률(%)

    def as_tuple(self) -> tuple:
        return (self.std_ymd, self.nav, self.net_assets, self.change_pct)


def _parse_number(s: str) -> Optional[float]:
    s = (s or "").strip().replace(",", "")
    if s in ("", "-", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_response_html(html: str) -> list[NavRow]:
    """응답 HTML에서 기준가 행을 파싱.

    응답이 전체 페이지인지 `<tbody>` 조각인지 아직 미확정이라
    `<tr>`를 전수 순회하면서 4컬럼 이상의 행만 데이터로 본다.

    Expected columns (in order, from the disclosure page):
        [0] 기준일 (YYYY-MM-DD or YYYY.MM.DD)
        [1] 순자산액(억원)
        [2] 기준가(원)
        [3] 기준가 등락률(%)

    실제 응답이 컬럼 순서가 다르거나 추가 컬럼이 있으면 이 함수만 손보면 됨.
    smoke.py로 원본 응답 한 번 떠보고 조정할 것.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[NavRow] = []
    for tr in soup.select("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 3:
            continue

        # 날짜 정규화: 2026-05-12 / 2026.05.12 / 2026/05/12 → 2026-05-12
        raw_ymd = cells[0].replace(".", "-").replace("/", "-")
        if len(raw_ymd) != 10 or raw_ymd[4] != "-" or raw_ymd[7] != "-":
            continue

        net_assets = _parse_number(cells[1]) if len(cells) > 1 else None
        nav = _parse_number(cells[2]) if len(cells) > 2 else None
        change_pct = _parse_number(cells[3]) if len(cells) > 3 else None

        if nav is None:
            # 기준가가 없으면 의미 있는 행이 아님
            continue

        rows.append(NavRow(
            std_ymd=raw_ymd,
            nav=nav,
            net_assets=net_assets,
            change_pct=change_pct,
        ))
    return rows


class FundScraper:
    """세션 유지 + 재시도 포함한 스크래퍼."""

    def __init__(self, sleep_between: float = 0.4, max_retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.sleep_between = sleep_between
        self.max_retries = max_retries
        self._warmed = False

    def warm_up(self, member_cd: str = "L34", fund_cd: str = "KLVL34001NQ") -> None:
        """세션 쿠키(JSESSIONID 등) 받기 위해 팝업 페이지 한 번 GET."""
        today = date.today().strftime("%Y%m%d")
        params = {"stdYmd": today, "memberCd": member_cd, "fundCd": fund_cd}
        r = self.session.get(POPUP_URL, params=params, timeout=15)
        r.raise_for_status()
        self._warmed = True
        logger.info(
            "Warmed up session. Cookies: %s",
            list(self.session.cookies.keys()),
        )

    def _post(self, member_cd: str, fund_cd: str, start: date, end: date) -> requests.Response:
        if not self._warmed:
            self.warm_up(member_cd, fund_cd)

        headers = {
            **AJAX_HEADERS,
            "Referer": (
                f"{POPUP_URL}?stdYmd={end.strftime('%Y%m%d')}"
                f"&memberCd={member_cd}&fundCd={fund_cd}"
            ),
        }
        data = {
            "memberCd": member_cd,
            "fundCd": fund_cd,
            "search_fundStdStartYmd": start.strftime("%Y-%m-%d"),
            "search_fundStdEndYmd": end.strftime("%Y-%m-%d"),
            "search_fundStdType": "D",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.session.post(AJAX_URL, data=data, headers=headers, timeout=30)
                r.raise_for_status()
                r.encoding = r.apparent_encoding or "utf-8"
                return r
            except requests.RequestException as e:
                last_exc = e
                wait = 2 ** attempt
                logger.warning(
                    "POST attempt %d/%d failed (%s); sleeping %ds",
                    attempt, self.max_retries, e, wait,
                )
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def fetch_raw(self, member_cd: str, fund_cd: str, start: date, end: date) -> str:
        r = self._post(member_cd, fund_cd, start, end)
        time.sleep(self.sleep_between)
        return r.text

    def fetch(self, member_cd: str, fund_cd: str, start: date, end: date) -> list[NavRow]:
        text = self.fetch_raw(member_cd, fund_cd, start, end)
        rows = parse_response_html(text)
        logger.info(
            "Fetched %d rows for %s/%s (%s ~ %s)",
            len(rows), member_cd, fund_cd, start, end,
        )
        return rows
