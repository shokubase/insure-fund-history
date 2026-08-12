"""펀드 분석 대시보드 생성.

SQLite에 저장된 기준가 데이터를 분석하여 단일 HTML 리포트 파일을 생성한다.

사용 예:
    python -m src.dashboard
    python -m src.dashboard --output report.html --risk-free 4.0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import date, datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from .storage import get_conn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fund_list(csv_path: str | Path) -> list[dict]:
    funds: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("memberCd") and row.get("fundCd"):
                funds.append(row)
    return funds


def load_nav_series(conn, member_cd: str, fund_cd: str) -> pd.Series:
    """DB에서 NAV 시계열을 pandas Series로 반환 (DatetimeIndex, float values)."""
    df = pd.read_sql_query(
        "SELECT std_ymd, nav FROM fund_nav WHERE member_cd=? AND fund_cd=? ORDER BY std_ymd",
        conn,
        params=(member_cd, fund_cd),
    )
    df["std_ymd"] = pd.to_datetime(df["std_ymd"])
    df = df.set_index("std_ymd")["nav"]
    return df


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------

def compute_basic_metrics(nav: pd.Series, risk_free: float) -> dict:
    first_date, last_date = nav.index[0], nav.index[-1]
    total_years = (last_date - first_date).days / 365.25
    if total_years <= 0:
        return {}

    nav_first, nav_last = nav.iloc[0], nav.iloc[-1]
    cagr = (nav_last / nav_first) ** (1 / total_years) - 1

    # 월간 기준. 보험펀드 기준가는 기초자산을 1~3일 늦게, 그것도 평활된 형태로
    # 반영해서 일간 변동성이 실제보다 눌린다 (N1M0: 일간 15.7% vs 월간 21.0%).
    # 월간으로 집계하면 그 잡음이 씻겨나가고, 지수·ETF 와도 같은 잣대가 된다.
    monthly_returns = nav.resample("ME").last().dropna().pct_change().dropna()
    if len(monthly_returns) >= 6:
        vol = monthly_returns.std() * sqrt(12)
    else:
        daily_returns = nav.pct_change().dropna()
        vol = daily_returns.std() * sqrt(len(daily_returns) / total_years)

    sharpe = (cagr - risk_free) / vol if vol > 0 else 0.0

    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    mdd = drawdown.min()

    return {
        "first_date": first_date.strftime("%Y-%m-%d"),
        "last_date": last_date.strftime("%Y-%m-%d"),
        "total_years": round(total_years, 1),
        "total_return": round((nav_last / nav_first - 1) * 100, 2),
        "cagr": round(cagr * 100, 2),
        "volatility": round(vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd * 100, 2),
    }


# ---------------------------------------------------------------------------
# Drawdown events
# ---------------------------------------------------------------------------

def find_drawdown_events(nav: pd.Series) -> list[dict]:
    cummax = nav.cummax()
    dd = (nav - cummax) / cummax

    events: list[dict] = []
    in_dd = False
    start = trough_date = None
    trough_val = 0.0

    for dt, val in dd.items():
        if not in_dd and val < 0:
            in_dd = True
            start = dt
            trough_date = dt
            trough_val = val
        elif in_dd:
            if val < trough_val:
                trough_date = dt
                trough_val = val
            if val >= 0:
                in_dd = False
                events.append({
                    "start": start.strftime("%Y-%m-%d"),
                    "trough": trough_date.strftime("%Y-%m-%d"),
                    "end": dt.strftime("%Y-%m-%d"),
                    "depth": round(trough_val * 100, 2),
                    "duration_days": (dt - start).days,
                })

    if in_dd and start is not None:
        events.append({
            "start": start.strftime("%Y-%m-%d"),
            "trough": trough_date.strftime("%Y-%m-%d"),
            "end": None,
            "depth": round(trough_val * 100, 2),
            "duration_days": (nav.index[-1] - start).days,
        })

    events.sort(key=lambda e: e["depth"])
    return events


def drawdown_summary(events: list[dict]) -> dict:
    if not events:
        return {"avg_drawdown": 0, "longest_days": 0, "longest_start": None, "longest_end": None}

    avg_dd = round(np.mean([abs(e["depth"]) for e in events]), 2)
    longest = max(events, key=lambda e: e["duration_days"])
    return {
        "avg_drawdown": avg_dd,
        "longest_days": longest["duration_days"],
        "longest_start": longest["start"],
        "longest_end": longest["end"] or "진행중",
    }


# ---------------------------------------------------------------------------
# LS vs DCA
# ---------------------------------------------------------------------------

def compute_ls_vs_dca(nav: pd.Series, window_months: int) -> dict | None:
    monthly = nav.resample("ME").last().dropna()
    n = len(monthly)
    if n <= window_months:
        return None

    nav_vals = monthly.values
    ls_returns = []
    dca_returns = []

    for i in range(n - window_months):
        end_nav = nav_vals[i + window_months]
        r_ls = end_nav / nav_vals[i] - 1
        r_dca = np.mean([end_nav / nav_vals[i + k] for k in range(window_months)]) - 1
        ls_returns.append(r_ls)
        dca_returns.append(r_dca)

    ls_arr = np.array(ls_returns)
    dca_arr = np.array(dca_returns)
    advantage = ls_arr - dca_arr

    ls_wins = advantage > 0
    win_rate = ls_wins.mean() * 100
    mlsa = advantage.mean() * 100
    losses = advantage[~ls_wins]
    mlsd = losses.mean() * 100 if len(losses) > 0 else 0.0

    return {
        "window": window_months,
        "observations": len(advantage),
        "win_rate": round(win_rate, 1),
        "mlsa": round(mlsa, 2),
        "mlsd": round(mlsd, 2),
    }


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def compute_correlation_matrix(conn, funds: list[dict]) -> dict | None:
    """펀드 간 일별 수익률 상관행렬 계산. 공통 날짜 기준."""
    if len(funds) < 2:
        return None

    series = {}
    for f in funds:
        label = f.get("fundCd") or f.get("name", "")
        nav = load_nav_series(conn, f["memberCd"], f["fundCd"])
        series[label] = nav.pct_change().dropna()

    df = pd.DataFrame(series).dropna()
    if len(df) < 30:
        return None

    corr = df.corr()
    names = list(corr.columns)
    matrix = [[round(corr.iloc[i, j], 4) for j in range(len(names))] for i in range(len(names))]
    return {"names": names, "matrix": matrix, "obs": len(df)}


# ---------------------------------------------------------------------------
# Analyze one fund
# ---------------------------------------------------------------------------

def _build_series_data(nav: pd.Series, risk_free: float, top_n: int) -> dict:
    """NAV 시계열에서 모든 분석 데이터를 생성."""
    basic = compute_basic_metrics(nav, risk_free)
    events = find_drawdown_events(nav)
    dd_summary_data = drawdown_summary(events)
    top_events = events[:top_n]

    ls_dca = []
    for w in [3, 12, 36]:
        result = compute_ls_vs_dca(nav, w)
        if result:
            ls_dca.append(result)

    step = max(1, len(nav) // 500)
    chart_nav = nav.iloc[::step]
    cummax = nav.cummax()
    dd_series = ((nav - cummax) / cummax).iloc[::step]

    chart_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in chart_nav.index],
        "nav": [round(v, 2) for v in chart_nav.values],
        "drawdown": [round(v * 100, 2) for v in dd_series.values],
    }

    daily_returns = nav.pct_change().dropna()
    daily_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in daily_returns.index],
        "returns": [round(v, 8) for v in daily_returns.values],
    }

    monthly_nav = nav.resample("ME").last().dropna()
    monthly_returns = monthly_nav.pct_change().dropna()
    monthly_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in monthly_returns.index],
        "returns": [round(v, 8) for v in monthly_returns.values],
    }

    return {
        "basic": basic,
        "dd_summary": dd_summary_data,
        "top_events": top_events,
        "ls_dca": ls_dca,
        "chart": chart_data,
        "daily": daily_data,
        "monthly": monthly_data,
    }


def analyze_fund(conn, member_cd: str, fund_cd: str, name: str,
                 risk_free: float, top_n: int,
                 krw_nav: pd.Series | None = None) -> dict | None:
    nav = load_nav_series(conn, member_cd, fund_cd)
    if len(nav) < 30:
        logger.warning("Skipping %s: only %d data points", fund_cd, len(nav))
        return None

    result = _build_series_data(nav, risk_free, top_n)
    result.update({"name": name, "member_cd": member_cd, "fund_cd": fund_cd,
                   "has_krw": krw_nav is not None})

    if krw_nav is not None and len(krw_nav) >= 30:
        result["krw"] = _build_series_data(krw_nav, risk_free, top_n)

    return result


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>펀드 분석 대시보드</title>
<script>
// Resolve the theme before first paint so the page never flashes light-on-dark.
(function () {
  var stored = null;
  try { stored = localStorage.getItem('fund_dashboard_theme'); } catch (e) {}
  var dark = stored ? stored === 'dark'
                    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
})();
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
<style>
  :root {
    /* Cooled and stepped down a notch so the cards sit in the same family as the
       side panel. Every surface moved together, so cards stay lighter than the page. */
    --bg: #e5e8f0;
    --card: #f3f5f9;
    --surface: #f3f5f9;
    --surface-2: #edeff5;
    --surface-3: #e0e4ed;
    --border: #d7dce6;
    --border-strong: #c0c7d6;
    --text: #12161f;
    /* Darker cards cost ~8% contrast, so the secondary inks step down to match:
       muted clears AA again (4.43 -> 4.91) and subtle beats its old value on white. */
    --muted: #646b79;
    --subtle: #848c9c;
    --accent: #2b57d2;
    --accent-hover: #2246ac;
    --accent-soft: #e4ebff;
    --red: #cf2a1e;
    --green: #088750;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow-xs: 0 1px 2px rgba(16,24,40,.05);
    --shadow-sm: 0 1px 3px rgba(16,24,40,.07), 0 1px 2px rgba(16,24,40,.04);
    --shadow-md: 0 6px 20px rgba(16,24,40,.09);
    --sidebar-w: 250px;
    /* The side panel keeps its own dark scale in both themes, so these are deliberately
       not tied to --surface/--text. */
    --side-bg: #171b26;
    --side-line: rgba(255,255,255,.09);
    --side-text: #e8eaf0;
    --side-muted: #9aa3b5;
    --side-subtle: #6f7789;
    --side-hover: rgba(255,255,255,.07);
    --side-active-bg: rgba(91,134,245,.20);
    --side-active-text: #93b0ff;
    --taa-band: rgba(207,42,30,.07);
    color-scheme: light;
  }
  :root[data-theme="dark"] {
    --bg: #0b0d12;
    --card: #14171f;
    --surface: #14171f;
    --surface-2: #191d25;
    --surface-3: #222731;
    --border: #242934;
    --border-strong: #3b4251;
    --text: #e6e9ef;
    --muted: #99a2b2;
    --subtle: #6e778a;
    --accent: #5b86f5;
    --accent-hover: #7599f7;
    --accent-soft: #1a2440;
    --red: #f97066;
    --green: #47cd89;
    --shadow-xs: 0 1px 2px rgba(0,0,0,.35);
    --shadow-sm: 0 1px 3px rgba(0,0,0,.4), 0 1px 2px rgba(0,0,0,.3);
    --shadow-md: 0 6px 20px rgba(0,0,0,.45);
    --side-bg: #101318;
    --taa-band: rgba(249,112,102,.10);
    color-scheme: dark;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Pretendard', Roboto, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.55;
         -webkit-font-smoothing: antialiased; font-size: 15px; }

  /* ── App shell ─────────────────────────────────────────── */
  .sidebar { position: fixed; top: 0; left: 0; bottom: 0; width: var(--sidebar-w); z-index: 50;
             background: var(--side-bg); border-right: 1px solid var(--side-line);
             display: flex; flex-direction: column; padding: 1.1rem .85rem 1rem; gap: 1.1rem; }
  .brand { display: flex; align-items: center; gap: .6rem; padding: .1rem .45rem .95rem;
           border-bottom: 1px solid var(--side-line); }
  .brand-mark { width: 32px; height: 32px; flex: none; border-radius: 9px; display: grid; place-items: center;
                background: linear-gradient(135deg, var(--accent), #7b52d3); color: #fff;
                font-size: .8rem; font-weight: 700; letter-spacing: -.02em; }
  .brand-text strong { display: block; font-size: .93rem; font-weight: 650; letter-spacing: -.015em;
                       color: var(--side-text); }
  .brand-text span { display: block; font-size: .7rem; color: var(--side-subtle); letter-spacing: .02em; }

  .side-nav { display: flex; flex-direction: column; gap: .15rem; }
  .nav-item { display: flex; align-items: center; gap: .6rem; width: 100%; padding: .58rem .65rem;
              border: none; background: none; border-radius: var(--radius-sm); font: inherit; font-size: .875rem;
              color: var(--side-muted); cursor: pointer; text-align: left; transition: background .12s, color .12s; }
  .nav-item:hover { background: var(--side-hover); color: var(--side-text); }
  .nav-item.active { background: var(--side-active-bg); color: var(--side-active-text); font-weight: 600; }
  .nav-item svg { width: 17px; height: 17px; flex: none; }
  .nav-badge { margin-left: auto; min-width: 21px; height: 19px; padding: 0 .38rem; border-radius: 10px;
               background: var(--side-hover); color: var(--side-muted); font-size: .69rem; font-weight: 600;
               display: none; align-items: center; justify-content: center; font-variant-numeric: tabular-nums; }
  .nav-badge.show { display: inline-flex; }
  .nav-item.active .nav-badge { background: var(--side-active-text); color: var(--side-bg); }
  .side-bottom { margin-top: auto; }
  .theme-toggle { display: flex; align-items: center; gap: .55rem; width: 100%; padding: .5rem .65rem;
                  border: 1px solid var(--side-line); background: var(--side-hover); border-radius: var(--radius-sm);
                  font: inherit; font-size: .8rem; color: var(--side-muted); cursor: pointer;
                  transition: background .12s, color .12s, border-color .12s; }
  .theme-toggle:hover { color: var(--side-active-text); border-color: var(--side-active-text);
                        background: var(--side-active-bg); }
  .theme-toggle svg { width: 16px; height: 16px; flex: none; }
  :root[data-theme="dark"] .icon-moon, :root:not([data-theme="dark"]) .icon-sun { display: none; }
  .side-foot { padding: .85rem .6rem 0; margin-top: .8rem; border-top: 1px solid var(--side-line);
               font-size: .72rem; color: var(--side-subtle); line-height: 1.85; }
  .side-foot b { color: var(--side-muted); font-weight: 600; }

  .main { margin-left: var(--sidebar-w); padding: 1.8rem 2rem 4rem; }
  .main-inner { max-width: 1280px; margin: 0 auto; }
  .page-head { margin-bottom: 1.35rem; }
  .page-head h1 { font-size: 1.4rem; font-weight: 650; letter-spacing: -.02em; }
  .page-head p { font-size: .855rem; color: var(--muted); margin-top: .22rem; }

  .tab-panel { display: none; }
  .tab-panel.active { display: block; animation: panelIn .18s ease both; }
  @keyframes panelIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

  .empty { padding: 3rem 1.25rem; text-align: center; font-size: .875rem; color: var(--subtle);
           background: var(--surface); border: 1px dashed var(--border-strong); border-radius: var(--radius); }

  @media (max-width: 900px) {
    .sidebar { position: sticky; top: 0; bottom: auto; width: auto; flex-direction: row; align-items: center;
               gap: .7rem; padding: .55rem .8rem; border-right: none; border-bottom: 1px solid var(--border);
               overflow-x: auto; }
    .brand { flex: none; border-bottom: none; border-right: 1px solid var(--border);
             padding: 0 .7rem 0 .2rem; white-space: nowrap; }
    .brand-text span { display: none; }
    .side-nav { flex: none; flex-direction: row; gap: .25rem; }
    .nav-item { white-space: nowrap; padding: .42rem .7rem; }
    .side-bottom { flex: none; margin-top: 0; margin-left: auto; padding-left: .5rem; }
    .theme-toggle { padding: .4rem .55rem; }
    .theme-toggle .theme-label, .side-foot { display: none; }
    .main { margin-left: 0; padding: 1.2rem 1rem 3rem; }
  }

  /* ── Cards ─────────────────────────────────────────────── */
  .card, .fund-section, .asset-filter {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.4rem 1.5rem; margin-bottom: 1.15rem; box-shadow: var(--shadow-xs); }
  .card > h2, .fund-section > h2, .asset-filter > h2 {
    font-size: 1.02rem; font-weight: 650; letter-spacing: -.012em; margin-bottom: .1rem; color: var(--text); }
  .card-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem;
               flex-wrap: wrap; margin-bottom: 1rem; }
  .card-head h2 { margin-bottom: 0; }
  .fund-meta { font-size: .8rem; color: var(--subtle); margin-bottom: 1.25rem; }
  .hint { font-size: .74rem; color: var(--subtle); margin: -.6rem 0 1rem; }
  h3 { font-size: .95rem; font-weight: 650; letter-spacing: -.01em; margin: 1.6rem 0 .75rem;
       padding-bottom: .4rem; border-bottom: 1px solid var(--border); }

  .metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
                  gap: .6rem; margin-bottom: 1.4rem; }
  @media (max-width: 900px) { .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  .metric-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-sm);
                 padding: .7rem .85rem; text-align: center; }
  .metric-card .label { font-size: .7rem; color: var(--subtle); letter-spacing: .01em; }
  .metric-card .value { font-size: 1.2rem; font-weight: 680; letter-spacing: -.02em; margin-top: .15rem;
                        font-variant-numeric: tabular-nums; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }

  .chart-container { position: relative; height: 300px; margin-bottom: 1.4rem; }
  /* Drag-to-select region + its readout, shared by every NAV chart.
     top/height are set from the chart's plot area at drag time so the band never spills
     over the axis gutters. */
  .drag-overlay { display: none; position: absolute; top: 0; height: 0; pointer-events: none;
                  background: color-mix(in srgb, var(--accent) 14%, transparent);
                  border-left: 1px dashed var(--accent); border-right: 1px dashed var(--accent); }
  .drag-stats { display: none; position: absolute; top: 8px; right: 8px; z-index: 10;
                background: var(--surface); color: var(--text); border: 1px solid var(--border);
                border-radius: 8px; padding: .5rem .8rem; font-size: .78rem; line-height: 1.5;
                box-shadow: var(--shadow-md); }
  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.4rem; }
  @media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }

  table { width: 100%; border-collapse: collapse; font-size: .83rem; margin-bottom: 1rem;
          font-variant-numeric: tabular-nums; }
  th, td { padding: .5rem .7rem; text-align: right; border-bottom: 1px solid var(--border); }
  th { background: var(--surface-2); font-weight: 600; font-size: .75rem; color: var(--muted);
       text-align: right; white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  tr:hover td { background: var(--accent-soft); }
  .ongoing { color: var(--red); font-style: italic; }

  /* ── Asset filter / selector ───────────────────────────── */
  .filter-chips { display: flex; flex-wrap: wrap; gap: .4rem; }
  .filter-chip { display: inline-flex; align-items: center; gap: .35rem; padding: .32rem .7rem;
                 border: 1px solid var(--border-strong); border-radius: 999px; font-size: .8rem;
                 cursor: pointer; transition: background .12s, border-color .12s, color .12s;
                 user-select: none; background: var(--surface); color: var(--muted); }
  .filter-chip:hover { border-color: var(--accent); color: var(--accent); }
  .filter-chip.active { background: var(--accent); color: #fff; border-color: var(--accent);
                        box-shadow: var(--shadow-xs); }
  .filter-chip input { display: none; }
  /* Keep the per-asset currency toggle quiet until the asset itself is picked. */
  .filter-chip:not(.active) .currency-toggle { opacity: .5; }
  .filter-chip:not(.active):hover .currency-toggle { opacity: 1; }
  .filter-chip.active .btn-currency { background: rgba(255,255,255,.18); color: #fff;
                                      border-color: rgba(255,255,255,.4); }
  .filter-chip.active .btn-currency.active { background: #fff; color: var(--accent); }
  .filter-actions { display: flex; gap: .4rem; }
  .filter-actions button { background: var(--surface); border: 1px solid var(--border-strong);
                           border-radius: var(--radius-sm); padding: .32rem .75rem; font: inherit;
                           font-size: .78rem; cursor: pointer; color: var(--muted); transition: all .12s; }
  .filter-actions button:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
  .fund-section.hidden { display: none; }

  /* ── Period selector ───────────────────────────────────── */
  .period-row { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin-top: .55rem; }
  .period-row:first-of-type { margin-top: 0; }
  .period-label { font-size: .7rem; color: var(--subtle); font-weight: 600; letter-spacing: .06em;
                  text-transform: uppercase; width: 5.2rem; flex: none; }
  .period-row label { font-size: .8rem; color: var(--muted); display: inline-flex;
                      align-items: center; gap: .35rem; }
  .period-chip { background: var(--surface); border: 1px solid var(--border-strong); border-radius: 999px;
                 padding: .28rem .72rem; font: inherit; font-size: .78rem; cursor: pointer;
                 color: var(--muted); transition: all .12s; font-variant-numeric: tabular-nums; }
  .period-chip:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
  .period-chip.active { background: var(--accent); color: #fff; border-color: var(--accent);
                        box-shadow: var(--shadow-xs); }
  .period-info { font-size: .78rem; color: var(--subtle); font-variant-numeric: tabular-nums; }
  .row-note { display: block; font-size: .68rem; color: var(--subtle); font-weight: 400; }
  .track-note { font-size: .78rem; color: var(--subtle); margin: -.4rem 0 1rem; }
  .track-ok { color: var(--green); font-weight: 600; }
  .track-bad { color: var(--red); font-weight: 600; }
  /* Inline variant — the portfolio picker lives inside the composition card. */
  .period-block { margin: 0 0 1rem; padding: .8rem 1rem .9rem; background: var(--surface-2);
                  border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .period-head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem;
                 flex-wrap: wrap; margin-bottom: .7rem; }
  .period-head h4 { font-size: .84rem; font-weight: 650; color: var(--text); }
  /* Lift the chips off the block's tinted backdrop — but never over .active. */
  .period-block .period-chip:not(.active) { background: var(--surface); }
  @media (max-width: 760px) { .period-label { width: auto; } }

  .selector-columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.4rem; }
  .selector-column h4 { font-size: .72rem; color: var(--subtle); font-weight: 600; letter-spacing: .06em;
                        text-transform: uppercase; margin-bottom: .5rem; padding-bottom: .32rem;
                        border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: .35rem; }
  .selector-column .filter-chips { margin-bottom: 0; }
  .col-ccy-toggle { display: inline-flex; vertical-align: middle; margin-left: auto; }
  .col-ccy-toggle .btn-currency { padding: .05rem .35rem; font-size: .65rem; }
  @media (max-width: 760px) { .selector-columns { grid-template-columns: 1fr; } }

  /* ── Currency toggle ───────────────────────────────────── */
  .currency-toggle { display: inline-flex; gap: 0; }
  .btn-currency { background: var(--surface-2); border: 1px solid var(--border-strong); padding: .3rem .8rem;
                  font: inherit; font-size: .78rem; cursor: pointer; color: var(--muted); transition: all .12s; }
  .btn-currency:first-child { border-radius: 6px 0 0 6px; }
  .btn-currency:last-child { border-radius: 0 6px 6px 0; border-left: none; }
  .btn-currency:hover { color: var(--accent); }
  .btn-currency.active { background: var(--accent); color: #fff; border-color: var(--accent); }

  /* ── Correlation matrix ────────────────────────────────── */
  .corr-table { width: auto; margin: 0 auto 1rem; }
  .corr-table th, .corr-table td { text-align: center; min-width: 78px; padding: .5rem; font-size: .8rem; }
  .corr-table th { background: var(--surface-2); font-size: .73rem; max-width: 120px;
                   overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .corr-table tr:hover td { background: inherit; }

  /* ── Portfolio analyzer ────────────────────────────────── */
  .portfolio-controls { margin-bottom: 1.4rem; }
  .fund-row { display: flex; align-items: center; gap: .6rem; padding: .38rem 0;
              border-bottom: 1px solid var(--border); }
  .fund-row:last-child { border-bottom: none; }
  .fund-row label { flex: 1; min-width: 0; cursor: pointer; display: flex; align-items: center;
                    gap: .45rem; font-size: .82rem; color: var(--muted); }
  .fund-row label span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .fund-row input[type=checkbox] { accent-color: var(--accent); width: 15px; height: 15px; flex: none; }
  .fund-row input[type=number] { width: 62px; padding: .25rem .45rem; border: 1px solid var(--border-strong);
                                 border-radius: 6px; text-align: right; font: inherit; font-size: .82rem;
                                 background: var(--surface); color: var(--text); }
  .fund-row input[type=number]:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
  /* A pinned weight is one the user typed — auto rows split whatever is left over. */
  .fund-row input[type=number].pinned { border-color: var(--accent); color: var(--accent); font-weight: 600; }
  .pf-weight-hint { font-size: .74rem; color: var(--subtle); }
  input[type=date] { padding: .28rem .45rem; border: 1px solid var(--border-strong); border-radius: 6px;
                     font: inherit; font-size: .8rem; background: var(--surface); color: var(--text); }
  .toolbar { display: flex; align-items: center; gap: .9rem; flex-wrap: wrap;
             margin: 1rem 0; padding: .7rem .9rem; background: var(--surface-2);
             border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .toolbar .sep { color: var(--border-strong); }
  .weight-sum { font-size: .84rem; font-weight: 600; color: var(--muted); }
  .weight-sum.warn { color: var(--red); }
  .btn-analyze { background: var(--accent); color: #fff; border: none; border-radius: var(--radius-sm);
                 padding: .55rem 1.35rem; font: inherit; font-size: .88rem; font-weight: 600; cursor: pointer;
                 transition: background .12s; box-shadow: var(--shadow-xs); }
  .btn-analyze:hover:not(:disabled) { background: var(--accent-hover); }
  .btn-analyze:disabled { background: var(--surface-3); color: var(--subtle); cursor: not-allowed; box-shadow: none; }
  .btn-ghost { background: var(--surface); color: var(--muted); border: 1px solid var(--border-strong);
               box-shadow: none; font-weight: 500; }
  .btn-ghost:hover:not(:disabled) { background: var(--surface-2); color: var(--accent); border-color: var(--accent); }
  .preset-bar { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; margin-top: .7rem; }
  .preset-bar > span { font-size: .78rem; color: var(--subtle); }
  #portfolio-results { margin-top: 1.4rem; }
  .preset-chip { position: relative; padding-right: 1.55rem !important; }
  .preset-chip .preset-del { position: absolute; right: .35rem; top: 50%; transform: translateY(-50%);
                             font-size: .72rem; color: var(--subtle); cursor: pointer; line-height: 1; }
  .preset-chip .preset-del:hover { color: var(--red); }

  /* ── 추세 신호 ─────────────────────────────────────────── */
  .sig-badge { display: inline-flex; align-items: center; gap: .3rem; padding: .12rem .5rem;
               border-radius: 999px; font-size: .74rem; font-weight: 650; letter-spacing: -.01em;
               border: 1px solid transparent; white-space: nowrap; }
  .sig-badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .sig-hold { color: var(--green); background: color-mix(in srgb, var(--green) 12%, transparent);
              border-color: color-mix(in srgb, var(--green) 30%, transparent); }
  .sig-cash { color: var(--red); background: color-mix(in srgb, var(--red) 12%, transparent);
              border-color: color-mix(in srgb, var(--red) 30%, transparent); }
  /* 가격은 위, 기울기는 아래 — 규칙상 보유지만 추세는 이미 꺾인 상태. */
  .sig-warn { color: #b7791f; background: rgba(183,121,31,.12); border-color: rgba(183,121,31,.32); }
  :root[data-theme="dark"] .sig-warn { color: #e3b341; background: rgba(227,179,65,.13);
                                       border-color: rgba(227,179,65,.3); }

  .taa-asset { margin-bottom: 1.1rem; }
  .taa-head { display: flex; align-items: baseline; gap: .6rem; flex-wrap: wrap; margin-bottom: .2rem; }
  .taa-head h3 { font-size: 1rem; font-weight: 650; letter-spacing: -.015em; }
  .taa-facts { display: flex; flex-wrap: wrap; gap: .35rem .5rem; margin: .55rem 0 .8rem; }
  .taa-fact { flex: 1 1 116px; padding: .45rem .6rem; border-radius: var(--radius-sm);
              background: var(--surface-2); border: 1px solid var(--border); }
  .taa-fact span { display: block; font-size: .69rem; color: var(--subtle); margin-bottom: .1rem; }
  .taa-fact b { font-size: .92rem; font-weight: 640; font-variant-numeric: tabular-nums; }
  .taa-sub { font-size: .78rem; color: var(--muted); margin: .35rem 0 .1rem; }
</style>
</head>
<body>
<aside class="sidebar">
  <div class="brand">
    <div class="brand-mark">FA</div>
    <div class="brand-text"><strong>펀드 분석</strong><span>Fund Analytics</span></div>
  </div>
  <nav class="side-nav" id="side-nav">
    <button class="nav-item active" data-tab="panel-compare">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/></svg>
      <span>자산 비교</span><span class="nav-badge" id="badge-compare">0</span>
    </button>
    <button class="nav-item" data-tab="panel-taa">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 3 4-6 5 4"/><path d="M3 21h18"/><path d="M3 8c4 3 8 1 12-2"/></svg>
      <span>추세 신호</span><span class="nav-badge" id="badge-taa">0</span>
    </button>
    <button class="nav-item" data-tab="panel-detail">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>
      <span>개별 자산 상세</span><span class="nav-badge" id="badge-detail">0</span>
    </button>
    <button class="nav-item" data-tab="panel-portfolio">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a9 9 0 1 0 9 9h-9V3z"/><path d="M15.5 3.6A9 9 0 0 1 20.4 8.5H15.5V3.6z"/></svg>
      <span>포트폴리오 분석</span><span class="nav-badge" id="badge-portfolio">0</span>
    </button>
  </nav>
  <div class="side-bottom">
    <button class="theme-toggle" id="theme-toggle" type="button" aria-label="테마 전환">
      <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
      <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
      <span class="theme-label" id="theme-label">다크 모드</span>
    </button>
    <div class="side-foot">
      <div>생성 <b>%%GENERATED_AT%%</b></div>
      <div>무위험수익률 <b>%%RISK_FREE%%%</b></div>
    </div>
  </div>
</aside>

<main class="main"><div class="main-inner">

<!-- ══ Tab: 자산 비교 ══════════════════════════════════════ -->
<section class="tab-panel active" id="panel-compare">
  <header class="page-head">
    <h1>자산 비교</h1>
    <p>자산을 선택하면 시작점을 100으로 정규화해 성과·위험을 봅니다. 2개 이상이면 공통 기간에서 상관관계까지 비교합니다.</p>
  </header>

  <div class="asset-filter">
    <div class="card-head">
      <h2>자산 선택</h2>
      <div class="filter-actions">
        <button id="filter-all">전체 선택</button>
        <button id="filter-none">전체 해제</button>
      </div>
    </div>
    <div class="selector-columns">
      <div class="selector-column"><h4>보험펀드 <span class="col-ccy-toggle" data-col="filter-insurance" data-modes="orig,jpy"></span></h4><div class="filter-chips" id="filter-chips-insurance"></div></div>
      <div class="selector-column"><h4>미국 <span class="col-ccy-toggle" data-col="filter-us" data-modes="orig,krw,jpy"></span></h4><div class="filter-chips" id="filter-chips-us"></div></div>
      <div class="selector-column"><h4>일본 <span class="col-ccy-toggle" data-col="filter-jp" data-modes="orig,krw"></span></h4><div class="filter-chips" id="filter-chips-jp"></div><h4 style="margin-top:1rem;">지수 <span class="col-ccy-toggle" data-col="filter-index" data-modes="orig,krw,jpy"></span></h4><div class="filter-chips" id="filter-chips-index"></div></div>
    </div>
  </div>

  <div class="asset-filter" id="compare-period" style="display:none;">
    <div class="card-head">
      <h2>기간 선택</h2>
      <span class="period-info" id="comp-period-info"></span>
    </div>
    <div class="period-row">
      <span class="period-label">빠른 선택</span>
      <div class="filter-chips" id="comp-period-presets"></div>
    </div>
    <div class="period-row">
      <span class="period-label">연도</span>
      <div class="filter-chips" id="comp-period-years"></div>
    </div>
    <div class="period-row">
      <span class="period-label">직접 입력</span>
      <label>시작일 <input type="date" id="comp-start"></label>
      <label>종료일 <input type="date" id="comp-end"></label>
    </div>
  </div>

  <section class="card" id="comparison-section" style="display:none;">
    <div class="card-head">
      <h2>성과 비교</h2>
      <div class="period-row" style="margin:0;">
        <span class="period-label" style="width:auto;">보수</span>
        <div class="filter-chips" id="fee-mode">
          <button class="period-chip active" type="button" data-fee="net">차감 후</button>
          <button class="period-chip" type="button" data-fee="gross">차감 전</button>
          <button class="period-chip" type="button" data-fee="both">둘 다</button>
        </div>
      </div>
    </div>
    <p class="fund-meta" id="comparison-meta"></p>
    <div class="chart-container" style="height:400px;position:relative;">
      <canvas id="comparison-chart"></canvas>
      <div class="drag-overlay" id="comp-overlay"></div>
      <div class="drag-stats" id="comp-stats" style="max-height:80%;overflow-y:auto;"></div>
    </div>
    <p class="hint">차트에서 드래그하여 구간 비교</p>
    <div id="comparison-summary"></div>
    <div id="comparison-tracking"></div>
  </section>

  <div class="empty" id="compare-empty">자산을 1개 이상 선택하세요.</div>
</section>

<!-- ══ Tab: 추세 신호 ══════════════════════════════════════ -->
<section class="tab-panel" id="panel-taa">
  <header class="page-head">
    <h1>추세 신호</h1>
    <p>Faber 의 TAA 규칙입니다. <b>기준가 &gt; 이동평균</b> 이면 보유, 아래면 현금.
       <b>자산 비교</b> 탭에서 고른 자산을 그대로 씁니다. 이동평균은 선택 기간과 무관하게
       항상 전체 이력으로 계산하므로, 기간을 좁혀도 첫 구간이 비지 않습니다.</p>
  </header>

  <div class="asset-filter" id="taa-config" style="display:none;">
    <div class="card-head">
      <h2>규칙 설정</h2>
      <span class="period-info" id="taa-rule-info"></span>
    </div>
    <div class="period-row">
      <span class="period-label">이동평균</span>
      <div class="filter-chips" id="taa-win"></div>
    </div>
    <div class="period-row">
      <span class="period-label">판정 주기</span>
      <div class="filter-chips" id="taa-cadence">
        <button class="period-chip active" type="button" data-v="month">월말</button>
        <button class="period-chip" type="button" data-v="day">매일</button>
      </div>
      <span class="hint" style="margin:0;">매일 판정은 위프소가 급증합니다 — 아래 백테스트로 확인하세요.</span>
    </div>
    <div class="period-row">
      <span class="period-label">기울기 필터</span>
      <div class="filter-chips" id="taa-slope">
        <button class="period-chip active" type="button" data-v="1">사용</button>
        <button class="period-chip" type="button" data-v="0">미사용</button>
      </div>
      <span class="hint" style="margin:0;">이동평균 기울기 &gt; 0 도 함께 요구 (= 200일 모멘텀 &gt; 0)</span>
    </div>
    <div class="period-row">
      <span class="period-label">밴드</span>
      <div class="filter-chips" id="taa-buffer"></div>
      <span class="hint" style="margin:0;">이동평균 ±밴드 안쪽은 신호로 보지 않음 (경계 진동 억제)</span>
    </div>
    <div class="period-row">
      <span class="period-label">준거점</span>
      <div class="filter-chips" id="taa-epdepth"></div>
      <span class="hint" style="margin:0;">이 낙폭 이상인 하락 이벤트를 "정말 팔았어야 했던 때"로 보고 채점</span>
    </div>
  </div>

  <div class="asset-filter" id="taa-period" style="display:none;">
    <div class="card-head">
      <h2>표시 · 백테스트 기간</h2>
      <span class="period-info" id="taa-period-info"></span>
    </div>
    <div class="period-row">
      <span class="period-label">빠른 선택</span>
      <div class="filter-chips" id="taa-period-presets"></div>
    </div>
    <div class="period-row">
      <span class="period-label">연도</span>
      <div class="filter-chips" id="taa-period-years"></div>
    </div>
    <div class="period-row">
      <span class="period-label">직접 입력</span>
      <label>시작일 <input type="date" id="taa-start"></label>
      <label>종료일 <input type="date" id="taa-end"></label>
    </div>
  </div>

  <section class="card" id="taa-status-card" style="display:none;">
    <div class="card-head"><h2>현재 신호</h2></div>
    <div id="taa-status"></div>
  </section>

  <div id="taa-assets"></div>

  <div class="empty" id="taa-empty">선택된 자산이 없습니다. <b>자산 비교</b> 탭에서 자산을 선택하세요.</div>
</section>

<!-- ══ Tab: 개별 자산 상세 ═════════════════════════════════ -->
<section class="tab-panel" id="panel-detail">
  <header class="page-head">
    <h1>개별 자산 상세</h1>
    <p><b>자산 비교</b> 탭에서 선택한 자산의 지표·차트·하락 이벤트를 개별로 확인합니다.</p>
  </header>

  <div class="asset-filter" id="detail-period" style="display:none;">
    <div class="card-head">
      <h2>기간 선택</h2>
      <span class="period-info" id="detail-period-info"></span>
    </div>
    <div class="period-row">
      <span class="period-label">빠른 선택</span>
      <div class="filter-chips" id="detail-period-presets"></div>
    </div>
    <div class="period-row">
      <span class="period-label">연도</span>
      <div class="filter-chips" id="detail-period-years"></div>
    </div>
    <div class="period-row">
      <span class="period-label">직접 입력</span>
      <label>시작일 <input type="date" id="detail-start"></label>
      <label>종료일 <input type="date" id="detail-end"></label>
    </div>
  </div>

  %%FUND_SECTIONS%%

  <div class="empty" id="detail-empty">선택된 자산이 없습니다. <b>자산 비교</b> 탭에서 자산을 선택하세요.</div>
</section>

<!-- ══ Tab: 포트폴리오 분석 ════════════════════════════════ -->
<section class="tab-panel" id="panel-portfolio">
  <header class="page-head">
    <h1>포트폴리오 분석</h1>
    <p>자산별 비중을 합계 100%로 맞춘 뒤 분석하면 가상 포트폴리오의 성과·위험·기여도를 계산합니다.</p>
  </header>

  <section class="card" id="portfolio">
    <h2>구성 자산 및 비중</h2>
    <p class="fund-meta">체크박스로 자산을 고르고 비중(%)을 입력하세요.</p>
    <div class="portfolio-controls">
      <div class="selector-columns" id="fund-selector">
        <div class="selector-column"><h4>보험펀드 <span class="col-ccy-toggle" data-col="pf-insurance" data-modes="orig,jpy"></span></h4><div id="fund-selector-insurance"></div></div>
        <div class="selector-column"><h4>미국 <span class="col-ccy-toggle" data-col="pf-us" data-modes="orig,krw,jpy"></span></h4><div id="fund-selector-us"></div></div>
        <div class="selector-column"><h4>일본 <span class="col-ccy-toggle" data-col="pf-jp" data-modes="orig,krw"></span></h4><div id="fund-selector-jp"></div><h4 style="margin-top:1rem;">지수 <span class="col-ccy-toggle" data-col="pf-index" data-modes="orig,krw,jpy"></span></h4><div id="fund-selector-index"></div></div>
      </div>
      <div class="toolbar">
        <div class="weight-sum" id="weight-sum">비중 합계: 0%</div>
        <span class="sep">|</span>
        <div class="filter-actions">
          <button id="pf-equal" type="button">균등 배분</button>
          <button id="pf-normalize" type="button">100%로 맞추기</button>
        </div>
        <span class="pf-weight-hint">체크하면 자동으로 균등 배분됩니다. 직접 입력한 값은 고정돼요.</span>
      </div>
      <div class="period-block" id="pf-period" style="display:none;">
        <div class="period-head">
          <h4>분석 기간</h4>
          <span class="period-info" id="pf-date-info"></span>
        </div>
        <div class="period-row">
          <span class="period-label">빠른 선택</span>
          <div class="filter-chips" id="pf-period-presets"></div>
        </div>
        <div class="period-row">
          <span class="period-label">연도</span>
          <div class="filter-chips" id="pf-period-years"></div>
        </div>
        <div class="period-row">
          <span class="period-label">직접 입력</span>
          <label>시작일 <input type="date" id="pf-start"></label>
          <label>종료일 <input type="date" id="pf-end"></label>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">
        <button class="btn-analyze" id="btn-analyze" disabled>포트폴리오 분석</button>
        <button class="btn-analyze btn-ghost" id="btn-save-preset" disabled>현재 설정 저장</button>
      </div>
      <div class="preset-bar">
        <span>저장된 포트폴리오</span>
        <div id="preset-chips" class="filter-chips"></div>
      </div>
    </div>
    <div class="empty" id="pf-period-warn" style="display:none;padding:1.6rem 1.25rem;"></div>
    <div id="portfolio-results" style="display:none">
      <p class="track-note" id="pf-lag-note" style="display:none;"></p>
      <div class="metrics-grid" id="pf-metrics"></div>
      <div class="chart-row">
        <div class="chart-container" style="position:relative;">
          <canvas id="pf-nav-chart"></canvas>
          <div class="drag-overlay" id="pf-selection-overlay"></div>
          <div class="drag-stats" id="pf-selection-stats"></div>
        </div>
        <div class="chart-container"><canvas id="pf-dd-chart"></canvas></div>
      </div>
      <p class="hint">NAV 차트에서 드래그하여 구간 분석 (클릭하면 해제)</p>
      <div id="pf-yearly"></div>
      <div id="pf-trailing"></div>
      <div id="pf-dd-table"></div>
      <div id="pf-ls-table"></div>
      <div id="pf-corr-table"></div>
    </div>
  </section>
</section>

</div></main>

<script>
const FUNDS = %%FUND_JSON%%;
const RISK_FREE = %%RISK_FREE_DECIMAL%%;
const CCY_SYMBOL = { USD: '$', JPY: '¥', KRW: '₩' };
function ccySym(fund) { return CCY_SYMBOL[fund.currency] || fund.currency; }

// ── Theme (light / dark) ──
// Chart.js resolves tick, grid and legend colours from its defaults once, when the
// chart is constructed, and stores them on the instance. Updating the defaults only
// covers charts built later, so already-live instances get overwritten in place.
function syncChartTheme() {
  const css = getComputedStyle(document.documentElement);
  const tick = css.getPropertyValue('--muted').trim();
  const grid = css.getPropertyValue('--border').trim();
  Chart.defaults.color = tick;
  Chart.defaults.borderColor = grid;
  document.querySelectorAll('canvas').forEach(c => {
    const ch = Chart.getChart(c);
    if (!ch) return;
    // Assign into the existing option objects — replacing them drops the resolver
    // entries Chart.js merged in (tick callbacks, legend generateLabels, ...).
    Object.values(ch.options.scales || {}).forEach(sc => {
      if (sc.grid) sc.grid.color = grid; else sc.grid = { color: grid };
      if (sc.border) sc.border.color = grid; else sc.border = { color: grid };
      if (sc.ticks) sc.ticks.color = tick; else sc.ticks = { color: tick };
    });
    const legend = ch.options.plugins && ch.options.plugins.legend;
    if (legend) {
      if (legend.labels) legend.labels.color = tick; else legend.labels = { color: tick };
    }
    ch.update('none');
  });
}

function setTheme(theme, persist) {
  document.documentElement.dataset.theme = theme;
  if (persist) { try { localStorage.setItem('fund_dashboard_theme', theme); } catch (e) {} }
  document.getElementById('theme-label').textContent = theme === 'dark' ? '라이트 모드' : '다크 모드';
  syncChartTheme();
}

document.getElementById('theme-toggle').addEventListener('click', () => {
  setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark', true);
});

// Follow the OS setting until the user picks a theme explicitly.
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
  let stored = null;
  try { stored = localStorage.getItem('fund_dashboard_theme'); } catch (err) {}
  if (!stored) setTheme(e.matches ? 'dark' : 'light', false);
});

setTheme(document.documentElement.dataset.theme, false);

// ── Tab navigation (sidebar) ──
function activateTab(tabId) {
  document.querySelectorAll('#side-nav .nav-item').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === tabId));
  // Charts created while their panel was hidden have zero size — re-measure now.
  const panel = document.getElementById(tabId);
  if (panel) panel.querySelectorAll('canvas').forEach(c => { const ch = Chart.getChart(c); if (ch) ch.resize(); });
  if (history.replaceState) history.replaceState(null, '', '#' + tabId.replace('panel-', ''));
}

document.querySelectorAll('#side-nav .nav-item').forEach(btn => {
  btn.addEventListener('click', () => activateTab(btn.dataset.tab));
});

// Keep sidebar badges and empty-state placeholders in sync with the selections.
function setBadge(id, n) {
  const el = document.getElementById(id);
  el.textContent = n;
  el.classList.toggle('show', n > 0);
}

function refreshTabState() {
  const picked = document.querySelectorAll('#filter-chips-insurance input:checked, #filter-chips-us input:checked, #filter-chips-jp input:checked, #filter-chips-index input:checked').length;
  setBadge('badge-compare', picked);
  setBadge('badge-taa', picked);
  setBadge('badge-detail', picked);
  document.getElementById('compare-empty').style.display = picked >= 1 ? 'none' : '';
  document.getElementById('detail-empty').style.display = picked > 0 ? 'none' : '';
  // Runs after the chip handlers have toggled section visibility and built any new charts.
  refreshDetailPeriod();
  refreshTaa();
}

window.addEventListener('DOMContentLoaded', () => {
  const hash = location.hash.slice(1);
  if (hash) {
    const target = document.getElementById('panel-' + hash);
    if (target) activateTab(target.id);
  }
  refreshTabState();
});

// Generic data getter respecting currency mode
function getDataByMode(fund, mode, key) {
  if (mode === 'krw' && fund.krw && fund.krw[key]) return fund.krw[key];
  if (mode === 'jpy' && fund.jpy && fund.jpy[key]) return fund.jpy[key];
  if (mode === 'usd' && fund.usd && fund.usd[key]) return fund.usd[key];
  return fund[key];
}

// Build currency toggle buttons for a fund
function buildCcyToggle(fund, idx, style, stateObj, onChange) {
  if (!fund.hasKrw && !fund.hasJpy && !fund.hasUsd) return null;
  const span = document.createElement('span');
  span.className = 'currency-toggle';
  span.style.cssText = style;
  const defaultMode = fund.currency === 'USD' ? 'krw' : 'orig';
  const btnList = [];
  // Native currency button
  btnList.push({ mode: 'orig', label: ccySym(fund), active: defaultMode === 'orig' });
  // USD button (for KRW assets)
  if (fund.hasUsd) btnList.push({ mode: 'usd', label: '$', active: false });
  // KRW button (only if native is not KRW)
  if (fund.hasKrw) btnList.push({ mode: 'krw', label: '₩', active: defaultMode === 'krw' });
  // JPY button
  if (fund.hasJpy) btnList.push({ mode: 'jpy', label: '¥', active: false });

  let btns = btnList.map((b, i) => {
    const radius = i === 0 ? 'border-radius:4px 0 0 4px;' : i === btnList.length - 1 ? 'border-radius:0 4px 4px 0;' : '';
    const bl = i > 0 ? 'border-left:none;' : '';
    return `<button class="btn-currency${b.active ? ' active' : ''}" data-mode="${b.mode}" style="padding:0.1rem 0.4rem;font-size:0.7rem;${radius}${bl}">${b.label}</button>`;
  }).join('');
  span.innerHTML = btns;
  span.querySelectorAll('.btn-currency').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault(); e.stopPropagation();
      stateObj[idx] = btn.dataset.mode;
      span.querySelectorAll('.btn-currency').forEach(b => b.classList.toggle('active', b === btn));
      if (onChange) onChange();
    });
  });
  return span;
}

// Column-level currency toggle: sets all child fund toggles at once
const MODE_LABELS = { orig: { USD: '$', JPY: '¥', KRW: '₩' }, krw: '₩', jpy: '¥' };

function buildColCcyToggles() {
  document.querySelectorAll('.col-ccy-toggle').forEach(span => {
    const colId = span.dataset.col;  // e.g. "filter-us", "corr-jp", "pf-insurance"
    const modes = span.dataset.modes.split(',');
    // Determine which selector group and state object
    const [group, region] = colId.split('-');  // "filter"+"us", "corr"+"jp", "pf"+"insurance"

    const modeLabel = { orig: region === 'us' ? '$' : region === 'jp' ? '¥' : '₩', krw: '₩', jpy: '¥' };
    // index column: orig depends on each fund's currency, so use generic label
    if (region === 'index') modeLabel.orig = '기본';

    modes.forEach((mode, i) => {
      const btn = document.createElement('button');
      btn.className = 'btn-currency';
      btn.textContent = modeLabel[mode];
      btn.style.cssText = i === 0 ? 'border-radius:4px 0 0 4px;' :
        i === modes.length - 1 ? 'border-radius:0 4px 4px 0;border-left:none;' : 'border-left:none;';
      btn.addEventListener('click', () => {
        // Update column toggle active state
        span.querySelectorAll('.btn-currency').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        // Find the sibling container and update all fund toggles
        const column = span.closest('.selector-column');
        column.querySelectorAll('.currency-toggle .btn-currency').forEach(cb => {
          if (cb.dataset.mode === mode) {
            cb.click();
          }
        });
      });
      span.appendChild(btn);
    });
  });
}
buildColCcyToggles();

// Fund region helper
function fundRegion(fund) {
  if (!fund.isBench) return 'insurance';
  if (fund.region) return fund.region;  // 'us', 'jp', 'index'
  if (fund.currency === 'JPY') return 'jp';
  return 'us';
}
function chipLabel(fund) {
  if (fund.currency === 'JPY' && fund.isBench) return fund.name;  // JPY: show full name (숫자 코드만으로는 식별 어려움)
  return fund.shortName || fund.name;
}
function compactLabel(fund) {
  // Short code only: N760, 1329, SPY, KS200, etc.
  if (fund.isBench) return fund.shortName;
  // Insurance: extract code from name like "달러미국채형(환오픈형)(N760)" → "N760"
  const m = fund.name.match(/\(([A-Z0-9]+)\)\s*$/);
  return m ? m[1] : fund.shortName;
}

// ── Asset Filter ──
const filterCurrencyState = {};
FUNDS.forEach((f, i) => { if (f.hasKrw || f.hasJpy) filterCurrencyState[i] = f.currency === 'USD' ? 'krw' : 'orig'; });

(function buildFilter() {
  const filterContainers = {
    insurance: document.getElementById('filter-chips-insurance'),
    us: document.getElementById('filter-chips-us'),
    jp: document.getElementById('filter-chips-jp'),
    index: document.getElementById('filter-chips-index'),
  };
  FUNDS.forEach((fund, idx) => {
    const chip = document.createElement('label');
    chip.className = 'filter-chip';
    chip.innerHTML = `<input type="checkbox" data-idx="${idx}">${chipLabel(fund)}`;

    const fToggle = buildCcyToggle(fund, idx, 'margin:0 0 0 0.3rem;display:inline-flex;', filterCurrencyState,
                                   () => { updateComparison(); refreshTaa(); });
    if (fToggle) chip.appendChild(fToggle);

    chip.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-currency')) return;
      setTimeout(() => {
        const checked = chip.querySelector('input').checked;
        chip.classList.toggle('active', checked);
        const section = document.getElementById('fund-' + idx);
        if (section) section.classList.toggle('hidden', !checked);
        if (checked && !section._chartsCreated) {
          createSingleChart(idx);
          section._chartsCreated = true;
        }
        updateComparison();
      }, 0);
    });
    filterContainers[fundRegion(fund)].appendChild(chip);
  });

  const allFilterChips = () => document.querySelectorAll('#filter-chips-insurance .filter-chip, #filter-chips-us .filter-chip, #filter-chips-jp .filter-chip, #filter-chips-index .filter-chip');

  document.getElementById('filter-all').addEventListener('click', () => {
    allFilterChips().forEach(chip => {
      const cb = chip.querySelector('input');
      if (!cb.checked) { cb.checked = true; chip.classList.add('active'); }
      const idx = cb.dataset.idx;
      const section = document.getElementById('fund-' + idx);
      if (section) { section.classList.remove('hidden');
        if (!section._chartsCreated) { createSingleChart(+idx); section._chartsCreated = true; }
      }
    });
    updateComparison();
  });

  document.getElementById('filter-none').addEventListener('click', () => {
    allFilterChips().forEach(chip => {
      const cb = chip.querySelector('input');
      cb.checked = false; chip.classList.remove('active');
      const section = document.getElementById('fund-' + cb.dataset.idx);
      if (section) section.classList.add('hidden');
    });
    updateComparison();
  });
})();

// ── Asset Comparison Chart ──
const COMPARISON_COLORS = [
  '#3b82f6','#ef4444','#16a34a','#f59e0b','#8b5cf6',  // blue, red, green, amber, purple
  '#ec4899','#06b6d4','#f97316','#6366f1','#14b8a6',  // pink, cyan, orange, indigo, teal
  '#a16207','#0d9488','#b91c1c','#4338ca','#a3e635',  // brown, dark teal, dark red, dark indigo, lime
  '#db2777','#0369a1','#c2410c','#7c3aed','#059669',  // magenta, dark blue, burnt orange, violet, emerald
  '#d97706','#be185d','#1d4ed8','#b45309','#7e22ce',  // dark amber, rose, royal blue, dark orange, deep purple
  '#15803d','#9333ea','#ea580c','#0891b2','#4f46e5',  // forest green, purple, deep orange, dark cyan, dark indigo
  '#ca8a04','#e11d48','#2dd4bf',                       // gold, crimson, aqua
];
// Compound each asset over ITS OWN dates, then read the result on a shared grid.
// Intersecting dates first and summing only the common days silently throws away the
// returns of any asset that quotes on days the others don't — the insurance funds price
// every calendar day while ETFs price only on trading days, so the funds were losing
// every weekend accrual (N1M0 read 8.03%/yr against a true 12.46%/yr).
function segmentReturns(daily, dates) {
  const src = daily.dates, ret = daily.returns;
  let j = 0;
  while (j < src.length && src[j] < dates[0]) j++;
  return dates.map(d => {
    let f = 1;
    while (j < src.length && src[j] <= d) { f *= (1 + ret[j]); j++; }
    return f - 1;
  });
}

let comparisonChart = null;
let compFullDates = [], compFullNavs = [], compSelectedIdxs = [];
let compSelectedCount = 0;

// ── Period picker ──
// Shared by the comparison and portfolio tabs. Both re-base their series to the range
// start, so narrowing the range is what makes "how did these do in 2022" a different
// question from "since inception".
function shiftYears(iso, n) {
  const d = new Date(iso);
  d.setFullYear(d.getFullYear() - n);
  return d.toISOString().slice(0, 10);
}

// Trailing windows longer than the available history would just repeat 전체 — drop them.
function periodPresetDefs(lo, hi) {
  return [
    { key: 'all', label: '전체', start: '', end: '' },
    { key: 'ytd', label: 'YTD', start: hi.slice(0, 4) + '-01-01', end: hi },
    { key: '1y', label: '최근 1년', start: shiftYears(hi, 1), end: hi },
    { key: '3y', label: '최근 3년', start: shiftYears(hi, 3), end: hi },
    { key: '5y', label: '최근 5년', start: shiftYears(hi, 5), end: hi },
    { key: '10y', label: '최근 10년', start: shiftYears(hi, 10), end: hi },
  ].filter(p => p.key === 'all' || p.start > lo);
}

// ids: { card, info, presets, years, start, end, infoLabel? }; onChange fires after every
// range edit. infoLabel names what the bounds mean — the detail tab spans a union, not a
// common period.

function makePeriodPicker(ids, onChange) {
  const el = k => document.getElementById(ids[k]);
  const startInput = el('start'), endInput = el('end');
  let range = { start: '', end: '' };   // '' on either side = open (use the full common range)
  let bounds = { start: '', end: '' };  // full common range of the current asset selection
  let preset = 'all';                   // 'all' | 'ytd' | '1y' | 'y2022' | ... | 'custom'

  // Crossing the two dates opens the *other* end rather than swapping them — swapping
  // would throw away the date the user just typed.
  const onEdit = (edited) => {
    let start = startInput.value, end = endInput.value;
    if (start && end && start > end) { if (edited === 'start') end = ''; else start = ''; }
    range = { start, end };
    preset = 'custom';
    onChange();
  };
  startInput.addEventListener('change', () => onEdit('start'));
  endInput.addEventListener('change', () => onEdit('end'));

  function render(note) {
    const lo = bounds.start, hi = bounds.end;
    el('card').style.display = '';
    el('info').textContent = `${(typeof ids.infoLabel === 'function' ? ids.infoLabel() : ids.infoLabel) || '전체 공통 기간'} ${lo} ~ ${hi}` + (note || '');

    startInput.min = endInput.min = lo;
    startInput.max = endInput.max = hi;
    startInput.value = range.start || lo;
    endInput.value = range.end || hi;

    const presetBox = el('presets');
    const defs = periodPresetDefs(lo, hi);
    const presetSig = defs.map(d => d.key).join(',') + '|' + hi;
    if (presetBox.dataset.sig !== presetSig) {
      presetBox.dataset.sig = presetSig;
      presetBox.innerHTML = '';
      defs.forEach(d => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'period-chip'; b.textContent = d.label; b.dataset.key = d.key;
        b.addEventListener('click', () => {
          preset = d.key;
          range = { start: d.start, end: d.end };
          onChange();
        });
        presetBox.appendChild(b);
      });
    }

    const yearBox = el('years');
    const y0 = +lo.slice(0, 4), y1 = +hi.slice(0, 4);
    const yearSig = y0 + '-' + y1;
    if (yearBox.dataset.sig !== yearSig) {
      yearBox.dataset.sig = yearSig;
      yearBox.innerHTML = '';
      for (let y = y0; y <= y1; y++) {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'period-chip'; b.textContent = y; b.dataset.key = 'y' + y;
        b.addEventListener('click', () => {
          // Clamp here, not in setBounds — a partial first/last year is still that year,
          // and trimming it there would demote the chip to "custom".
          const s = y + '-01-01', e = y + '-12-31';
          preset = 'y' + y;
          range = { start: s < bounds.start ? bounds.start : s,
                    end: e > bounds.end ? bounds.end : e };
          onChange();
        });
        yearBox.appendChild(b);
      }
    }

    [...presetBox.children, ...yearBox.children]
      .forEach(b => b.classList.toggle('active', b.dataset.key === preset));
  }

  return {
    range: () => range,
    narrowed: () => !!(range.start || range.end),
    filter: dates => dates.filter(d =>
      (!range.start || d >= range.start) && (!range.end || d <= range.end)),
    hide() { el('card').style.display = 'none'; },
    // Re-point at a new common range, keeping the picked window usable, then redraw.
    setBounds(lo, hi, note) {
      bounds = { start: lo, end: hi };
      let { start, end } = range;
      const before = start + '|' + end;
      if (start && start < lo) start = lo;
      if (end && end > hi) end = hi;
      if (start && start > hi) start = '';
      if (end && end < lo) end = '';
      range = { start, end };
      if (!start && !end) preset = 'all';
      else if (start + '|' + end !== before) preset = 'custom';
      render(note);
    },
  };
}

const compPeriod = makePeriodPicker({
  card: 'compare-period', info: 'comp-period-info', presets: 'comp-period-presets',
  years: 'comp-period-years', start: 'comp-start', end: 'comp-end',
  infoLabel: () => compSelectedCount === 1 ? '전체 기간' : '전체 공통 기간',
}, () => updateComparison());

// The portfolio is analysed on demand, so a range edit only re-runs it once asked for.
let pfAnalyzed = false;
const pfPeriod = makePeriodPicker({
  card: 'pf-period', info: 'pf-date-info', presets: 'pf-period-presets',
  years: 'pf-period-years', start: 'pf-start', end: 'pf-end',
}, () => {
  updateWeightSum();
  if (pfAnalyzed) runPortfolioAnalysis();
});

function updateComparison() {
  const section = document.getElementById('comparison-section');
  const empty = document.getElementById('compare-empty');
  const selected = [];
  document.querySelectorAll('#filter-chips-insurance input:checked, #filter-chips-us input:checked, #filter-chips-jp input:checked, #filter-chips-index input:checked').forEach(cb => {
    selected.push(+cb.dataset.idx);
  });
  refreshTabState();

  // keepPeriod: the range picker stays up when the range itself is what needs fixing.
  const hideSection = (msg, keepPeriod) => {
    section.style.display = 'none';
    if (!keepPeriod) compPeriod.hide();
    empty.textContent = msg;
    empty.style.display = '';
  };

  if (selected.length < 1) { hideSection('자산을 1개 이상 선택하세요.'); return; }

  // Find common date range (respecting per-asset currency toggle)
  const dailySets = selected.map(idx => {
    const fund = FUNDS[idx];
    return getDataByMode(fund, filterCurrencyState[idx] || 'krw', 'daily');
  });
  const dateSets = dailySets.map(d => new Set(d.dates));
  const commonAll = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d))).sort();

  if (commonAll.length < 2) {
    hideSection(selected.length === 1 ? '선택한 자산의 데이터가 부족합니다.'
                                     : '선택한 자산들의 공통 기간이 없습니다. 조합을 바꿔보세요.');
    return;
  }

  compSelectedCount = selected.length;
  compPeriod.setBounds(commonAll[0], commonAll[commonAll.length - 1]);

  const common = compPeriod.filter(commonAll);
  if (common.length < 2) { hideSection('선택한 기간에 공통 데이터가 없습니다. 기간을 넓혀보세요.', true); return; }

  section.style.display = '';
  empty.style.display = 'none';

  // Build NAV series normalized to 100 at start
  const datasets = selected.map((idx, si) => {
    const nav = [100];
    // nav has length common.length+1, dates need synthetic first date
    segmentReturns(dailySets[si], common).forEach(r => nav.push(nav[nav.length - 1] * (1 + r)));
    return nav;
  });

  const firstDate = new Date(common[0]);
  firstDate.setDate(firstDate.getDate() - 1);
  const chartDates = [firstDate.toISOString().slice(0, 10), ...common];

  // Add the fee back on to recover what the underlying holdings actually returned.
  const grossSets = selected.map((idx, si) => addBackFee(datasets[si], chartDates, FUNDS[idx].fee));
  const showNavs = feeMode === 'gross'
    ? datasets.map((nav, si) => grossSets[si] || nav)
    : datasets;

  // Downsample
  const step = Math.max(1, Math.floor(chartDates.length / 600));
  const dsDates = chartDates.filter((_, i) => i % step === 0);

  // Store full data for drag-select
  compFullDates = chartDates;
  compFullNavs = showNavs;
  compSelectedIdxs = selected;

  const feeLabel = { net: '보수 차감 후', gross: '보수 차감 전', both: '보수 차감 전후' }[feeMode];
  const spanLabel = compPeriod.narrowed() ? '선택 기간' : selected.length === 1 ? '기간' : '공통 기간';
  document.getElementById('comparison-meta').textContent =
    `${spanLabel}: ${common[0]} ~ ${common[common.length-1]}` +
    ` (${common.length}거래일) | 시작점 = 100으로 정규화 | ${feeLabel}`;

  // A one-year range on a 'year' axis draws a single tick — scale the unit to the span.
  const spanDays = (new Date(chartDates[chartDates.length-1]) - new Date(chartDates[0])) / 86400000;
  const timeUnit = spanDays > 1500 ? 'year' : spanDays > 400 ? 'quarter'
                 : spanDays > 120 ? 'month' : spanDays > 35 ? 'week' : 'day';

  const chartDatasets = selected.map((idx, si) => ({
    label: (FUNDS[idx].shortName || FUNDS[idx].name) + (feeMode === 'gross' && grossSets[si] ? ' (보수 전)' : ''),
    data: showNavs[si].filter((_, i) => i % step === 0).map(v => +v.toFixed(2)),
    borderColor: COMPARISON_COLORS[idx % COMPARISON_COLORS.length],
    backgroundColor: 'transparent',
    fill: false, pointRadius: 0, borderWidth: 1.8,
  }));
  // "둘 다": the fee-free twin rides above its own fund in the same colour, dashed.
  if (feeMode === 'both') {
    selected.forEach((idx, si) => {
      if (!grossSets[si]) return;
      chartDatasets.push({
        label: (FUNDS[idx].shortName || FUNDS[idx].name) + ' (보수 전)',
        data: grossSets[si].filter((_, i) => i % step === 0).map(v => +v.toFixed(2)),
        borderColor: COMPARISON_COLORS[idx % COMPARISON_COLORS.length],
        backgroundColor: 'transparent', borderDash: [5, 4],
        fill: false, pointRadius: 0, borderWidth: 1.4,
      });
    });
  }

  if (comparisonChart) comparisonChart.destroy();
  comparisonChart = new Chart(document.getElementById('comparison-chart'), {
    type: 'line',
    data: { labels: dsDates, datasets: chartDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: 'time', time: { unit: timeUnit }, ticks: { maxTicksLimit: 10 } },
        y: { beginAtZero: false }
      },
      plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 14, font: { size: 11 } } } }
    }
  });

  // Render summary table
  renderComparisonSummary(selected, chartDates, showNavs);
  renderTrackingSummary(selected, chartDates, datasets, grossSets);
}

function renderComparisonSummary(selected, dates, navSets) {
  const el = document.getElementById('comparison-summary');
  if (selected.length < 1) { el.innerHTML = ''; return; }
  if (dates.length < 30) {
    el.innerHTML = '<p class="fund-meta" style="margin-top:1rem;">선택 기간이 30거래일 미만이라 ' +
                   '요약 지표(CAGR·변동성 등)는 생략했습니다.</p>';
    return;
  }

  const n = dates.length;
  const totalDays = (new Date(dates[n-1]) - new Date(dates[0])) / 86400000;
  const totalYears = totalDays / 365.25;

  // Per-asset metrics
  const metrics = selected.map((idx, si) => {
    const nav = navSets[si];
    const daily = getPfFundData(FUNDS[idx], idx, 'daily');

    // Period (from asset's own data, not common)
    const firstDate = daily.dates[0];
    const lastDate = daily.dates[daily.dates.length - 1];

    // Return over the selected window, and its annualised form
    const totalReturn = (nav[n-1] / nav[0] - 1) * 100;
    const cagr = (Math.pow(nav[n-1] / nav[0], 1 / totalYears) - 1) * 100;

    // Volatility (monthly basis over the shown window)
    let volRaw = monthlyVol(dates, nav);
    if (volRaw === null) {
      const dr = [];
      for (let i = 1; i < n; i++) dr.push(nav[i] / nav[i-1] - 1);
      const mean = dr.reduce((s,v) => s+v, 0) / dr.length;
      const variance = dr.reduce((s,v) => s + (v-mean)**2, 0) / (dr.length - 1);
      volRaw = Math.sqrt(variance) * Math.sqrt(dr.length / totalYears);
    }
    const vol = volRaw * 100;

    // MDD
    let peak = nav[0], mdd = 0;
    for (const v of nav) { peak = Math.max(peak, v); mdd = Math.min(mdd, (v - peak) / peak); }

    // Average drawdown
    let inDd = false, ddDepths = [];
    let ddPeak = nav[0];
    for (let i = 0; i < nav.length; i++) {
      ddPeak = Math.max(ddPeak, nav[i]);
      const dd = (nav[i] - ddPeak) / ddPeak;
      if (!inDd && dd < 0) { inDd = true; }
      else if (inDd && dd >= 0) { inDd = false; }
      if (inDd) ddDepths.push(dd);
    }
    const avgDd = ddDepths.length > 0 ? (ddDepths.reduce((s,v)=>s+v,0)/ddDepths.length)*100 : 0;

    return {
      idx, name: compactLabel(FUNDS[idx]),
      color: COMPARISON_COLORS[idx % COMPARISON_COLORS.length],
      firstDate, totalReturn, cagr, vol, mdd: mdd * 100, avgDd,
    };
  });

  // Monthly correlation matrix (using monthly data)
  const monthlyData = selected.map(idx => {
    const m = getPfFundData(FUNDS[idx], idx, 'monthly');
    const map = {};
    m.dates.forEach((d, i) => { map[d] = m.returns[i]; });
    return { dates: new Set(m.dates), map };
  });
  let commonM = compPeriod
    .filter([...monthlyData[0].dates].filter(d => monthlyData.every(m => m.dates.has(d))))
    .sort();

  const pc = v => +v > 0 ? 'positive' : +v < 0 ? 'negative' : '';
  const fp = (v, s) => (s && v > 0 ? '+' : '') + v.toFixed(2) + '%';

  // Summary table
  let header = '<tr><th></th>';
  metrics.forEach(m => { header += `<th><span style="color:${m.color};">●</span> ${m.name}</th>`; });
  header += '</tr>';

  let rows = '';
  // Row: 데이터 시작
  rows += '<tr><td>데이터 시작</td>';
  metrics.forEach(m => { rows += `<td>${m.firstDate}</td>`; });
  rows += '</tr>';
  // Row: 총보수 — blank for benchmarks, which have no wrapper
  if (metrics.some(m => FUNDS[m.idx].fee)) {
    rows += '<tr><td>총보수<span class="row-note">연, 기준가에 반영됨</span></td>';
    metrics.forEach(m => { rows += `<td>${FUNDS[m.idx].fee ? FUNDS[m.idx].fee.toFixed(2) + '%' : '—'}</td>`; });
    rows += '</tr>';
  }
  // Row: total return over the selected window
  rows += `<tr><td>총 수익률<span class="row-note">${dates[0]} → ${dates[n-1]}</span></td>`;
  metrics.forEach(m => { rows += `<td class="${pc(m.totalReturn)}">${fp(m.totalReturn, true)}</td>`; });
  rows += '</tr>';
  // Row: CAGR — annualising much less than a year turns noise into a headline number.
  // The bar is in days, not years: a calendar-year pick spans ~360 trading-day-bounded days.
  const annualisable = totalDays >= 350;
  rows += `<tr><td>CAGR<span class="row-note">연환산</span></td>`;
  metrics.forEach(m => {
    rows += annualisable ? `<td class="${pc(m.cagr)}">${fp(m.cagr, true)}</td>`
                         : '<td title="기간이 1년 미만이라 연환산하지 않습니다">—</td>';
  });
  rows += '</tr>';
  // Row: Volatility
  rows += '<tr><td>변동성<span class="row-note">월간 기준, 연환산</span></td>';
  metrics.forEach(m => { rows += `<td>${m.vol.toFixed(2)}%</td>`; });
  rows += '</tr>';
  // Row: MDD
  rows += '<tr><td>MDD</td>';
  metrics.forEach(m => { rows += `<td class="negative">${m.mdd.toFixed(2)}%</td>`; });
  rows += '</tr>';
  // Row: Avg Drawdown
  rows += '<tr><td>평균 드로다운</td>';
  metrics.forEach(m => { rows += `<td class="negative">${m.avgDd.toFixed(2)}%</td>`; });
  rows += '</tr>';

  // Correlation matrix (separate heatmap table)
  let corrHtml = '';
  if (selected.length >= 2 && commonM.length >= 6) {
    const arrays = monthlyData.map(m => commonM.map(d => m.map[d]));
    const means = arrays.map(arr => arr.reduce((s,v)=>s+v,0)/commonM.length);

    // Compute full correlation matrix
    const corrMatrix = [];
    for (let i = 0; i < arrays.length; i++) {
      const row = [];
      for (let j = 0; j < arrays.length; j++) {
        if (i === j) { row.push(1); continue; }
        let sXY=0,sX2=0,sY2=0;
        for (let k=0;k<commonM.length;k++) {
          const dx=arrays[i][k]-means[i], dy=arrays[j][k]-means[j];
          sXY+=dx*dy; sX2+=dx*dx; sY2+=dy*dy;
        }
        row.push(Math.sqrt(sX2*sY2)>0 ? sXY/Math.sqrt(sX2*sY2) : 0);
      }
      corrMatrix.push(row);
    }

    function corrCellStyle(v) {
      if (v >= 1) return 'background:#1d4ed8;color:#fff;';
      if (v >= 0) return `background:rgba(37,99,235,${(v*0.5).toFixed(2)});color:${v>0.7?'#fff':'var(--text)'};`;
      return `background:rgba(220,38,38,${(Math.abs(v)*0.5).toFixed(2)});color:${v<-0.7?'#fff':'var(--text)'};`;
    }

    let corrHeader = '<tr><th></th>' + metrics.map(m => `<th><span style="color:${m.color};">●</span> ${m.name}</th>`).join('') + '</tr>';
    let corrRows = corrMatrix.map((row, i) =>
      '<tr><th><span style="color:' + metrics[i].color + ';">●</span> ' + metrics[i].name + '</th>' +
      row.map(v => `<td style="${corrCellStyle(v)}text-align:center;padding:0.5rem;">${v.toFixed(2)}</td>`).join('') + '</tr>'
    ).join('');

    corrHtml = `
      <h3>상관행렬 (월간 수익률, ${commonM.length}개월)</h3>
      <table class="corr-table" style="font-size:0.8rem;">
        ${corrHeader}${corrRows}
      </table>`;
  }

  el.innerHTML = `
    <h3>${selected.length === 1 ? '요약' : '요약 비교'} (${compPeriod.narrowed() ? '선택 기간' : selected.length === 1 ? '기간' : '공통 기간'} ${dates[0]} ~ ${dates[n-1]})</h3>
    <div style="overflow-x:auto;">
    <table style="font-size:0.8rem;min-width:100%;">
      ${header}${rows}
    </table>
    </div>
    ${corrHtml}`;
}

// ── Fees and index tracking ──
// The published 기준가 is already net of the wrapper + underlying fee, so adding the fee
// back reconstructs what the holdings themselves returned. Comparing THAT against the
// tracked index isolates the manager's tracking from the cost of the wrapper.
let feeMode = 'net';   // 'net' | 'gross' | 'both'

const MS_PER_YEAR = 365 * 86400000;

// Returns null for anything without a fee on file (benchmarks, N9K0).
function addBackFee(nav, dates, feePct) {
  if (!feePct) return null;
  const t0 = new Date(dates[0]).getTime();
  return nav.map((v, i) =>
    v * Math.pow(1 + feePct / 100, (new Date(dates[i]).getTime() - t0) / MS_PER_YEAR));
}

// Benchmarks carry their ticker in shortName (see render_html).
function benchIdx(code) {
  return FUNDS.findIndex(f => f.isBench && f.shortName === code);
}

(function () {
  document.querySelectorAll('#fee-mode .period-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      feeMode = btn.dataset.fee;
      document.querySelectorAll('#fee-mode .period-chip')
        .forEach(b => b.classList.toggle('active', b === btn));
      updateComparison();
    });
  });
})();

// Fund vs the index it tracks, over the selected window.
function renderTrackingSummary(selected, dates, netNavs, grossSets) {
  const el = document.getElementById('comparison-tracking');
  const n = dates.length;
  const years = (new Date(dates[n - 1]) - new Date(dates[0])) / MS_PER_YEAR;
  const pos = selected.reduce((m, idx, si) => (m[idx] = si, m), {});

  const pairs = [];
  const missing = [];
  selected.forEach((idx, si) => {
    const code = FUNDS[idx].bench;
    if (!code) return;
    const bi = benchIdx(code);
    if (bi < 0) return;
    if (!(bi in pos)) { missing.push({ fund: compactLabel(FUNDS[idx]), code }); return; }
    pairs.push({ idx, si, code, bsi: pos[bi], bidx: bi });
  });

  if (pairs.length === 0) {
    el.innerHTML = missing.length === 0 ? '' :
      `<h3>지수 추적</h3><p class="track-note">` +
      missing.map(m => `<b>${m.fund}</b>의 추적 지수 <b>${m.code}</b>`).join(', ') +
      `를 함께 선택하면 추적 비교가 표시됩니다.</p>`;
    return;
  }

  const ret = nav => (nav[n - 1] / nav[0] - 1) * 100;
  const ann = nav => (Math.pow(nav[n - 1] / nav[0], 1 / years) - 1) * 100;
  // Under a year an annualised gap is mostly noise, so compare cumulative instead.
  const annualisable = years >= 350 / 365;
  const f = (v, s) => (s && v > 0 ? '+' : '') + v.toFixed(2) + '%';
  const cls = v => v > 0 ? 'positive' : v < 0 ? 'negative' : '';

  const rows = pairs.map(p => {
    const net = netNavs[p.si], gross = grossSets[p.si] || netNavs[p.si], bench = netNavs[p.bsi];
    const m = annualisable ? ann : ret;
    const gapGross = m(gross) - m(bench);
    const gapNet = m(net) - m(bench);
    // Tracking is judged on the fee-free series: that is the part the manager controls.
    const verdict = Math.abs(gapGross) <= (annualisable ? 1 : 1.5) ? 'track-ok'
                  : Math.abs(gapGross) <= (annualisable ? 3 : 4.5) ? '' : 'track-bad';
    // A price-only index understates its own holdings by the dividend yield, so a fund
    // tracking it should sit ABOVE the line — the gap is not a tracking failure.
    const po = FUNDS[p.bidx].priceOnly;
    return `<tr>
      <td>${compactLabel(FUNDS[p.idx])}</td>
      <td>${FUNDS[p.bidx].shortName}${po ? '<span class="row-note">배당 미포함 지수</span>' : ''}</td>
      <td>${FUNDS[p.idx].fee ? FUNDS[p.idx].fee.toFixed(2) + '%' : '—'}</td>
      <td class="${cls(m(net))}">${f(m(net), true)}</td>
      <td class="${cls(m(gross))}">${f(m(gross), true)}</td>
      <td class="${cls(m(bench))}">${f(m(bench), true)}</td>
      <td class="${po ? '' : verdict}">${f(gapGross, true)}${po ? ' *' : ''}</td>
      <td class="${cls(gapNet)}">${f(gapNet, true)}</td>
    </tr>`;
  }).join('');

  const basis = annualisable ? '연환산' : `누적 (${(years * 12).toFixed(0)}개월)`;
  let hint = missing.length === 0 ? '' :
    `<p class="track-note">` + missing.map(m => `<b>${m.fund}</b> → <b>${m.code}</b>`).join(', ') +
    `도 함께 선택하면 추가로 표시됩니다.</p>`;
  if (pairs.some(p => FUNDS[p.bidx].priceOnly)) {
    hint = `<p class="track-note">* 배당이 빠진 가격지수라 배당수익률만큼 펀드가 앞서는 것이 정상입니다. ` +
           `이 행의 추적 차이는 그만큼 과소평가되니 판정 색을 붙이지 않았습니다.</p>` + hint;
  }

  el.innerHTML = `
    <h3>지수 추적 (${basis})</h3>
    <p class="track-note">
      <b>추적 차이</b> = 보수 전 − 지수. 운용이 지수를 얼마나 따라갔는지로,
      0에 가까울수록 좋습니다. <b>실현 차이</b> = 보수 후 − 지수. 보수까지 낸 뒤
      실제로 받은 결과입니다. 통화 변환은 자산별 통화 토글을 따릅니다.
    </p>
    <div style="overflow-x:auto;">
    <table style="font-size:.8rem;min-width:100%;">
      <tr><th>펀드</th><th>추적 지수</th><th>총보수</th>
          <th>보수 후</th><th>보수 전</th><th>지수</th>
          <th>추적 차이</th><th>실현 차이</th></tr>
      ${rows}
    </table>
    </div>${hint}`;
}

// ── Shared drag-to-select plumbing ──
// Paints the selection band over the plot area only (never the axis gutters) and hands the
// picked date range to `onSelect`, which returns false when it has nothing to show — the
// band is dropped in that case so a fruitless drag never leaves shading behind.
// The pointer is captured on press, so releasing outside the canvas still completes the
// selection instead of silently aborting it.
function attachDragCore(canvasId, overlayId, statsId, chartRef, onSelect) {
  const canvas = document.getElementById(canvasId);
  const overlay = document.getElementById(overlayId);
  const statsBox = document.getElementById(statsId);
  if (!canvas || !overlay || !statsBox) return null;

  canvas.style.cursor = 'crosshair';
  canvas.style.touchAction = 'none';   // let the pointer drag win over touch scrolling

  let dragStart = null, dragging = false;

  function clear() {
    overlay.style.display = 'none'; statsBox.style.display = 'none';
    dragStart = null; dragging = false;
  }
  // Chart.js x values are local-midnight timestamps; toISOString would shift them a day
  // back in KST, so format the date in local time.
  function localDate(ms) {
    const d = new Date(ms), p = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
  }
  function posX(e, chart) {
    const rect = canvas.getBoundingClientRect();
    return Math.max(chart.chartArea.left, Math.min(e.clientX - rect.left, chart.chartArea.right));
  }
  function paint(chart, x) {
    const a = chart.chartArea;
    overlay.style.top = a.top + 'px';
    overlay.style.height = a.height + 'px';
    overlay.style.left = Math.min(dragStart, x) + 'px';
    overlay.style.width = Math.abs(x - dragStart) + 'px';
  }

  canvas.addEventListener('pointerdown', (e) => {
    const chart = chartRef();
    if (!chart || e.button !== 0) return;
    // A press anywhere on the canvas starts a drag — x is clamped to the plot, so grabbing
    // the y-axis gutter works too.
    dragStart = posX(e, chart); dragging = true;
    overlay.style.display = 'block'; paint(chart, dragStart);
    statsBox.style.display = 'none';
    try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
    e.preventDefault();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const chart = chartRef();
    if (!chart) { clear(); return; }
    paint(chart, posX(e, chart));
  });
  canvas.addEventListener('pointerup', (e) => {
    if (!dragging) return;
    dragging = false;
    const chart = chartRef();
    if (!chart) { clear(); return; }
    const x = posX(e, chart);
    if (Math.abs(x - dragStart) < 5) { clear(); return; }   // a plain click clears
    const scale = chart.scales.x;
    const d1 = localDate(scale.getValueForPixel(Math.min(dragStart, x)));
    const d2 = localDate(scale.getValueForPixel(Math.max(dragStart, x)));
    if (!onSelect(d1, d2)) clear();
  });
  canvas.addEventListener('pointercancel', clear);
  return clear;
}

// Index range of [d1, d2] within a sorted date array; null when too thin to summarize.
function selectionRange(fullDates, d1, d2) {
  if (!fullDates || fullDates.length === 0) return null;
  const si = fullDates.findIndex(d => d >= d1);
  let ei = -1;
  for (let i = fullDates.length - 1; i >= 0; i--) { if (fullDates[i] <= d2) { ei = i; break; } }
  if (si < 0 || ei < 0 || ei - si < 2) return null;
  return [si, ei];
}

// Drag-select for comparison chart
attachDragCore('comparison-chart', 'comp-overlay', 'comp-stats', () => comparisonChart, (d1, d2) => {
  const range = selectionRange(compFullDates, d1, d2);
  if (!range) return false;
  const [si, ei] = range;
  const statsBox = document.getElementById('comp-stats');
  const pc = v => +v > 0 ? 'positive' : +v < 0 ? 'negative' : '';
  let html = `<div style="font-weight:600;margin-bottom:0.3rem;">${compFullDates[si]} ~ ${compFullDates[ei]}</div>`;
  compSelectedIdxs.forEach((idx, ai) => {
    const nav = compFullNavs[ai];
    const ret = ((nav[ei] / nav[si] - 1) * 100).toFixed(2);
    const color = COMPARISON_COLORS[idx % COMPARISON_COLORS.length];
    html += `<div><span style="display:inline-block;width:10px;height:10px;background:${color};border-radius:2px;margin-right:4px;"></span>${FUNDS[idx].shortName || FUNDS[idx].name}: <b class="${pc(ret)}">${+ret > 0 ? '+' : ''}${ret}%</b></div>`;
  });
  statsBox.innerHTML = html;
  statsBox.style.display = 'block';
  return true;
});

function renderChart(canvasId, labels, data, color, opts) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{
      label: opts.label || '', data,
      borderColor: color, backgroundColor: opts.bg || 'rgba(0,0,0,0.08)',
      fill: true, pointRadius: 0, borderWidth: 1.5,
    }]},
    options: { responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: 'time', time: { unit: 'year' }, ticks: { maxTicksLimit: 8 } },
        y: opts.yOpts || { beginAtZero: false }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// Initialize trailing return section for individual fund
const fundTrailingCharts = {};
function initFundTrailing(prefix, dailyData) {
  const container = document.querySelector(`.trailing-section[data-prefix="${prefix}"]`);
  if (!container || !dailyData || dailyData.dates.length < 365) return;

  // Reconstruct NAV from daily returns (start=1000)
  const nav = [1000];
  for (let i = 0; i < dailyData.returns.length; i++) nav.push(nav[nav.length-1] * (1 + dailyData.returns[i]));
  const dates = [(() => { const d = new Date(dailyData.dates[0]); d.setDate(d.getDate()-1); return d.toISOString().slice(0,10); })(), ...dailyData.dates];
  const n = nav.length;
  const totalYears = (new Date(dates[n-1]) - new Date(dates[0])) / (365.25 * 86400000);
  const maxWindow = Math.floor(totalYears);
  if (maxWindow < 1) return;

  const windows = [];
  for (let y = 1; y <= Math.min(maxWindow, 10); y++) windows.push(y);

  const uid = prefix.replace(/[^a-z0-9]/gi, '_');
  const chips = windows.map(y =>
    `<label class="filter-chip${y === 1 ? ' active' : ''}" data-window="${y}"><input type="radio" name="tr-${uid}" value="${y}" ${y===1?'checked':''} style="display:none">${y}Y</label>`
  ).join('');

  container.innerHTML = `
    <h3>Rolling Trailing Returns</h3>
    <div class="filter-chips" style="margin-bottom:0.8rem;">${chips}</div>
    <div class="metrics-grid" id="tr-metrics-${uid}"></div>
    <div class="chart-container" style="height:220px;"><canvas id="tr-chart-${uid}"></canvas></div>`;

  function showWindow(wy) {
    const returns = [], rDates = [];
    for (let i = 0; i < n; i++) {
      const sd = new Date(dates[i]), ed = new Date(sd);
      ed.setFullYear(ed.getFullYear() + wy);
      const es = ed.toISOString().slice(0,10);
      let ei = -1;
      for (let j = i+1; j < n; j++) { if (dates[j] >= es) { ei = j; break; } }
      if (ei < 0) break;
      returns.push((Math.pow(nav[ei]/nav[i], 1/wy) - 1) * 100);
      rDates.push(dates[i]);
    }
    if (returns.length === 0) return;

    const avg = returns.reduce((s,v)=>s+v,0)/returns.length;
    const vari = returns.reduce((s,v)=>s+(v-avg)**2,0)/(returns.length-1);
    const std = Math.sqrt(vari), se = std/Math.sqrt(returns.length);
    const sorted = [...returns].sort((a,b)=>a-b);
    const median = sorted[Math.floor(sorted.length/2)];
    const min = sorted[0], max = sorted[sorted.length-1];
    const winRate = (returns.filter(r=>r>0).length/returns.length*100);

    const pc = v => v>0?'positive':v<0?'negative':'';
    const fp = (v,s) => (s&&v>0?'+':'')+v.toFixed(2)+'%';

    document.getElementById('tr-metrics-'+uid).innerHTML = `
      <div class="metric-card"><div class="label">관측수</div><div class="value">${returns.length}</div></div>
      <div class="metric-card"><div class="label">평균 CAGR</div><div class="value ${pc(avg)}">${fp(avg,1)}</div></div>
      <div class="metric-card"><div class="label">중앙값</div><div class="value ${pc(median)}">${fp(median,1)}</div></div>
      <div class="metric-card"><div class="label">표준편차</div><div class="value">${std.toFixed(2)}%</div></div>
      <div class="metric-card"><div class="label">표준오차</div><div class="value">${se.toFixed(2)}%</div></div>
      <div class="metric-card"><div class="label">최소</div><div class="value ${pc(min)}">${fp(min,1)}</div></div>
      <div class="metric-card"><div class="label">최대</div><div class="value ${pc(max)}">${fp(max,1)}</div></div>
      <div class="metric-card"><div class="label">양수 비율</div><div class="value ${winRate>50?'positive':'negative'}">${winRate.toFixed(1)}%</div></div>`;

    const step = Math.max(1, Math.floor(rDates.length/400));
    const cd = rDates.filter((_,i)=>i%step===0), cr = returns.filter((_,i)=>i%step===0);

    if (fundTrailingCharts[uid]) fundTrailingCharts[uid].destroy();
    fundTrailingCharts[uid] = new Chart(document.getElementById('tr-chart-'+uid), {
      type:'line',
      data:{labels:cd,datasets:[
        {label:wy+'Y CAGR (%)',data:cr.map(v=>+v.toFixed(2)),borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,0.12)',fill:true,pointRadius:0,borderWidth:1.5},
        {label:'평균',data:cd.map(()=>+avg.toFixed(2)),borderColor:'#888',borderDash:[5,5],pointRadius:0,borderWidth:1},
        {label:'0%',data:cd.map(()=>0),borderColor:'#ef4444',borderDash:[3,3],pointRadius:0,borderWidth:1},
      ]},
      options:{responsive:true,maintainAspectRatio:false,
        scales:{x:{type:'time',time:{unit:'year'},ticks:{maxTicksLimit:8}},y:{ticks:{callback:v=>v+'%'}}},
        plugins:{legend:{display:true,position:'top',labels:{boxWidth:12,font:{size:11}}}}}
    });
  }

  container.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      container.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      showWindow(+chip.dataset.window);
    });
  });
  showWindow(1);
}

// Reusable drag-to-select for any NAV chart
function attachDragSelect(canvasId, overlayId, statsId, chartRef, fullDatesOrFn, fullNavOrFn) {
  const statsBox = document.getElementById(statsId);
  if (!statsBox) return;

  function showStats(d1, d2) {
    const fullDates = typeof fullDatesOrFn === 'function' ? fullDatesOrFn() : fullDatesOrFn;
    const fullNav = typeof fullNavOrFn === 'function' ? fullNavOrFn() : fullNavOrFn;
    const range = selectionRange(fullDates, d1, d2);
    if (!range) return false;
    const [si, ei] = range;
    const dates = fullDates.slice(si, ei+1), nav = fullNav.slice(si, ei+1), n = nav.length;
    const totalDays = (new Date(dates[n-1]) - new Date(dates[0])) / 86400000;
    const totalYears = totalDays / 365.25;
    const totalReturn = ((nav[n-1]/nav[0]-1)*100).toFixed(2);
    const cagr = totalYears > 0 ? ((Math.pow(nav[n-1]/nav[0],1/totalYears)-1)*100).toFixed(2) : '-';
    let volRaw = monthlyVol(dates, nav);
    if (volRaw === null) {
      const dr = []; for (let i=1;i<n;i++) dr.push(nav[i]/nav[i-1]-1);
      const mean = dr.reduce((s,v)=>s+v,0)/dr.length;
      const vari = dr.reduce((s,v)=>s+(v-mean)**2,0)/(dr.length-1);
      volRaw = Math.sqrt(vari)*Math.sqrt(totalYears > 0 ? dr.length/totalYears : 252);
    }
    const vol = (volRaw*100).toFixed(2);
    let peak = nav[0], mdd = 0;
    for (const v of nav) { peak = Math.max(peak,v); mdd = Math.min(mdd,(v-peak)/peak); }
    const pc = v => +v>0?'positive':+v<0?'negative':'';
    statsBox.innerHTML =
      `<div style="font-weight:600;margin-bottom:0.3rem;">${dates[0]} ~ ${dates[n-1]}</div>`+
      `<div>수익률: <b class="${pc(totalReturn)}">${+totalReturn>0?'+':''}${totalReturn}%</b></div>`+
      `<div>CAGR: <b class="${pc(cagr)}">${+cagr>0?'+':''}${cagr}%</b></div>`+
      `<div>변동성(월간): <b>${vol}%</b></div>`+
      `<div>MDD: <b class="negative">${(mdd*100).toFixed(2)}%</b></div>`;
    statsBox.style.display = 'block';
    return true;
  }

  attachDragCore(canvasId, overlayId, statsId, chartRef, showStats);
}

const fundCharts = {};
const fundSeries = {};   // chart prefix -> { dates, nav } backing the drag-select readout
const dragBound = new Set();

function rebuildNav(dailyData) {
  const nav = [1000];
  for (let i = 0; i < dailyData.returns.length; i++) nav.push(nav[nav.length-1] * (1 + dailyData.returns[i]));
  const d0 = new Date(dailyData.dates[0]); d0.setDate(d0.getDate()-1);
  return { dates: [d0.toISOString().slice(0,10), ...dailyData.dates], nav };
}

// The blocks a fund renders in the detail tab — one per available currency view.
function fundVariants(idx) {
  const f = FUNDS[idx];
  const out = [{ key: 'orig', data: f, label: '기준가' }];
  if (f.usd) out.push({ key: 'usd-conv', data: f.usd, label: '기준가 (USD)' });
  if (f.krw) out.push({ key: 'krw', data: f.krw, label: '기준가 (KRW)' });
  if (f.jpy) out.push({ key: 'jpy', data: f.jpy, label: '기준가 (JPY)' });
  return out;
}

// Full-resolution NAV at real 기준가 levels: chart.nav[0] is the true starting value and
// the daily returns carry it forward, so any slice keeps its actual level (unlike
// rebuildNav, which restarts every series at 1000).
function rebuildActual(data) {
  const nav = [data.chart.nav[0]];
  for (const r of data.daily.returns) nav.push(nav[nav.length - 1] * (1 + r));
  const d0 = new Date(data.daily.dates[0]); d0.setDate(d0.getDate() - 1);
  return { dates: [d0.toISOString().slice(0, 10), ...data.daily.dates], nav };
}

function destroyChart(id) { const c = Chart.getChart(id); if (c) c.destroy(); }

// Draw one variant's NAV + drawdown pair. Chart.js refuses a canvas that already holds a
// chart, so both are torn down first.
function drawVariantCharts(idx, v, dates, nav, ddPct) {
  const prefix = `chart-${idx}-${v.key}`;
  destroyChart(`${prefix}-nav`);
  destroyChart(`${prefix}-dd`);
  fundCharts[`${idx}-${v.key}`] = renderChart(`${prefix}-nav`, dates, nav,
    '#3b82f6', { label: v.label, bg: 'rgba(59,130,246,0.12)' });
  renderChart(`${prefix}-dd`, dates, ddPct,
    '#ef4444', { label: '드로다운 (%)', bg: 'rgba(239,68,68,0.16)', yOpts: { max: 0 } });
}

// Back to the Python-rendered view: its own downsampled chart payload and full history.
function paintVariantFull(idx, v) {
  const prefix = `chart-${idx}-${v.key}`;
  drawVariantCharts(idx, v, v.data.chart.dates, v.data.chart.nav, v.data.chart.drawdown);
  fundSeries[prefix] = rebuildActual(v.data);
  initFundTrailing(prefix, v.data.daily);
}

function createSingleChart(idx) {
  fundVariants(idx).forEach(v => {
    const prefix = `chart-${idx}-${v.key}`;
    paintVariantFull(idx, v);
    // Bind once — the readout reads fundSeries, so it follows later repaints on its own.
    if (!dragBound.has(prefix)) {
      attachDragSelect(`${prefix}-nav`, `${prefix}-overlay`, `${prefix}-stats`,
        () => fundCharts[`${idx}-${v.key}`],
        () => fundSeries[prefix].dates, () => fundSeries[prefix].nav);
      dragBound.add(prefix);
    }
  });
}

// ── 개별 자산 상세: period ──
// The blocks here are server-rendered from Python, so 전체 puts that markup back rather
// than recomputing it — the default view stays the one Python vouched for.
const detailSnapshots = {};
const DETAIL_PARTS = ['metrics', 'ddtable', 'lstable'];

function snapshotVariant(prefix) {
  DETAIL_PARTS.forEach(part => {
    const id = `${prefix}-${part}`;
    if (id in detailSnapshots) return;
    const el = document.getElementById(id);
    detailSnapshots[id] = el ? el.innerHTML : '';
  });
}

function restoreVariant(idx, v) {
  const prefix = `chart-${idx}-${v.key}`;
  DETAIL_PARTS.forEach(part => {
    const id = `${prefix}-${part}`;
    const el = document.getElementById(id);
    if (el && id in detailSnapshots) el.innerHTML = detailSnapshots[id];
  });
  paintVariantFull(idx, v);
}

function visibleFundIdxs() {
  return [...document.querySelectorAll('#panel-detail .fund-section')]
    .filter(s => /^fund-\d+$/.test(s.id) && !s.classList.contains('hidden'))
    .map(s => +s.id.slice(5));
}

function paintVariantRanged(idx, v) {
  const prefix = `chart-${idx}-${v.key}`;
  snapshotVariant(prefix);
  const metricsEl = document.getElementById(`${prefix}-metrics`);
  const ddEl = document.getElementById(`${prefix}-ddtable`);
  const lsEl = document.getElementById(`${prefix}-lstable`);
  const trailEl = document.querySelector(`.trailing-section[data-prefix="${prefix}"]`);

  const full = rebuildActual(v.data);
  const r = detailPeriod.range();
  const keep = [];
  for (let i = 0; i < full.dates.length; i++) {
    const d = full.dates[i];
    if ((!r.start || d >= r.start) && (!r.end || d <= r.end)) keep.push(i);
  }
  const dates = keep.map(i => full.dates[i]);
  const nav = keep.map(i => full.nav[i]);
  const m = dates.length >= 2 ? calcMetrics(dates, nav) : null;

  if (trailEl) trailEl.innerHTML = '';
  if (!m) {
    destroyChart(`${prefix}-nav`);
    destroyChart(`${prefix}-dd`);
    fundSeries[prefix] = { dates: dates.length ? dates : full.dates, nav: nav.length ? nav : full.nav };
    if (metricsEl) metricsEl.innerHTML =
      `<p class="fund-meta" style="margin:0;">선택 기간에 이 자산의 데이터가 없습니다 (${dates.length}거래일).</p>`;
    if (ddEl) ddEl.innerHTML = '';
    if (lsEl) lsEl.innerHTML = '';
    return;
  }
  fundSeries[prefix] = { dates, nav };

  // Python averages the drawdown summary over every event but tables only the worst 5.
  const allEvents = findDrawdowns(dates, nav, Infinity);
  const top = allEvents.slice(0, 5);
  const avgDd = allEvents.length
    ? (allEvents.reduce((s, e) => s + Math.abs(+e.depth), 0) / allEvents.length).toFixed(2) : '0.00';
  const longest = allEvents.length ? allEvents.reduce((a, b) => b.days > a.days ? b : a, allEvents[0]) : null;

  // Same bar as the comparison table: annualising a sub-year window invents a headline.
  const totalDays = (new Date(dates[dates.length - 1]) - new Date(dates[0])) / 86400000;
  const annualisable = totalDays >= 350;
  const pctCls = v2 => +v2 > 0 ? 'positive' : +v2 < 0 ? 'negative' : '';
  const fmtPct = (v2, sign) => (sign && +v2 > 0 ? '+' : '') + v2 + '%';
  const dash = '<div class="value" title="기간이 1년 미만이라 연환산하지 않습니다">—</div>';

  if (metricsEl) metricsEl.innerHTML = `
    <div class="metric-card"><div class="label">기간</div><div class="value">${m.totalYears}년</div></div>
    <div class="metric-card"><div class="label">총 수익률</div><div class="value ${pctCls(m.totalReturn)}">${fmtPct(m.totalReturn, true)}</div></div>
    <div class="metric-card"><div class="label">CAGR</div>${annualisable ? `<div class="value ${pctCls(m.cagr)}">${fmtPct(m.cagr, true)}</div>` : dash}</div>
    <div class="metric-card"><div class="label">변동성 (월간)</div><div class="value">${m.volatility}%</div></div>
    <div class="metric-card"><div class="label">샤프비율</div>${annualisable ? `<div class="value">${m.sharpe}</div>` : dash}</div>
    <div class="metric-card"><div class="label">MDD</div><div class="value negative">${m.mdd}%</div></div>
    <div class="metric-card"><div class="label">평균 하락폭</div><div class="value negative">-${avgDd}%</div></div>
    <div class="metric-card"><div class="label">최장 하락 기간</div><div class="value">${longest ? longest.days.toLocaleString() : 0}일</div></div>`;

  const step = Math.max(1, Math.floor(dates.length / 500));
  drawVariantCharts(idx, v,
    dates.filter((_, i) => i % step === 0),
    nav.filter((_, i) => i % step === 0).map(x => +x.toFixed(2)),
    m.drawdownSeries.filter((_, i) => i % step === 0).map(x => +(x * 100).toFixed(2)));

  if (ddEl) {
    ddEl.innerHTML = top.length === 0 ? '' : `
      <h3>주요 하락 이벤트 (Top ${top.length})</h3>
      <table><tr><th>#</th><th>시작</th><th>저점</th><th>회복</th><th>하락폭</th><th>기간</th></tr>` +
      top.map((e, i) =>
        `<tr><td>${i + 1}</td><td>${e.start}</td><td>${e.trough}</td>` +
        `<td>${e.end || '<span class="ongoing">진행중</span>'}</td>` +
        `<td class="negative">${e.depth}%</td><td>${e.days.toLocaleString()}일</td></tr>`).join('') +
      '</table>';
  }

  const lsDca = [3, 12, 36].map(w => calcLsDca(nav, dates, w)).filter(Boolean);
  if (lsEl) {
    lsEl.innerHTML = lsDca.length === 0
      ? '<p style="color:var(--subtle);">데이터 부족으로 LS vs DCA 분석 불가</p>'
      : `<h3>LS vs DCA 분석</h3>
         <table><tr><th>기간</th><th>관측수</th><th>LS 승률</th><th>MLSA</th><th>MLSD</th></tr>` +
        lsDca.map(x =>
          `<tr><td>${x.window}개월</td><td>${x.observations.toLocaleString()}</td>` +
          `<td class="${+x.winRate > 50 ? 'positive' : 'negative'}">${x.winRate}%</td>` +
          `<td class="${+x.mlsa > 0 ? 'positive' : 'negative'}">${+x.mlsa > 0 ? '+' : ''}${x.mlsa}%</td>` +
          `<td class="negative">${x.mlsd}%</td></tr>`).join('') + '</table>';
  }

  // Rolling windows need well over a year of history; the picker can easily cut below
  // that, and initFundTrailing just bails — so leave a reason behind for it to overwrite.
  const clipped = { dates: [], returns: [] };
  keep.forEach(i => { if (i >= 1) { clipped.dates.push(full.dates[i]); clipped.returns.push(v.data.daily.returns[i - 1]); } });
  if (trailEl) trailEl.innerHTML =
    '<h3>Rolling Trailing Returns</h3><p class="fund-meta" style="margin:0;">' +
    '선택 기간이 짧아 롤링 구간을 만들 수 없습니다 (1년 이상 필요).</p>';
  initFundTrailing(prefix, clipped);
}

function applyDetailRange() {
  const narrowed = detailPeriod.narrowed();
  visibleFundIdxs().forEach(idx => {
    const section = document.getElementById('fund-' + idx);
    if (!section || !section._chartsCreated) return;
    const meta = document.getElementById(`fund-${idx}-meta`);
    if (narrowed) {
      fundVariants(idx).forEach(v => paintVariantRanged(idx, v));
      // The header still advertises the full history — say which slice is on screen.
      const r = detailPeriod.range();
      if (meta) meta.textContent = `${meta.dataset.full} → 선택 기간 ${r.start || '처음'} ~ ${r.end || '끝'}`;
      section._ranged = true;
    } else if (section._ranged) {
      fundVariants(idx).forEach(v => restoreVariant(idx, v));
      if (meta) meta.textContent = meta.dataset.full;
      section._ranged = false;
    }
  });
}

function refreshDetailPeriod() {
  const idxs = visibleFundIdxs();
  if (idxs.length === 0) { detailPeriod.hide(); return; }
  // No common-period requirement here: each asset is shown on its own, so the picker
  // spans the union and every block clips to whatever it actually has.
  let lo = null, hi = null;
  idxs.forEach(i => {
    const d = FUNDS[i].daily.dates;
    if (lo === null || d[0] < lo) lo = d[0];
    if (hi === null || d[d.length - 1] > hi) hi = d[d.length - 1];
  });
  detailPeriod.setBounds(lo, hi);
  applyDetailRange();
}

const detailPeriod = makePeriodPicker({
  card: 'detail-period', info: 'detail-period-info', presets: 'detail-period-presets',
  years: 'detail-period-years', start: 'detail-start', end: 'detail-end',
  infoLabel: '전체 기간',
}, () => refreshDetailPeriod());

// ══ 추세 신호 (Faber TAA) ══════════════════════════════════════════════
// 규칙: 기준가 > 이동평균 → 보유, 아니면 현금. 이동평균은 언제나 전체 이력으로 계산하고
// 표시·백테스트 구간만 잘라낸다 — 구간 시작 200일치를 못 채워 신호가 비는 걸 막기 위해서다.

const taaState = { win: 200, cadence: 'month', slope: true, buffer: 0, epDepth: -20 };
const taaCharts = {};

// 매도 구간을 차트 배경에 칠한다. annotation 플러그인을 쓰지 않으려고 직접 그린다.
Chart.register({
  id: 'taaBands',
  beforeDatasetsDraw(chart) {
    const bands = (chart.options.plugins.taaBands || {}).bands;
    if (!bands || !bands.length) return;
    const { ctx, chartArea: a, scales: { x } } = chart;
    ctx.save();
    ctx.fillStyle = getComputedStyle(document.documentElement)
      .getPropertyValue('--taa-band') || 'rgba(207,42,30,.08)';
    bands.forEach(([s, e]) => {
      const l = Math.max(x.getPixelForValue(new Date(s).getTime()), a.left);
      const r = Math.min(x.getPixelForValue(new Date(e).getTime()), a.right);
      if (r > l) ctx.fillRect(l, a.top, r - l, a.bottom - a.top);
    });
    ctx.restore();
  },
});

// 전체 해상도 기준가. chart.nav 는 500포인트로 다운샘플된 값이라 이동평균 계산에 못 쓴다.
// 대신 chart.nav[0](실제 첫 기준가)에서 일별 수익률을 복리로 쌓아 매 거래일을 되살린다.
function taaLevels(fund, mode) {
  const daily = getDataByMode(fund, mode, 'daily');
  const chart = getDataByMode(fund, mode, 'chart');
  if (!daily || !chart || !daily.dates.length) return null;
  const nav = [chart.nav[0]];
  for (let i = 0; i < daily.returns.length; i++) nav.push(nav[nav.length - 1] * (1 + daily.returns[i]));
  return { dates: [chart.dates[0], ...daily.dates], nav };
}

function smaSeries(nav, w) {
  const out = new Array(nav.length).fill(null);
  let sum = 0;
  for (let i = 0; i < nav.length; i++) {
    sum += nav[i];
    if (i >= w) sum -= nav[i - w];
    if (i >= w - 1) out[i] = sum / w;
  }
  return out;
}

// 이동평균의 1일 변화는 (P_t - P_{t-w}) / w 라서, 그 부호는 w일 모멘텀의 부호와 정확히 같다.
// 기울기 필터를 "200일 전보다 높은가"로 구현하는 근거이자, 화면에서 둘을 나란히 두는 이유.
function taaCompute(series, st) {
  const { dates, nav } = series, n = nav.length, w = st.win;
  const ma = smaSeries(nav, w);
  const gap = new Array(n).fill(null);     // 이격도 %
  const slope = new Array(n).fill(null);   // 이동평균 1일 기울기, 연율 %
  const want = new Array(n).fill(null);    // 그날의 규칙 판정
  for (let i = 0; i < n; i++) {
    if (ma[i] == null) continue;
    gap[i] = (nav[i] / ma[i] - 1) * 100;
    if (ma[i - 1] != null) slope[i] = (ma[i] / ma[i - 1] - 1) * 100 * 252;
    let ok = nav[i] > ma[i] * (1 + st.buffer / 100);
    // 밴드 안쪽이면 직전 상태를 유지 — 여기서 want 를 확정하지 않고 뒤에서 이어받는다.
    if (nav[i] < ma[i] * (1 - st.buffer / 100)) ok = false;
    else if (!ok) ok = null;
    if (ok && st.slope && !(i >= w && nav[i] > nav[i - w])) ok = false;
    want[i] = ok;
  }

  const firstValid = ma.findIndex(v => v != null);

  // 그날그날의 규칙 판정. 밴드 안쪽(want === null)이면 직전 판정을 그대로 물고 간다.
  const daily = new Array(n).fill(null);
  let cur = null;
  for (let i = firstValid; i < n; i++) {
    if (want[i] !== null) cur = want[i];
    if (cur === null) cur = nav[i] > ma[i];   // 밴드 안에서 시작한 경우의 초기값
    daily[i] = cur;
  }

  // 그달의 마지막 거래일인가. 마지막 데이터가 실제 월말이 아니면 확정일로 치지 않는다 —
  // 아직 안 끝난 달의 중간 시세로 신호가 떴다고 표시하면 없는 신호를 만들어내는 셈이다.
  const isMonthEnd = i => {
    if (i + 1 < n) return dates[i + 1].slice(0, 7) !== dates[i].slice(0, 7);
    const d = new Date(dates[i] + 'T00:00:00'), m = d.getMonth();
    d.setDate(d.getDate() + 1);
    return d.getMonth() !== m;
  };

  // 판정 주기 적용. 월말 규칙은 그달 마지막 거래일에만 상태를 바꾼다 (Faber 원문의 월말 체크).
  const pos = new Array(n).fill(null);
  let held = null;
  for (let i = firstValid; i < n; i++) {
    if (held === null || st.cadence === 'day' || isMonthEnd(i)) held = daily[i];
    pos[i] = held;
  }
  // 월말 모드에서 이번 달이 아직 안 끝났다면, 오늘 기준 판정은 "잠정"으로만 보여준다.
  const settled = st.cadence === 'day' || isMonthEnd(n - 1);
  return { dates, nav, ma, gap, slope, pos, daily, firstValid, settled };
}

// 실제 매매는 신호 다음 거래일 기준가로 체결된다고 본다 (기준가 반영 지연 T+1).
function taaBacktest(t, si, ei) {
  let eq = 1, bh = 1, peak = 1, mdd = 0, bpeak = 1, bmdd = 0, trades = 0, inMkt = 0, days = 0;
  const eqCurve = [], bhCurve = [];
  for (let i = si; i <= ei; i++) {
    if (i > si) {
      const r = t.nav[i] / t.nav[i - 1] - 1;
      const held = !!t.pos[i - 1];
      eq *= 1 + (held ? r : 0);
      bh *= 1 + r;
      if (t.pos[i] !== t.pos[i - 1]) trades++;
      inMkt += held ? 1 : 0;
      days++;
    }
    peak = Math.max(peak, eq); mdd = Math.min(mdd, eq / peak - 1);
    bpeak = Math.max(bpeak, bh); bmdd = Math.min(bmdd, bh / bpeak - 1);
    eqCurve.push(eq * 100); bhCurve.push(bh * 100);
  }
  const yrs = (new Date(t.dates[ei]) - new Date(t.dates[si])) / (365.25 * 86400000);
  return {
    eqCurve, bhCurve, trades, exposure: days ? inMkt / days * 100 : 0,
    taaCagr: yrs > 0 ? (Math.pow(eq, 1 / yrs) - 1) * 100 : null,
    bhCagr: yrs > 0 ? (Math.pow(bh, 1 / yrs) - 1) * 100 : null,
    taaMdd: mdd * 100, bhMdd: bmdd * 100, years: yrs,
  };
}

// 신호별로 "그 판단이 옳았나"를 뒤에서 채점한다. 매도 구간은 기준가가 내렸으면 적중,
// 보유 구간은 올랐으면 적중 — 신호가 실제로 하락을 걸러줬는지를 그대로 보여준다.
function taaEvents(t, si, ei) {
  const ev = [];
  for (let i = Math.max(si, t.firstValid + 1); i <= ei; i++) {
    if (t.pos[i] === t.pos[i - 1]) continue;
    ev.push({ i, date: t.dates[i], buy: t.pos[i], nav: t.nav[i], gap: t.gap[i], slope: t.slope[i] });
  }
  ev.forEach((e, k) => {
    const end = k + 1 < ev.length ? ev[k + 1].i : ei;
    e.endDate = t.dates[end];
    e.days = Math.round((new Date(t.dates[end]) - new Date(e.date)) / 86400000);
    e.move = (t.nav[end] / t.nav[e.i] - 1) * 100;   // 구간 중 기준가 변화
    e.hit = e.buy ? e.move > 0 : e.move < 0;
    e.open = k === ev.length - 1;
  });
  return ev;
}

// ── 준거점 ──────────────────────────────────────────────────────────
// "정말 팔았어야 했던 때"를 눈대중으로 고르면 채점이 사후 합리화가 된다. 그래서 규칙과
// 무관하게 기준가 자체에서 큰 하락 구간(고점→저점→회복)을 뽑아 준거점으로 쓴다.
// 고점에 팔고 저점에 사는 게 이론적 상한(회피율·참여율 100%)이고, 규칙은 그 대비로 채점된다.
function taaEpisodes(t, minDepth) {
  const { nav } = t, n = nav.length, out = [];
  let peak = nav[0], peakIdx = 0, inDd = false, trIdx = 0, trVal = 0;
  for (let i = 0; i < n; i++) {
    if (nav[i] >= peak) {
      if (inDd && trVal <= minDepth) out.push({ peak: peakIdx, trough: trIdx, end: i, depth: trVal * 100 });
      inDd = false; peak = nav[i]; peakIdx = i;
      continue;
    }
    const d = (nav[i] - peak) / peak;
    if (!inDd) { inDd = true; trIdx = i; trVal = d; }
    else if (d < trVal) { trIdx = i; trVal = d; }
  }
  if (inDd && trVal <= minDepth) out.push({ peak: peakIdx, trough: trIdx, end: null, depth: trVal * 100 });
  return out;
}

function taaScore(t, ep, ei) {
  const end = ep.end == null || ep.end > ei ? ei : ep.end;
  const seg = (a, b, useRule) => {
    let v = 1;
    for (let i = a + 1; i <= b; i++) {
      const r = t.nav[i] / t.nav[i - 1] - 1;
      v *= 1 + (!useRule || t.pos[i - 1] ? r : 0);
    }
    return (v - 1) * 100;
  };
  const bhFall = seg(ep.peak, ep.trough, false), ruleFall = seg(ep.peak, ep.trough, true);
  const bhRise = seg(ep.trough, end, false), ruleRise = seg(ep.trough, end, true);
  let sell = null, buy = null;
  for (let i = ep.peak + 1; i <= ep.trough; i++) if (!t.pos[i] && t.pos[i - 1]) { sell = i; break; }
  for (let i = ep.trough + 1; i <= end; i++) if (t.pos[i] && !t.pos[i - 1]) { buy = i; break; }
  return {
    end, bhFall, ruleFall, bhRise, ruleRise, sell, buy,
    alreadyCash: sell === null && !t.pos[ep.peak],
    // 회피율: 보유가 잃은 것 중 규칙이 안 잃은 몫. 참여율: 반등 중 규칙이 먹은 몫.
    avoided: bhFall < 0 ? (1 - ruleFall / bhFall) * 100 : null,
    captured: bhRise > 0 ? ruleRise / bhRise * 100 : null,
  };
}

const taaFmt = (v, d) => v == null || !isFinite(v) ? '—' : v.toFixed(d == null ? 2 : d);
const taaSigned = (v, d) => v == null || !isFinite(v) ? '—'
  : `<span class="${v >= 0 ? 'positive' : 'negative'}">${v >= 0 ? '+' : ''}${v.toFixed(d == null ? 2 : d)}%</span>`;

function taaBadge(t, i) {
  const up = t.pos[i];
  const slopeNeg = t.slope[i] != null && t.slope[i] < 0;
  if (!up) return '<span class="sig-badge sig-cash">현금</span>';
  return slopeNeg ? '<span class="sig-badge sig-warn">보유 · 기울기 음전환</span>'
                  : '<span class="sig-badge sig-hold">보유</span>';
}

function taaScoreCard(t, si, ei) {
  const lo = Math.max(si, t.firstValid);
  const eps = taaEpisodes(t, taaState.epDepth / 100).filter(e => e.peak >= lo && e.trough <= ei);
  const head = `<h4 class="taa-sub">준거점 채점 — ${Math.abs(taaState.epDepth)}% 이상 하락 이벤트</h4>
    <p class="hint" style="margin-top:0;">고점에 팔고 저점에 사는 게 상한(회피율·참여율 100%).
       회피율은 보유가 잃은 낙폭 중 규칙이 피한 몫, 참여율은 반등 중 규칙이 먹은 몫입니다.</p>`;
  if (!eps.length) return head + `<p class="fund-meta">이 기간에 ${Math.abs(taaState.epDepth)}% 이상 하락 이벤트가 없습니다.</p>`;

  const sc = eps.map(e => ({ e, s: taaScore(t, e, ei) }));
  const rows = sc.map(({ e, s }) => {
    const when = (i, ref, label) => i === null
      ? `<span class="row-note">${label}</span>`
      : `${t.dates[i]}<br><span class="row-note">${ref >= 0 ? '+' : ''}${Math.round((new Date(t.dates[i]) - new Date(t.dates[ref])) / 86400000)}일</span>`;
    return `<tr>
      <td>${t.dates[e.peak]}<br><span class="row-note">저점 ${t.dates[e.trough]}</span></td>
      <td>${taaSigned(s.bhFall, 1)}</td>
      <td>${taaSigned(s.ruleFall, 1)}</td>
      <td><b>${s.avoided == null ? '—' : taaFmt(s.avoided, 0) + '%'}</b></td>
      <td>${when(s.sell, e.peak, s.alreadyCash ? '이미 현금' : '안 팔았음')}</td>
      <td>${s.captured == null ? '—' : taaFmt(s.captured, 0) + '%'}</td>
      <td>${when(s.buy, e.trough, t.pos[e.trough] ? '계속 보유' : '안 샀음')}</td>
    </tr>`;
  }).join('');

  const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  const mAvoid = avg(sc.map(x => x.s.avoided).filter(v => v != null));
  const mCapt = avg(sc.map(x => x.s.captured).filter(v => v != null));

  // 1차 필터로 쓸 만한지는 회피율이 아니라 재현율이 정한다. 신호가 안 뜬 이벤트는
  // 2차 매크로 분석으로도 되살릴 수 없다 — 분석할 대상 자체가 없으므로.
  const caught = sc.filter(x => x.s.sell !== null);
  const recall = caught.length / eps.length * 100;
  // 정밀도: 이 구간의 매도 신호 중 실제 하락 이벤트 안에서 나온 비율.
  let sells = 0, inside = 0;
  for (let i = lo + 1; i <= ei; i++) {
    if (t.pos[i] || !t.pos[i - 1]) continue;
    sells++;
    if (eps.some(e => i >= e.peak && i <= e.trough)) inside++;
  }
  // 신호가 떴을 때 저점까지 아직 남아 있던 낙폭. 이게 작으면 이미 늦은 신호다.
  const mRemain = avg(caught.map(x => (t.nav[eps[sc.indexOf(x)].trough] / t.nav[x.s.sell] - 1) * 100));

  return head + `<table style="font-size:.8rem;min-width:100%;">
      <tr><th>고점 / 저점</th><th>보유 낙폭</th><th>규칙 낙폭</th><th>회피율</th><th>매도 시점</th>
          <th>반등 참여율</th><th>매수 시점</th></tr>
      ${rows}
      <tr><td><b>평균 (${sc.length}건)</b></td><td></td><td></td>
          <td><b>${mAvoid == null ? '—' : taaFmt(mAvoid, 0) + '%'}</b></td><td></td>
          <td><b>${mCapt == null ? '—' : taaFmt(mCapt, 0) + '%'}</b></td><td></td></tr>
    </table>
    <div class="taa-facts">
      <div class="taa-fact"><span>재현율 (놓치지 않음)</span><b>${taaFmt(recall, 0)}%
        <span class="row-note">${caught.length}/${eps.length}건</span></b></div>
      <div class="taa-fact"><span>정밀도 (헛발 아님)</span><b>${sells ? taaFmt(inside / sells * 100, 0) + '%' : '—'}
        <span class="row-note">${inside}/${sells}회</span></b></div>
      <div class="taa-fact"><span>신호 시점에 남은 낙폭</span><b>${taaSigned(mRemain, 1)}</b></div>
    </div>
    <p class="hint">재현율은 2차 분석으로 못 올립니다 — 신호가 안 뜬 이벤트는 검토 대상에 아예 안 오릅니다.
       정밀도가 낮은 건 2차 분석으로 걸러낼 여지가 크다는 뜻이고요.
       "남은 낙폭"이 클수록 신호 시점이 아직 행동할 수 있는 자리였다는 의미입니다.</p>`;
}

function refreshTaa() {
  const selected = [];
  document.querySelectorAll('#filter-chips-insurance input:checked, #filter-chips-us input:checked, #filter-chips-jp input:checked, #filter-chips-index input:checked')
    .forEach(cb => selected.push(+cb.dataset.idx));

  const cfg = document.getElementById('taa-config');
  const statusCard = document.getElementById('taa-status-card');
  const box = document.getElementById('taa-assets');
  const empty = document.getElementById('taa-empty');

  Object.values(taaCharts).forEach(c => c && c.destroy());
  Object.keys(taaCharts).forEach(k => delete taaCharts[k]);

  if (!selected.length) {
    cfg.style.display = statusCard.style.display = 'none';
    taaPeriod.hide(); box.innerHTML = ''; empty.style.display = ''; return;
  }
  cfg.style.display = ''; empty.style.display = 'none';

  document.getElementById('taa-rule-info').textContent =
    `${taaState.win}일 이동평균 · ${taaState.cadence === 'month' ? '월말 판정' : '매일 판정'}`
    + (taaState.slope ? ' · 기울기 필터' : '') + (taaState.buffer ? ` · ±${taaState.buffer}% 밴드` : '');

  // 이동평균은 전체 이력으로 먼저 계산하고, 기간 선택은 그 뒤에 잘라내기만 한다.
  const built = selected.map(idx => {
    const s = taaLevels(FUNDS[idx], filterCurrencyState[idx] || 'krw');
    return s && s.nav.length > taaState.win + 20 ? { idx, t: taaCompute(s, taaState) } : { idx, t: null };
  });
  const usable = built.filter(b => b.t);

  if (!usable.length) {
    statusCard.style.display = 'none'; taaPeriod.hide();
    box.innerHTML = `<div class="empty">선택한 자산의 이력이 ${taaState.win}일 이동평균을 만들기에 부족합니다.</div>`;
    return;
  }

  let lo = null, hi = null;
  usable.forEach(({ t }) => {
    const d0 = t.dates[t.firstValid], d1 = t.dates[t.dates.length - 1];
    if (lo === null || d0 < lo) lo = d0;
    if (hi === null || d1 > hi) hi = d1;
  });
  taaPeriod.setBounds(lo, hi);
  const range = taaPeriod.range();

  statusCard.style.display = '';
  box.innerHTML = '';

  const rows = [], pending = [];
  usable.forEach(({ idx, t }) => {
    const n = t.dates.length;
    let si = Math.max(t.firstValid, 0), ei = n - 1;
    while (si < n && range.start && t.dates[si] < range.start) si++;
    while (ei > 0 && range.end && t.dates[ei] > range.end) ei--;
    if (ei - si < 20) return;

    const name = FUNDS[idx].shortName || FUNDS[idx].name;
    const color = COMPARISON_COLORS[idx % COMPARISON_COLORS.length];
    const bt = taaBacktest(t, si, ei);
    const ev = taaEvents(t, si, ei);
    const last = ev.length ? ev[ev.length - 1] : null;
    const closed = ev.filter(e => !e.open);
    const hits = closed.filter(e => e.hit).length;

    // 마지막 봉이 곧 오늘일 때만 "잠정 판정"이 의미가 있다. 과거 구간을 잘라 본 경우엔 숨긴다.
    const live = ei === t.dates.length - 1 && !t.settled;
    const flips = live && t.daily[ei] !== t.pos[ei];
    const pendCell = !live ? '<span class="row-note">확정</span>'
      : flips ? `<span class="sig-badge ${t.daily[ei] ? 'sig-hold' : 'sig-cash'}">${t.daily[ei] ? '매수' : '매도'}로 전환</span>`
              : '<span class="row-note">유지</span>';

    rows.push(`<tr><td>${name}</td><td>${taaBadge(t, ei)}</td>
      <td>${taaFmt(t.nav[ei])}</td><td>${taaFmt(t.ma[ei])}</td>
      <td>${taaSigned(t.gap[ei])}</td><td>${taaSigned(t.slope[ei], 1)}</td>
      <td>${pendCell}</td>
      <td>${last ? `${last.date}<br><span class="row-note">${last.buy ? '매수' : '매도'} · ${last.days}일 경과</span>` : '—'}</td></tr>`);

    const cid = `taa-c-${idx}`;
    const evRows = ev.slice(-8).reverse().map(e => `<tr>
      <td>${e.date}</td>
      <td>${e.buy ? '<span class="sig-badge sig-hold">매수</span>' : '<span class="sig-badge sig-cash">매도</span>'}</td>
      <td>${taaFmt(e.nav)}</td><td>${taaSigned(e.gap)}</td><td>${taaSigned(e.slope, 1)}</td>
      <td>${e.days}일${e.open ? ' <span class="row-note">진행 중</span>' : ''}</td>
      <td>${taaSigned(e.move)}</td>
      <td>${e.open ? '—' : (e.hit ? '<span class="positive">적중</span>' : '<span class="negative">헛발</span>')}</td>
    </tr>`).join('');

    box.insertAdjacentHTML('beforeend', `
    <section class="card taa-asset">
      <div class="taa-head"><h3>${name}</h3>${taaBadge(t, ei)}
        ${flips ? `<span class="sig-badge ${t.daily[ei] ? 'sig-hold' : 'sig-cash'}">월말 확정 시 ${t.daily[ei] ? '매수' : '매도'}로 전환</span>` : ''}
        <span class="fund-meta" style="margin:0;">${t.dates[si]} ~ ${t.dates[ei]}</span></div>
      <div class="taa-facts">
        <div class="taa-fact"><span>기준가</span><b>${taaFmt(t.nav[ei])}</b></div>
        <div class="taa-fact"><span>${taaState.win}일 이동평균</span><b>${taaFmt(t.ma[ei])}</b></div>
        <div class="taa-fact"><span>이격도</span><b>${taaSigned(t.gap[ei])}</b></div>
        <div class="taa-fact"><span>이동평균 기울기(연율)</span><b>${taaSigned(t.slope[ei], 1)}</b></div>
        <div class="taa-fact"><span>${taaState.win}일 모멘텀</span><b>${taaSigned(ei >= taaState.win ? (t.nav[ei] / t.nav[ei - taaState.win] - 1) * 100 : null)}</b></div>
        <div class="taa-fact"><span>시장 노출</span><b>${taaFmt(bt.exposure, 0)}%</b></div>
      </div>
      <div class="chart-container" style="height:300px;"><canvas id="${cid}"></canvas></div>
      <p class="hint">붉은 배경 = 규칙상 현금 구간</p>
      <div class="chart-container" style="height:150px;"><canvas id="${cid}-g"></canvas></div>
      <p class="hint">이격도(기준가 − 이동평균, %)와 이동평균 기울기(연율 %). 기울기 부호는 ${taaState.win}일 모멘텀 부호와 항상 일치합니다.</p>

      <h4 class="taa-sub">규칙 vs 그냥 보유 (${bt.years.toFixed(1)}년)</h4>
      <table style="font-size:.8rem;min-width:100%;">
        <tr><th></th><th>CAGR</th><th>MDD</th><th>거래</th><th>시장 노출</th></tr>
        <tr><td>추세 규칙</td><td>${taaSigned(bt.taaCagr)}</td><td>${taaSigned(bt.taaMdd, 1)}</td>
            <td>${bt.trades}회</td><td>${taaFmt(bt.exposure, 0)}%</td></tr>
        <tr><td>그냥 보유</td><td>${taaSigned(bt.bhCagr)}</td><td>${taaSigned(bt.bhMdd, 1)}</td>
            <td>—</td><td>100%</td></tr>
      </table>
      <p class="hint">보수·세금·펀드 변경 지연은 반영하지 않았습니다. 체결은 신호 다음 거래일 기준가로 가정.</p>

      ${taaScoreCard(t, si, ei)}

      <h4 class="taa-sub">최근 신호 ${closed.length ? `(종료된 ${closed.length}건 중 ${hits}건 적중 · ${(hits / closed.length * 100).toFixed(0)}%)` : ''}</h4>
      ${ev.length ? `<table style="font-size:.8rem;min-width:100%;">
        <tr><th>신호일</th><th>구분</th><th>기준가</th><th>이격도</th><th>기울기</th><th>지속</th><th>구간 등락</th><th>결과</th></tr>
        ${evRows}</table>` : '<p class="fund-meta">이 기간에 신호 전환이 없습니다.</p>'}
    </section>`);

    pending.push({ cid, t, si, ei, color });
  });

  document.getElementById('taa-status').innerHTML = rows.length ? `
    <table style="font-size:.82rem;min-width:100%;">
      <tr><th>자산</th><th>확정 상태</th><th>기준가</th><th>이동평균</th><th>이격도</th><th>기울기(연율)</th>
          <th>이번 달 말 판정</th><th>마지막 신호</th></tr>
      ${rows.join('')}
    </table>` : '<p class="fund-meta">선택 기간에 신호를 만들 데이터가 없습니다.</p>';

  pending.forEach(p => taaDrawCharts(p));
}

function taaDrawCharts({ cid, t, si, ei, color }) {
  const step = Math.max(1, Math.floor((ei - si + 1) / 700));
  const keep = [];
  for (let i = si; i <= ei; i += step) keep.push(i);
  if (keep[keep.length - 1] !== ei) keep.push(ei);
  const labels = keep.map(i => t.dates[i]);

  // 배경 밴드는 다운샘플 전 원본에서 뽑는다 — 짧은 현금 구간이 통째로 사라지지 않도록.
  const bands = [];
  let open = null;
  for (let i = si; i <= ei; i++) {
    if (!t.pos[i] && open === null) open = t.dates[i];
    if (t.pos[i] && open !== null) { bands.push([open, t.dates[i]]); open = null; }
  }
  if (open !== null) bands.push([open, t.dates[ei]]);

  const spanDays = (new Date(t.dates[ei]) - new Date(t.dates[si])) / 86400000;
  const unit = spanDays > 1500 ? 'year' : spanDays > 400 ? 'quarter' : spanDays > 120 ? 'month' : 'week';
  const muted = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() || '#888';

  taaCharts[cid] = new Chart(document.getElementById(cid), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: '기준가', data: keep.map(i => +t.nav[i].toFixed(2)), borderColor: color,
          fill: false, pointRadius: 0, borderWidth: 1.6 },
        { label: `${taaState.win}일 이동평균`, data: keep.map(i => t.ma[i] == null ? null : +t.ma[i].toFixed(2)),
          borderColor: muted, borderDash: [6, 4], fill: false, pointRadius: 0, borderWidth: 1.4 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      scales: { x: { type: 'time', time: { unit }, ticks: { maxTicksLimit: 9 } }, y: { beginAtZero: false } },
      plugins: {
        taaBands: { bands },
        legend: { display: true, position: 'top', labels: { boxWidth: 14, font: { size: 11 } } },
      },
    },
  });

  taaCharts[cid + '-g'] = new Chart(document.getElementById(cid + '-g'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: '이격도 %', data: keep.map(i => t.gap[i] == null ? null : +t.gap[i].toFixed(2)),
          borderColor: color, backgroundColor: color + '22', fill: 'origin',
          pointRadius: 0, borderWidth: 1.2, yAxisID: 'y' },
        { label: '이동평균 기울기 (연율 %)', data: keep.map(i => t.slope[i] == null ? null : +t.slope[i].toFixed(2)),
          borderColor: muted, fill: false, pointRadius: 0, borderWidth: 1.2, yAxisID: 'y1' },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      scales: {
        x: { type: 'time', time: { unit }, ticks: { maxTicksLimit: 9 } },
        y: { position: 'left', grid: { color: c => c.tick.value === 0 ? 'rgba(128,128,128,.55)' : 'rgba(128,128,128,.12)' } },
        y1: { position: 'right', grid: { display: false } },
      },
      plugins: { taaBands: { bands }, legend: { display: true, position: 'top', labels: { boxWidth: 14, font: { size: 10 } } } },
    },
  });
}

const taaPeriod = makePeriodPicker({
  card: 'taa-period', info: 'taa-period-info', presets: 'taa-period-presets',
  years: 'taa-period-years', start: 'taa-start', end: 'taa-end',
  infoLabel: '이동평균이 확정된 전체 기간',
}, () => refreshTaa());

// 설정 칩. 값이 바뀌면 이동평균부터 전부 다시 계산한다.
function taaChips(boxId, defs, apply) {
  const box = document.getElementById(boxId);
  if (defs) box.innerHTML = defs.map(d =>
    `<button class="period-chip${d.on ? ' active' : ''}" type="button" data-v="${d.v}">${d.label}</button>`).join('');
  box.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    box.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
    apply(b.dataset.v);
    refreshTaa();
  }));
}
taaChips('taa-win', [100, 150, 200, 250].map(v => ({ v, label: v + '일', on: v === 200 })),
  v => taaState.win = +v);
taaChips('taa-cadence', null, v => taaState.cadence = v);
taaChips('taa-slope', null, v => taaState.slope = v === '1');
taaChips('taa-buffer', [0, 1, 2, 3].map(v => ({ v, label: v ? `±${v}%` : '없음', on: v === 0 })),
  v => taaState.buffer = +v);
taaChips('taa-epdepth', [-10, -15, -20, -30].map(v => ({ v, label: `${v}%`, on: v === -20 })),
  v => taaState.epDepth = +v);

function toggleFundView(btn) {
  const group = btn.parentElement.dataset.group;
  // Hide all views in this group
  btn.parentElement.querySelectorAll('.btn-currency').forEach(b => {
    b.classList.remove('active');
    const el = document.getElementById(b.dataset.view);
    if (el) el.style.display = 'none';
  });
  // Show selected
  btn.classList.add('active');
  const target = document.getElementById(btn.dataset.view);
  if (target) target.style.display = '';
}

// ── Portfolio Analyzer ──

// Per-fund currency mode for portfolio
const pfFundCurrency = {};
FUNDS.forEach((f, i) => { if (f.hasKrw || f.hasJpy) pfFundCurrency[i] = f.currency === 'USD' ? 'krw' : 'orig'; });

// Build fund selector UI
(function buildSelector() {
  const pfContainers = {
    insurance: document.getElementById('fund-selector-insurance'),
    us: document.getElementById('fund-selector-us'),
    jp: document.getElementById('fund-selector-jp'),
    index: document.getElementById('fund-selector-index'),
  };
  FUNDS.forEach((fund, idx) => {
    const row = document.createElement('div');
    row.className = 'fund-row';
    row.innerHTML = `
      <label><input type="checkbox" data-idx="${idx}">
        <span>${chipLabel(fund)}</span></label>
      <span class="pf-ccy-slot"></span>
      <input type="number" min="0" max="100" value="" data-idx="${idx}" style="width:70px" placeholder="0"> %`;
    const pfToggle = buildCcyToggle(fund, idx, 'margin:0;display:inline-flex;', pfFundCurrency, null);
    if (pfToggle) row.querySelector('.pf-ccy-slot').appendChild(pfToggle);
    pfContainers[fundRegion(fund)].appendChild(row);

    const cb = row.querySelector('input[type=checkbox]');
    const num = row.querySelector('input[type=number]');

    cb.addEventListener('change', () => {
      // Checking or unchecking always hands the row back to the auto split.
      pfPinned.delete(idx);
      if (!cb.checked) num.value = '';
      rebalanceWeights();
    });
    num.addEventListener('input', () => {
      cb.checked = +num.value > 0;
      if (cb.checked) pfPinned.add(idx); else { pfPinned.delete(idx); num.value = ''; }
      rebalanceWeights();
    });
  });
})();

function getPfFundData(fund, idx, key) {
  const mode = pfFundCurrency[idx] || (fund.currency === 'USD' ? 'krw' : 'orig');
  return getDataByMode(fund, mode, key);
}

// ── Weights ──
// Typing a weight pins that row; every unpinned row just splits whatever is left, so the
// common cases (equal weight, "60% here and share the rest") need no arithmetic.
const pfPinned = new Set();

function pfRows() {
  return [...document.querySelectorAll('#fund-selector .fund-row')].map(row => ({
    row,
    cb: row.querySelector('input[type=checkbox]'),
    num: row.querySelector('input[type=number]'),
    idx: +row.querySelector('input[type=checkbox]').dataset.idx,
  }));
}

function rebalanceWeights() {
  const checked = pfRows().filter(r => r.cb.checked);
  const auto = checked.filter(r => !pfPinned.has(r.idx));
  if (auto.length > 0) {
    const pinnedSum = checked.filter(r => pfPinned.has(r.idx))
      .reduce((s, r) => s + (+r.num.value || 0), 0);
    const left = Math.max(0, 100 - pinnedSum);
    // Floor to 0.1 and give the remainder to the first row so the total lands exactly on 100.
    const each = Math.floor(left / auto.length * 10) / 10;
    auto.forEach(r => { r.num.value = each; });
    const drift = +(left - each * auto.length).toFixed(1);
    if (drift !== 0) auto[0].num.value = +(each + drift).toFixed(1);
  }
  pfRows().forEach(r => r.num.classList.toggle('pinned', r.cb.checked && pfPinned.has(r.idx)));
  updateWeightSum();
}

document.getElementById('pf-equal').addEventListener('click', () => {
  pfPinned.clear();
  rebalanceWeights();
});

// Keeps the ratios the user set and scales them onto 100 — the escape hatch for when
// every row is pinned and the total drifted.
document.getElementById('pf-normalize').addEventListener('click', () => {
  const checked = pfRows().filter(r => r.cb.checked);
  const sum = checked.reduce((s, r) => s + (+r.num.value || 0), 0);
  if (checked.length === 0 || sum <= 0) return;
  let acc = 0;
  checked.forEach((r, i) => {
    const v = i === checked.length - 1 ? +(100 - acc).toFixed(1)
                                       : +((+r.num.value || 0) / sum * 100).toFixed(1);
    acc = +(acc + v).toFixed(1);
    r.num.value = v;
    pfPinned.add(r.idx);
  });
  rebalanceWeights();
});

function getSelections() {
  const rows = document.querySelectorAll('#fund-selector-insurance .fund-row, #fund-selector-us .fund-row, #fund-selector-jp .fund-row, #fund-selector-index .fund-row');
  const sel = [];
  rows.forEach(row => {
    const cb = row.querySelector('input[type=checkbox]');
    const w = +row.querySelector('input[type=number]').value;
    if (cb.checked && w > 0) sel.push({ idx: +cb.dataset.idx, weight: w / 100 });
  });
  return sel;
}

function updateWeightSum() {
  const sel = getSelections();
  const sum = sel.reduce((s, x) => s + x.weight * 100, 0);
  const el = document.getElementById('weight-sum');
  el.textContent = `비중 합계: ${sum.toFixed(1)}%`;
  el.className = 'weight-sum' + (Math.abs(sum - 100) > 0.1 ? ' warn' : '');
  document.getElementById('btn-analyze').disabled = sel.length === 0 || Math.abs(sum - 100) > 0.1;
  document.getElementById('btn-save-preset').disabled = sel.length === 0;
  setBadge('badge-portfolio', sel.length);

  // Re-point the period picker at the new selection's common range
  if (sel.length === 0) { pfPeriod.hide(); return; }
  const dailySets = sel.map(s => getPfFundData(FUNDS[s.idx], s.idx, 'daily'));
  const dateSets = dailySets.map(d => new Set(d.dates));
  const common = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d))).sort();
  if (common.length === 0) {
    pfPeriod.hide();
    document.getElementById('pf-date-info').textContent = '공통 기간 없음';
    return;
  }
  // Name the asset(s) that decided the start date (latest first date = bottleneck)
  const startDates = sel.map(s => {
    const d = getPfFundData(FUNDS[s.idx], s.idx, 'daily');
    return { idx: s.idx, firstDate: d.dates[0] };
  });
  const latestFirst = startDates.reduce((a, b) => a.firstDate > b.firstDate ? a : b).firstDate;
  const bottlenecks = startDates.filter(s => s.firstDate === latestFirst).map(s => chipLabel(FUNDS[s.idx]));
  pfPeriod.setBounds(common[0], common[common.length - 1], ` (← ${bottlenecks.join(', ')})`);
}

// Build portfolio NAV from weighted daily returns
function buildPortfolio(selections) {
  // Find common date range (respecting per-fund currency mode)
  const dailySets = selections.map(s => getPfFundData(FUNDS[s.idx], s.idx, 'daily'));
  const dateSets = dailySets.map(d => new Set(d.dates));
  let commonDates = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d))).sort();

  commonDates = pfPeriod.filter(commonDates);

  const dates = commonDates;
  if (dates.length === 0) return null;
  // Each asset contributes everything it earned since the previous grid date, so funds
  // that quote on non-trading days keep those returns (see segmentReturns).
  const segs = dailySets.map(f => segmentReturns(f, dates));
  const returns = dates.map((d, i) => {
    let r = 0;
    selections.forEach((s, si) => { r += segs[si][i] * s.weight; });
    return r;
  });

  // Build NAV (start = 1000)
  const nav = [1000];
  for (let i = 0; i < returns.length; i++) {
    nav.push(nav[nav.length - 1] * (1 + returns[i]));
  }
  // dates for NAV: add a synthetic first date (day before first return)
  const firstDate = new Date(dates[0]);
  firstDate.setDate(firstDate.getDate() - 1);
  const navDates = [firstDate.toISOString().slice(0, 10), ...dates];

  return { dates: navDates, nav, returns, returnDates: dates };
}

// Metrics calculation (mirrors Python)
// Annualised volatility on a MONTHLY basis.
// Insurance-fund NAV reflects its holdings 1-3 days late and smoothed, which crushes
// daily volatility (N1M0: 15.7% daily vs 21.0% monthly) and makes a blend with an ETF
// look less risky than either leg. Monthly aggregation washes that out with no
// lag model to guess at, and puts funds and ETFs on one yardstick.
function monthlyVol(dates, nav) {
  const last = {};
  dates.forEach((d, i) => { last[d.slice(0, 7)] = nav[i]; });
  const vals = Object.keys(last).sort().map(k => last[k]);
  if (vals.length < 7) return null;   // need 6+ returns
  const r = [];
  for (let i = 1; i < vals.length; i++) r.push(vals[i] / vals[i - 1] - 1);
  const m = r.reduce((s, v) => s + v, 0) / r.length;
  const varr = r.reduce((s, v) => s + (v - m) ** 2, 0) / (r.length - 1);
  return Math.sqrt(varr) * Math.sqrt(12);
}

function calcMetrics(dates, nav) {
  const n = nav.length;
  const firstDate = new Date(dates[0]);
  const lastDate = new Date(dates[n - 1]);
  const totalYears = (lastDate - firstDate) / (365.25 * 86400000);
  if (totalYears <= 0) return null;

  const totalReturn = (nav[n - 1] / nav[0] - 1) * 100;
  const cagr = (Math.pow(nav[n - 1] / nav[0], 1 / totalYears) - 1);

  let vol = monthlyVol(dates, nav);
  if (vol === null) {
    const dr = [];
    for (let i = 1; i < n; i++) dr.push(nav[i] / nav[i - 1] - 1);
    const mean = dr.reduce((s, v) => s + v, 0) / dr.length;
    const variance = dr.reduce((s, v) => s + (v - mean) ** 2, 0) / (dr.length - 1);
    vol = Math.sqrt(variance) * Math.sqrt(dr.length / totalYears);
  }
  const sharpe = vol > 0 ? (cagr - RISK_FREE) / vol : 0;

  // Drawdown series
  let peak = nav[0];
  const dd = nav.map(v => { peak = Math.max(peak, v); return (v - peak) / peak; });
  const mdd = Math.min(...dd);

  return {
    firstDate: dates[0], lastDate: dates[n - 1],
    totalYears: totalYears.toFixed(1),
    totalReturn: totalReturn.toFixed(2),
    cagr: (cagr * 100).toFixed(2),
    volatility: (vol * 100).toFixed(2),
    sharpe: sharpe.toFixed(2),
    mdd: (mdd * 100).toFixed(2),
    drawdownSeries: dd,
  };
}

function findDrawdowns(dates, nav, topN) {
  let peak = nav[0];
  const dd = nav.map(v => { peak = Math.max(peak, v); return (v - peak) / peak; });

  const events = [];
  let inDd = false, start = 0, troughIdx = 0, troughVal = 0;

  for (let i = 0; i < dd.length; i++) {
    if (!inDd && dd[i] < 0) {
      inDd = true; start = i; troughIdx = i; troughVal = dd[i];
    } else if (inDd) {
      if (dd[i] < troughVal) { troughIdx = i; troughVal = dd[i]; }
      if (dd[i] >= 0) {
        inDd = false;
        const dStart = new Date(dates[start]), dEnd = new Date(dates[i]);
        events.push({
          start: dates[start], trough: dates[troughIdx], end: dates[i],
          depth: (troughVal * 100).toFixed(2),
          days: Math.round((dEnd - dStart) / 86400000),
        });
      }
    }
  }
  if (inDd) {
    const dStart = new Date(dates[start]), dEnd = new Date(dates[dates.length - 1]);
    events.push({
      start: dates[start], trough: dates[troughIdx], end: null,
      depth: (troughVal * 100).toFixed(2),
      days: Math.round((dEnd - dStart) / 86400000),
    });
  }
  events.sort((a, b) => +a.depth - +b.depth);
  return events.slice(0, topN);
}

function calcLsDca(nav, dates, windowMonths) {
  // Resample to monthly (last value per month)
  const monthly = {};
  dates.forEach((d, i) => {
    const ym = d.slice(0, 7); // YYYY-MM
    monthly[ym] = nav[i];
  });
  const keys = Object.keys(monthly).sort();
  const vals = keys.map(k => monthly[k]);
  const n = vals.length;
  if (n <= windowMonths) return null;

  const advantages = [];
  for (let i = 0; i <= n - windowMonths - 1; i++) {
    const endNav = vals[i + windowMonths];
    const rLs = endNav / vals[i] - 1;
    let sumRatio = 0;
    for (let k = 0; k < windowMonths; k++) sumRatio += endNav / vals[i + k];
    const rDca = sumRatio / windowMonths - 1;
    advantages.push(rLs - rDca);
  }

  const wins = advantages.filter(a => a > 0).length;
  const winRate = (wins / advantages.length * 100).toFixed(1);
  const mlsa = (advantages.reduce((s, v) => s + v, 0) / advantages.length * 100).toFixed(2);
  const losses = advantages.filter(a => a <= 0);
  const mlsd = losses.length > 0 ? (losses.reduce((s, v) => s + v, 0) / losses.length * 100).toFixed(2) : '0.00';

  return { window: windowMonths, observations: advantages.length, winRate, mlsa, mlsd };
}

// Rolling trailing return analysis
let trailingChart = null;

function renderTrailingReturns(dates, nav) {
  const el = document.getElementById('pf-trailing');
  const n = nav.length;
  if (n < 2) { el.innerHTML = ''; return; }

  const totalYears = (new Date(dates[n-1]) - new Date(dates[0])) / (365.25 * 86400000);
  const maxWindow = Math.floor(totalYears);
  if (maxWindow < 1) { el.innerHTML = ''; return; }

  // Build date lookup: date string → index
  const dateIdx = {};
  dates.forEach((d, i) => { dateIdx[d] = i; });

  // Build buttons
  const windows = [];
  for (let y = 1; y <= Math.min(maxWindow, 10); y++) windows.push(y);

  const chips = windows.map(y =>
    `<label class="filter-chip${y === 1 ? ' active' : ''}" data-window="${y}"><input type="radio" name="trailing-window" value="${y}" ${y === 1 ? 'checked' : ''} style="display:none">${y}Y</label>`
  ).join('');

  el.innerHTML = `
    <h3>Rolling Trailing Returns</h3>
    <div class="filter-chips" id="trailing-chips" style="margin-bottom:1rem;">${chips}</div>
    <div class="metrics-grid" id="trailing-metrics"></div>
    <div class="chart-container" style="height:250px;"><canvas id="trailing-chart"></canvas></div>`;

  function calcRolling(windowYears) {
    const returns = [];
    const returnDates = [];
    for (let i = 0; i < n; i++) {
      const startDate = new Date(dates[i]);
      const endDate = new Date(startDate);
      endDate.setFullYear(endDate.getFullYear() + windowYears);
      const endStr = endDate.toISOString().slice(0, 10);

      // Find closest date >= endStr
      let ei = -1;
      for (let j = i + 1; j < n; j++) {
        if (dates[j] >= endStr) { ei = j; break; }
      }
      if (ei < 0) break;

      const r = (Math.pow(nav[ei] / nav[i], 1 / windowYears) - 1) * 100; // annualized
      returns.push(r);
      returnDates.push(dates[i]);
    }
    return { returns, dates: returnDates };
  }

  function showRolling(windowYears) {
    const { returns, dates: rDates } = calcRolling(windowYears);
    if (returns.length === 0) return;

    const avg = returns.reduce((s,v) => s+v, 0) / returns.length;
    const variance = returns.reduce((s,v) => s + (v-avg)**2, 0) / (returns.length - 1);
    const std = Math.sqrt(variance);
    const se = std / Math.sqrt(returns.length);
    const min = Math.min(...returns);
    const max = Math.max(...returns);
    const median = [...returns].sort((a,b) => a-b)[Math.floor(returns.length / 2)];
    const positive = returns.filter(r => r > 0).length;
    const winRate = (positive / returns.length * 100);

    const pctCls = v => v > 0 ? 'positive' : v < 0 ? 'negative' : '';
    const fmtPct = (v, sign) => (sign && v > 0 ? '+' : '') + v.toFixed(2) + '%';

    document.getElementById('trailing-metrics').innerHTML = `
      <div class="metric-card"><div class="label">관측수</div><div class="value">${returns.length}</div></div>
      <div class="metric-card"><div class="label">평균 CAGR</div><div class="value ${pctCls(avg)}">${fmtPct(avg, true)}</div></div>
      <div class="metric-card"><div class="label">중앙값</div><div class="value ${pctCls(median)}">${fmtPct(median, true)}</div></div>
      <div class="metric-card"><div class="label">표준편차</div><div class="value">${std.toFixed(2)}%</div></div>
      <div class="metric-card"><div class="label">표준오차</div><div class="value">${se.toFixed(2)}%</div></div>
      <div class="metric-card"><div class="label">최소</div><div class="value ${pctCls(min)}">${fmtPct(min, true)}</div></div>
      <div class="metric-card"><div class="label">최대</div><div class="value ${pctCls(max)}">${fmtPct(max, true)}</div></div>
      <div class="metric-card"><div class="label">양수 비율</div><div class="value ${winRate > 50 ? 'positive' : 'negative'}">${winRate.toFixed(1)}%</div></div>`;

    // Chart: rolling return over time
    const step = Math.max(1, Math.floor(rDates.length / 400));
    const cDates = rDates.filter((_, i) => i % step === 0);
    const cReturns = returns.filter((_, i) => i % step === 0);

    if (trailingChart) trailingChart.destroy();
    trailingChart = new Chart(document.getElementById('trailing-chart'), {
      type: 'line',
      data: { labels: cDates, datasets: [
        { label: windowYears + 'Y CAGR (%)', data: cReturns.map(v => +v.toFixed(2)),
          borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.12)',
          fill: true, pointRadius: 0, borderWidth: 1.5 },
        { label: '평균', data: cDates.map(() => +avg.toFixed(2)),
          borderColor: '#888', borderDash: [5, 5], pointRadius: 0, borderWidth: 1 },
        { label: '0%', data: cDates.map(() => 0),
          borderColor: '#ef4444', borderDash: [3, 3], pointRadius: 0, borderWidth: 1 },
      ]},
      options: { responsive: true, maintainAspectRatio: false,
        scales: {
          x: { type: 'time', time: { unit: 'year' }, ticks: { maxTicksLimit: 8 } },
          y: { ticks: { callback: v => v + '%' } }
        },
        plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 12, font: { size: 11 } } } }
      }
    });
  }

  // Wire up chips
  el.querySelectorAll('#trailing-chips .filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      el.querySelectorAll('#trailing-chips .filter-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      showRolling(+chip.dataset.window);
    });
  });

  showRolling(1); // default
}

// Render portfolio results
let pfNavChart = null, pfDdChart = null;
let pfFullDates = [], pfFullNav = []; // full (non-downsampled) data for selection analysis

function renderPortfolio(pf) {
  const m = calcMetrics(pf.dates, pf.nav);
  if (!m) return;

  // 변동성·샤프는 monthlyVol 로 옮겨서 기준가 지연·평활 문제를 피했다. 남는 것은
  // 일간에서 뽑는 MDD·하락 이벤트뿐이라, 보험펀드가 섞이면 그 점만 알린다.
  const sel = getSelections();
  const hasFund = sel.some(x => !FUNDS[x.idx].isBench);
  const lagNote = document.getElementById('pf-lag-note');
  lagNote.style.display = hasFund ? '' : 'none';
  lagNote.innerHTML = hasFund
    ? '변동성·샤프비율은 월간 기준이라 기준가 지연·평활의 영향을 받지 않습니다. 다만 ' +
      '<b>MDD와 하락 이벤트는 일간 기준</b>이라, 평활된 보험펀드 기준가에서는 실제 낙폭보다 ' +
      '얕게 나올 수 있습니다.'
    : '';

  const events = findDrawdowns(pf.dates, pf.nav, 5);
  const lsDca = [3, 12, 36].map(w => calcLsDca(pf.nav, pf.dates, w)).filter(Boolean);

  // Drawdown summary
  let avgDd = 0, longestDays = 0, longestStart = '-', longestEnd = '-';
  if (events.length > 0) {
    avgDd = (events.reduce((s, e) => s + Math.abs(+e.depth), 0) / events.length).toFixed(2);
    const longest = events.reduce((a, b) => b.days > a.days ? b : a, events[0]);
    longestDays = longest.days;
    longestStart = longest.start;
    longestEnd = longest.end || '진행중';
  }

  document.getElementById('portfolio-results').style.display = 'block';

  // Metrics grid
  const pctCls = v => +v > 0 ? 'positive' : +v < 0 ? 'negative' : '';
  const fmtPct = (v, sign) => (sign && +v > 0 ? '+' : '') + v + '%';
  document.getElementById('pf-metrics').innerHTML = `
    <div class="metric-card"><div class="label">기간</div><div class="value">${m.totalYears}년</div></div>
    <div class="metric-card"><div class="label">총 수익률</div><div class="value ${pctCls(m.totalReturn)}">${fmtPct(m.totalReturn, true)}</div></div>
    <div class="metric-card"><div class="label">CAGR</div><div class="value ${pctCls(m.cagr)}">${fmtPct(m.cagr, true)}</div></div>
    <div class="metric-card"><div class="label">변동성 (월간)</div><div class="value">${m.volatility}%</div></div>
    <div class="metric-card"><div class="label">샤프비율</div><div class="value">${m.sharpe}</div></div>
    <div class="metric-card"><div class="label">MDD</div><div class="value negative">${m.mdd}%</div></div>
    <div class="metric-card"><div class="label">평균 하락폭</div><div class="value negative">-${avgDd}%</div></div>
    <div class="metric-card"><div class="label">최장 하락 기간</div><div class="value">${longestDays}일</div></div>`;

  // Store full data for drag-selection analysis
  pfFullDates = pf.dates;
  pfFullNav = pf.nav;

  // Trailing returns
  renderTrailingReturns(pf.dates, pf.nav);

  // Charts (downsample)
  const step = Math.max(1, Math.floor(pf.dates.length / 500));
  const chartDates = pf.dates.filter((_, i) => i % step === 0);
  const chartNav = pf.nav.filter((_, i) => i % step === 0);
  const chartDd = m.drawdownSeries.filter((_, i) => i % step === 0).map(v => +(v * 100).toFixed(2));

  if (pfNavChart) pfNavChart.destroy();
  if (pfDdChart) pfDdChart.destroy();

  pfNavChart = new Chart(document.getElementById('pf-nav-chart'), {
    type: 'line',
    data: { labels: chartDates, datasets: [{
      label: '포트폴리오 NAV', data: chartNav.map(v => +v.toFixed(2)),
      borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.12)',
      fill: true, pointRadius: 0, borderWidth: 1.5,
    }]},
    options: { responsive: true, maintainAspectRatio: false,
      scales: { x: { type: 'time', time: { unit: 'year' }, ticks: { maxTicksLimit: 8 } }, y: { beginAtZero: false } },
      plugins: { legend: { display: false } }
    }
  });

  pfDdChart = new Chart(document.getElementById('pf-dd-chart'), {
    type: 'line',
    data: { labels: chartDates, datasets: [{
      label: '드로다운 (%)', data: chartDd,
      borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.16)',
      fill: true, pointRadius: 0, borderWidth: 1.5,
    }]},
    options: { responsive: true, maintainAspectRatio: false,
      scales: { x: { type: 'time', time: { unit: 'year' }, ticks: { maxTicksLimit: 8 } }, y: { max: 0 } },
      plugins: { legend: { display: false } }
    }
  });

  // Drawdown events table
  if (events.length > 0) {
    let rows = events.map((e, i) =>
      `<tr><td>${i + 1}</td><td>${e.start}</td><td>${e.trough}</td>` +
      `<td>${e.end || '<span class="ongoing">진행중</span>'}</td>` +
      `<td class="negative">${e.depth}%</td><td>${e.days.toLocaleString()}일</td></tr>`
    ).join('');
    document.getElementById('pf-dd-table').innerHTML = `
      <h3>주요 하락 이벤트 (Top ${events.length})</h3>
      <table><tr><th>#</th><th>시작</th><th>저점</th><th>회복</th><th>하락폭</th><th>기간</th></tr>${rows}</table>`;
  } else {
    document.getElementById('pf-dd-table').innerHTML = '';
  }

  // LS vs DCA table
  if (lsDca.length > 0) {
    let rows = lsDca.map(r => {
      const wCls = +r.winRate > 50 ? 'positive' : 'negative';
      const mCls = +r.mlsa > 0 ? 'positive' : 'negative';
      return `<tr><td>${r.window}개월</td><td>${r.observations.toLocaleString()}</td>` +
        `<td class="${wCls}">${r.winRate}%</td>` +
        `<td class="${mCls}">${+r.mlsa > 0 ? '+' : ''}${r.mlsa}%</td>` +
        `<td class="negative">${r.mlsd}%</td></tr>`;
    }).join('');
    document.getElementById('pf-ls-table').innerHTML = `
      <h3>LS vs DCA 분석</h3>
      <table><tr><th>기간</th><th>관측수</th><th>LS 승률</th><th>MLSA</th><th>MLSD</th></tr>${rows}</table>`;
  } else {
    document.getElementById('pf-ls-table').innerHTML = '<p style="color:var(--subtle);">데이터 부족으로 LS vs DCA 분석 불가</p>';
  }
}

// Correlation matrix for selected assets (weekly returns to avoid timing mismatch)
function calcCorrelation(selections) {
  if (selections.length < 2) return null;

  // Use weekly returns for correlation (respecting per-fund currency mode)
  const fundData = selections.map(s => {
    const f = getPfFundData(FUNDS[s.idx], s.idx, 'monthly');
    const m = {};
    f.dates.forEach((d, i) => { m[d] = f.returns[i]; });
    return { name: FUNDS[s.idx].shortName, map: m, dates: new Set(f.dates) };
  });

  let common = pfPeriod
    .filter([...fundData[0].dates].filter(d => fundData.every(f => f.dates.has(d))))
    .sort();
  if (common.length < 6) return null;

  // Build return arrays for common dates
  const arrays = fundData.map(f => common.map(d => f.map[d]));
  const n = common.length;
  const names = fundData.map(f => f.name);

  // Compute means
  const means = arrays.map(arr => arr.reduce((s, v) => s + v, 0) / n);

  // Compute correlation matrix
  const matrix = [];
  for (let i = 0; i < arrays.length; i++) {
    const row = [];
    for (let j = 0; j < arrays.length; j++) {
      let sumXY = 0, sumX2 = 0, sumY2 = 0;
      for (let k = 0; k < n; k++) {
        const dx = arrays[i][k] - means[i];
        const dy = arrays[j][k] - means[j];
        sumXY += dx * dy;
        sumX2 += dx * dx;
        sumY2 += dy * dy;
      }
      const denom = Math.sqrt(sumX2 * sumY2);
      row.push(denom > 0 ? sumXY / denom : 0);
    }
    matrix.push(row);
  }
  return { names, matrix, obs: n };
}

function renderCorrelation(selections) {
  const el = document.getElementById('pf-corr-table');
  const corr = calcCorrelation(selections);
  if (!corr) { el.innerHTML = ''; return; }

  function cellStyle(v) {
    if (v >= 1) return 'background:#1d4ed8;color:#fff;';
    if (v >= 0) return `background:rgba(37,99,235,${(v*0.5).toFixed(2)});color:${v>0.7?'#fff':'var(--text)'};`;
    return `background:rgba(220,38,38,${(Math.abs(v)*0.5).toFixed(2)});color:${v<-0.7?'#fff':'var(--text)'};`;
  }

  const header = '<tr><th></th>' + corr.names.map(n => `<th>${n}</th>`).join('') + '</tr>';
  const rows = corr.matrix.map((row, i) =>
    '<tr><th>' + corr.names[i] + '</th>' +
    row.map(v => `<td style="${cellStyle(v)}">${v.toFixed(2)}</td>`).join('') + '</tr>'
  ).join('');

  el.innerHTML = `
    <h3>상관행렬 (Correlation Matrix)</h3>
    <p class="fund-meta">월간 수익률 기준 | ${pfPeriod.narrowed() ? '선택' : '공통'} 기간 관측수: ${corr.obs.toLocaleString()}개월</p>
    <table class="corr-table">${header}${rows}</table>`;
}

// Yearly return breakdown per asset + portfolio
function renderYearlyBreakdown(selections, pf) {
  const el = document.getElementById('pf-yearly');
  if (!pf || pf.dates.length < 30) { el.innerHTML = ''; return; }

  // Get year range from portfolio dates
  const startYear = +pf.dates[0].slice(0, 4);
  const endYear = +pf.dates[pf.dates.length - 1].slice(0, 4);
  const years = [];
  for (let y = startYear; y <= endYear; y++) years.push(y);
  if (years.length < 1) { el.innerHTML = ''; return; }

  // Build per-asset daily return lookups on common dates (from pf.returnDates)
  const assetLookups = selections.map(s => {
    const daily = getPfFundData(FUNDS[s.idx], s.idx, 'daily');
    const m = {};
    daily.dates.forEach((d, i) => { m[d] = daily.returns[i]; });
    return m;
  });
  const weights = selections.map(s => s.weight);
  const names = selections.map(s => FUNDS[s.idx].shortName || FUNDS[s.idx].name);

  // Compute yearly: asset return, portfolio return, and exact contribution
  // Contribution = sum of (asset_daily_return × weight) for each day, compounded
  // Build per-asset FULL daily return data (not just common dates)
  const assetFullDailys = selections.map(s => getPfFundData(FUNDS[s.idx], s.idx, 'daily'));

  function yearlyData(year) {
    // Get portfolio dates in this year (common dates for portfolio calculation)
    const yearDates = pf.returnDates ? pf.returnDates.filter(d => d.slice(0,4) === ''+year) : [];
    if (yearDates.length === 0) return null;

    // Asset yearly returns: use each asset's OWN full daily data, not common dates
    const assetReturns = assetFullDailys.map(daily => {
      let cum = 1;
      for (let i = 0; i < daily.dates.length; i++) {
        if (daily.dates[i].slice(0,4) === ''+year) cum *= (1 + daily.returns[i]);
      }
      return (cum - 1) * 100;
    });

    // Exact contribution: cumulate daily weighted returns per asset
    // Portfolio daily return = sum(weight_i × asset_i_daily_return)
    // Asset i contribution to portfolio = product of (1 + pf_daily) attributed to asset i
    // Simpler accurate method: sum daily contributions then compound
    const dailyContribs = assetLookups.map((lookup, i) =>
      yearDates.map(d => (lookup[d] || 0) * weights[i])
    );

    // Portfolio yearly return from daily portfolio returns
    let pfCum = 1;
    for (const d of yearDates) {
      let dayR = 0;
      assetLookups.forEach((lookup, i) => { dayR += (lookup[d] || 0) * weights[i]; });
      pfCum *= (1 + dayR);
    }
    const pfReturn = (pfCum - 1) * 100;

    // Contribution using Brinson-style: compound daily contributions
    const contribs = dailyContribs.map(dc => {
      // Approximate: sum of daily contributions scaled by portfolio growth
      // Exact arithmetic attribution is complex; use log-based approximation
      let sum = 0;
      let pfGrowth = 1;
      for (let j = 0; j < yearDates.length; j++) {
        let dayPfR = 0;
        assetLookups.forEach((lookup, k) => { dayPfR += (lookup[yearDates[j]] || 0) * weights[k]; });
        sum += dc[j] / pfGrowth;  // scale by portfolio level at that point
        pfGrowth *= (1 + dayPfR);
      }
      return sum * (pfCum) * 100 / 100;  // this doesn't add up perfectly either
    });

    // Simplest accurate method: just scale so contributions sum to portfolio return
    const rawContribs = assetReturns.map((r, i) => r * weights[i]);
    const rawSum = rawContribs.reduce((s, v) => s + v, 0);
    const scaledContribs = rawSum !== 0
      ? rawContribs.map(c => c * pfReturn / rawSum)
      : rawContribs;

    // Detect partial year: first and last month in this year's data
    const firstMonth = +yearDates[0].slice(5,7);
    const lastMonth = +yearDates[yearDates.length-1].slice(5,7);
    const isPartial = firstMonth > 1 || lastMonth < 12;
    const monthRange = isPartial ? `${yearDates[0].slice(5,7)}~${yearDates[yearDates.length-1].slice(5,7)}월` : null;

    return { assetReturns, pfReturn, contribs: scaledContribs, monthRange };
  }

  function cellBg(v) {
    if (v === null) return '';
    const opacity = Math.min(Math.abs(v) / 50, 0.5);
    const color = v > 0 ? `rgba(22,163,74,${opacity})` : v < 0 ? `rgba(220,38,38,${opacity})` : '';
    return `background:${color};`;
  }

  // Split "코리아인덱스(N1M0)" into name + code so the header stays narrow;
  // fixed layout would otherwise size every column by its longest label.
  const labels = names.map(n => {
    const m = n.match(/^\s*(.+?)\s*\(([^()]+)\)\s*$/);
    return m ? { main: m[1], sub: m[2] } : { main: n, sub: '' };
  });
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

  // Equal-width asset columns: fixed layout splits the leftover space evenly.
  const pfSep = 'border-left:3px solid var(--border-strong);';
  const pfBg = 'background:var(--surface-2);';
  const cols = `<colgroup><col style="width:84px;">${labels.map(()=>'<col>').join('')}<col style="width:112px;"></colgroup>`;

  let header = '<tr><th>연도</th>';
  labels.forEach((l, i) => {
    const sub = l.sub ? `${esc(l.sub)} · ` : '';
    header += `<th title="${esc(names[i])}"><div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(l.main)}</div>`
            + `<span style="font-weight:400;color:var(--subtle);">${sub}${(weights[i]*100).toFixed(0)}%</span></th>`;
  });
  header += `<th style="${pfSep}${pfBg}">포트폴리오</th></tr>`;

  let rows = '';
  for (const year of years) {
    const yd = yearlyData(year);
    if (!yd) { rows += `<tr><td><b>${year}</b></td>${labels.map(()=>'<td>-</td>').join('')}<td style="${pfSep}${pfBg}">-</td></tr>`; continue; }

    const yearLabel = yd.monthRange ? `${year}<br><span style="font-size:0.6rem;color:var(--subtle);">${yd.monthRange}</span>` : `${year}`;
    let row = `<tr><td><b>${yearLabel}</b></td>`;
    yd.assetReturns.forEach((r, i) => {
      const c = yd.contribs[i];
      row += `<td style="${cellBg(r)}padding:0.3rem 0.4rem;">${r > 0 ? '+' : ''}${r.toFixed(1)}%<br><span style="font-size:0.65rem;opacity:0.6;">${c > 0 ? '+' : ''}${c.toFixed(1)}%p</span></td>`;
    });
    row += `<td style="${pfSep}${pfBg}${cellBg(yd.pfReturn)}font-size:0.82rem;font-weight:700;">${yd.pfReturn > 0 ? '+' : ''}${yd.pfReturn.toFixed(2)}%</td>`;
    row += '</tr>';
    rows += row;
  }

  el.innerHTML = `
    <h3>연도별 자산 수익률 및 기여도</h3>
    <div style="overflow-x:auto;">
    <table style="width:100%;min-width:${140 + labels.length * 110}px;table-layout:fixed;font-size:0.75rem;line-height:1.3;">
      ${cols}
      ${header}
      ${rows}
    </table>
    </div>`;
}

// ── Portfolio Presets (localStorage) ──
const PRESET_KEY = 'fund_dashboard_presets';
const MAX_PRESETS = 5;

function loadPresets() {
  try { return JSON.parse(localStorage.getItem(PRESET_KEY)) || []; }
  catch { return []; }
}

function savePresets(presets) {
  localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
}

function getCurrentConfig() {
  const sel = getSelections();
  if (sel.length === 0) return null;
  return sel.map(s => ({
    idx: s.idx,
    weight: s.weight,
    ccy: pfFundCurrency[s.idx] || 'orig',
    name: FUNDS[s.idx].shortName || FUNDS[s.idx].name,
  }));
}

function applyPreset(preset) {
  // A preset carries deliberate weights, so every row it names counts as pinned.
  pfPinned.clear();
  // Clear all
  document.querySelectorAll('#fund-selector-insurance .fund-row, #fund-selector-us .fund-row, #fund-selector-jp .fund-row, #fund-selector-index .fund-row').forEach(row => {
    const cb = row.querySelector('input[type=checkbox]');
    const num = row.querySelector('input[type=number]');
    if (cb) cb.checked = false;
    if (num) num.value = '';
  });

  // Apply preset
  preset.items.forEach(item => {
    const rows = document.querySelectorAll('#fund-selector-insurance .fund-row, #fund-selector-us .fund-row, #fund-selector-jp .fund-row, #fund-selector-index .fund-row');
    rows.forEach(row => {
      const cb = row.querySelector('input[type=checkbox]');
      if (cb && +cb.dataset.idx === item.idx) {
        cb.checked = true;
        row.querySelector('input[type=number]').value = Math.round(item.weight * 100);
        // Set currency
        pfPinned.add(item.idx);
        pfFundCurrency[item.idx] = item.ccy;
        const ccyBtns = row.querySelectorAll('.currency-toggle .btn-currency');
        ccyBtns.forEach(b => b.classList.toggle('active', b.dataset.mode === item.ccy));
      }
    });
  });
  rebalanceWeights();
}

function renderPresetChips() {
  const container = document.getElementById('preset-chips');
  const presets = loadPresets();
  container.innerHTML = '';
  presets.forEach((preset, i) => {
    const chip = document.createElement('label');
    chip.className = 'filter-chip preset-chip';
    chip.innerHTML = `${preset.name}<span class="preset-del" data-idx="${i}">&times;</span>`;
    chip.addEventListener('click', (e) => {
      if (e.target.classList.contains('preset-del')) {
        const presets = loadPresets();
        presets.splice(+e.target.dataset.idx, 1);
        savePresets(presets);
        renderPresetChips();
        return;
      }
      applyPreset(preset);
    });
    container.appendChild(chip);
  });
  document.querySelector('.preset-bar').style.display = presets.length ? '' : 'none';
}

document.getElementById('btn-save-preset').addEventListener('click', () => {
  const config = getCurrentConfig();
  if (!config) return;
  const presets = loadPresets();
  if (presets.length >= MAX_PRESETS) {
    if (!confirm(`최대 ${MAX_PRESETS}개까지 저장 가능합니다. 가장 오래된 프리셋을 삭제하고 저장할까요?`)) return;
    presets.shift();
  }
  const label = config.map(c => c.name + Math.round(c.weight*100)).join('/');
  const name = prompt('프리셋 이름:', label);
  if (!name) return;
  presets.push({ name, items: config });
  savePresets(presets);
  renderPresetChips();
});

renderPresetChips();

// Annualised figures off a handful of days are noise, so refuse the window instead.
const PF_MIN_DAYS = 30;

function runPortfolioAnalysis() {
  const sel = getSelections();
  if (sel.length === 0) return;
  const results = document.getElementById('portfolio-results');
  const warn = document.getElementById('pf-period-warn');
  const pf = buildPortfolio(sel);
  if (!pf || pf.returnDates.length < PF_MIN_DAYS) {
    results.style.display = 'none';
    warn.textContent = `선택 기간의 공통 거래일이 ${PF_MIN_DAYS}일 미만입니다 ` +
                       `(현재 ${pf ? pf.returnDates.length : 0}일). 기간을 넓혀주세요.`;
    warn.style.display = '';
    return;
  }
  warn.style.display = 'none';
  renderPortfolio(pf);
  renderYearlyBreakdown(sel, pf);
  renderCorrelation(sel);
  clearSelection();
}

document.getElementById('btn-analyze').addEventListener('click', () => {
  pfAnalyzed = true;
  runPortfolioAnalysis();
});

// ── Drag-to-select on portfolio NAV chart ──
attachDragSelect('pf-nav-chart', 'pf-selection-overlay', 'pf-selection-stats',
  () => pfNavChart, () => pfFullDates, () => pfFullNav);

function clearSelection() {
  document.getElementById('pf-selection-overlay').style.display = 'none';
  document.getElementById('pf-selection-stats').style.display = 'none';
}

// A resize moves every plot area, which would leave painted bands misaligned — drop them.
window.addEventListener('resize', () => {
  document.querySelectorAll('.drag-overlay, .drag-stats').forEach(el => { el.style.display = 'none'; });
});
</script>
</body>
</html>
"""


def _fmt_pct(val: float, with_sign: bool = True) -> str:
    sign = "+" if val > 0 and with_sign else ""
    return f"{sign}{val:.2f}%"


def _pct_class(val: float) -> str:
    if val > 0:
        return "positive"
    elif val < 0:
        return "negative"
    return ""


def _render_analysis_block(data: dict, canvas_id_prefix: str) -> str:
    """Render metrics + charts + tables for one analysis variant."""
    b = data["basic"]
    ds = data["dd_summary"]

    metrics = f"""\
    <div class="metrics-grid" id="{canvas_id_prefix}-metrics">
      <div class="metric-card"><div class="label">기간</div><div class="value">{b['total_years']}년</div></div>
      <div class="metric-card"><div class="label">총 수익률</div><div class="value {_pct_class(b['total_return'])}">{_fmt_pct(b['total_return'])}</div></div>
      <div class="metric-card"><div class="label">CAGR</div><div class="value {_pct_class(b['cagr'])}">{_fmt_pct(b['cagr'])}</div></div>
      <div class="metric-card"><div class="label">변동성 (월간)</div><div class="value">{b['volatility']:.2f}%</div></div>
      <div class="metric-card"><div class="label">샤프비율</div><div class="value">{b['sharpe']:.2f}</div></div>
      <div class="metric-card"><div class="label">MDD</div><div class="value negative">{_fmt_pct(b['mdd'], False)}</div></div>
      <div class="metric-card"><div class="label">평균 하락폭</div><div class="value negative">-{ds['avg_drawdown']:.2f}%</div></div>
      <div class="metric-card"><div class="label">최장 하락 기간</div><div class="value">{ds['longest_days']}일</div></div>
    </div>"""

    charts = f"""\
    <div class="chart-row">
      <div class="chart-container" style="position:relative;">
        <canvas id="{canvas_id_prefix}-nav"></canvas>
        <div class="drag-overlay" id="{canvas_id_prefix}-overlay"></div>
        <div class="drag-stats" id="{canvas_id_prefix}-stats"></div>
      </div>
      <div class="chart-container"><canvas id="{canvas_id_prefix}-dd"></canvas></div>
    </div>
    <p class="hint">차트에서 드래그하여 구간 분석</p>"""

    events = data["top_events"]
    if events:
        rows = ""
        for i, e in enumerate(events, 1):
            end_str = e["end"] if e["end"] else '<span class="ongoing">진행중</span>'
            rows += f"<tr><td>{i}</td><td>{e['start']}</td><td>{e['trough']}</td>"
            rows += f"<td>{end_str}</td><td class='negative'>{e['depth']:.2f}%</td>"
            rows += f"<td>{e['duration_days']:,}일</td></tr>\n"
        dd_inner = f"""\
    <h3>주요 하락 이벤트 (Top {len(events)})</h3>
    <table><tr><th>#</th><th>시작</th><th>저점</th><th>회복</th><th>하락폭</th><th>기간</th></tr>
      {rows}</table>"""
    else:
        dd_inner = ""
    dd_table = f'<div id="{canvas_id_prefix}-ddtable">{dd_inner}</div>'

    ls_dca = data["ls_dca"]
    if ls_dca:
        ls_rows = ""
        for r in ls_dca:
            win_cls = "positive" if r["win_rate"] > 50 else "negative"
            mlsa_cls = _pct_class(r["mlsa"])
            ls_rows += f"<tr><td>{r['window']}개월</td><td>{r['observations']:,}</td>"
            ls_rows += f"<td class='{win_cls}'>{r['win_rate']:.1f}%</td>"
            ls_rows += f"<td class='{mlsa_cls}'>{_fmt_pct(r['mlsa'])}</td>"
            ls_rows += f"<td class='negative'>{_fmt_pct(r['mlsd'], False)}</td></tr>\n"
        ls_inner = f"""\
    <h3>LS vs DCA 분석</h3>
    <table><tr><th>기간</th><th>관측수</th><th>LS 승률</th><th>MLSA</th><th>MLSD</th></tr>
      {ls_rows}</table>"""
    else:
        ls_inner = '<p style="color:var(--subtle);">데이터 부족으로 LS vs DCA 분석 불가</p>'
    ls_table = f'<div id="{canvas_id_prefix}-lstable">{ls_inner}</div>'

    trailing = f'<div class="trailing-section" data-prefix="{canvas_id_prefix}"></div>'

    return f"{metrics}\n{charts}\n{trailing}\n{dd_table}\n{ls_table}"


def render_fund_section(fund: dict, idx: int) -> str:
    b = fund["basic"]
    has_krw = fund.get("has_krw", False) and "krw" in fund

    # Currency toggle for foreign currency assets
    has_usd = fund.get("has_usd", False) and "usd" in fund
    has_jpy = fund.get("has_jpy", False) and "jpy" in fund
    ccy_label = fund.get("currency_label", "USD")
    default_view = "orig"

    toggle_html = ""
    if has_krw or has_usd or has_jpy:
        btns = f'<button class="btn-currency active" data-view="fund-{idx}-orig" onclick="toggleFundView(this)">{ccy_label or "KRW"}</button>'
        if has_usd:
            btns += f'<button class="btn-currency" data-view="fund-{idx}-usd-conv" onclick="toggleFundView(this)">USD</button>'
        if has_krw:
            btns += f'<button class="btn-currency" data-view="fund-{idx}-krw" onclick="toggleFundView(this)">KRW</button>'
        if has_jpy:
            btns += f'<button class="btn-currency" data-view="fund-{idx}-jpy" onclick="toggleFundView(this)">JPY</button>'
        toggle_html = f'<div class="currency-toggle" style="margin-bottom:1rem;" data-group="fund-{idx}">{btns}</div>'

    orig_block = _render_analysis_block(fund, f"chart-{idx}-orig")
    orig_div = f'<div id="fund-{idx}-orig">{orig_block}</div>'

    usd_div = ""
    if has_usd:
        usd_block = _render_analysis_block(fund["usd"], f"chart-{idx}-usd-conv")
        usd_div = f'<div id="fund-{idx}-usd-conv" style="display:none">{usd_block}</div>'

    krw_div = ""
    if has_krw:
        krw_block = _render_analysis_block(fund["krw"], f"chart-{idx}-krw")
        krw_div = f'<div id="fund-{idx}-krw" style="display:none">{krw_block}</div>'

    jpy_div = ""
    if has_jpy:
        jpy_block = _render_analysis_block(fund["jpy"], f"chart-{idx}-jpy")
        jpy_div = f'<div id="fund-{idx}-jpy" style="display:none">{jpy_block}</div>'

    return f"""\
<section class="fund-section hidden" id="fund-{idx}">
  <h2>{fund['name']}</h2>
  <p class="fund-meta" id="fund-{idx}-meta" data-full="{fund['member_cd']} / {fund['fund_cd']} | {b['first_date']} ~ {b['last_date']}">{fund['member_cd']} / {fund['fund_cd']} | {b['first_date']} ~ {b['last_date']}</p>
  {toggle_html}
  {orig_div}
  {usd_div}
  {krw_div}
  {jpy_div}
</section>"""


def render_correlation_section(corr_data: dict | None) -> str:
    if not corr_data:
        return ""

    names = corr_data["names"]
    matrix = corr_data["matrix"]

    def _cell_style(val: float) -> str:
        if val >= 1.0:
            return "background: #1d4ed8; color: #fff;"
        # Blue scale for positive, red scale for negative
        if val >= 0:
            opacity = val
            return f"background: rgba(37,99,235,{opacity * 0.5:.2f}); color: {'#fff' if val > 0.7 else 'var(--text)'};"
        else:
            opacity = abs(val)
            return f"background: rgba(220,38,38,{opacity * 0.5:.2f}); color: {'#fff' if val < -0.7 else 'var(--text)'};"

    header = "<tr><th></th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>"
    rows = ""
    for i, name in enumerate(names):
        cells = "".join(
            f'<td style="{_cell_style(matrix[i][j])}">{matrix[i][j]:.2f}</td>'
            for j in range(len(names))
        )
        rows += f"<tr><th>{name}</th>{cells}</tr>\n"

    return f"""\
<section class="fund-section">
  <h2>상관행렬 (Correlation Matrix)</h2>
  <p class="fund-meta">일별 수익률 기준 | 공통 기간 관측수: {corr_data['obs']:,}일</p>
  <table class="corr-table">
    {header}
    {rows}
  </table>
</section>"""


def render_html(fund_results: list[dict], risk_free: float) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = "\n".join(
        render_fund_section(f, i) for i, f in enumerate(fund_results)
    )
    # Chart data for JS — include chart + daily return data for portfolio analyzer
    chart_payload = []
    for f in fund_results:
        entry = {
            "chart": f["chart"],
            "daily": f["daily"],
            "monthly": f["monthly"],
            "name": f["name"],
            "shortName": f["fund_cd"] if f["member_cd"] == "BENCH" else f["name"],
            "hasKrw": f.get("has_krw", False),
            "isBench": f["member_cd"] == "BENCH",
            "region": f.get("region", ""),
            "currency": f.get("currency_label", "KRW"),
        }
        if f.get("total_fee") is not None:
            entry["fee"] = f["total_fee"]
        if f.get("benchmark"):
            entry["bench"] = f["benchmark"]
        if f.get("price_only"):
            entry["priceOnly"] = True
        if f.get("krw"):
            entry["krw"] = {
                "chart": f["krw"]["chart"],
                "daily": f["krw"]["daily"],
                "monthly": f["krw"]["monthly"],
            }
        if f.get("has_usd") and f.get("usd"):
            entry["hasUsd"] = True
            entry["usd"] = {
                "chart": f["usd"]["chart"],
                "daily": f["usd"]["daily"],
                "monthly": f["usd"]["monthly"],
            }
        if f.get("has_jpy") and f.get("jpy"):
            entry["hasJpy"] = True
            entry["jpy"] = {
                "chart": f["jpy"]["chart"],
                "daily": f["jpy"]["daily"],
                "monthly": f["jpy"]["monthly"],
            }
        chart_payload.append(entry)
    return (
        HTML_TEMPLATE
        .replace("%%GENERATED_AT%%", generated_at)
        .replace("%%RISK_FREE%%", str(risk_free))
        .replace("%%RISK_FREE_DECIMAL%%", str(risk_free / 100.0))
        .replace("%%FUND_SECTIONS%%", sections)
        .replace("%%FUND_JSON%%", json.dumps(chart_payload, ensure_ascii=False))
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="펀드 분석 대시보드 생성")
    ap.add_argument("--fund-list", default="fund_list.csv")
    ap.add_argument("--benchmark-list", default="benchmark_list.csv")
    ap.add_argument("--db", default="data/fund_history.db")
    ap.add_argument("--output", default="data/dashboard.html")
    ap.add_argument("--risk-free", type=float, default=3.5,
                    help="무위험 수익률 (%%); 기본 3.5%%")
    ap.add_argument("--top-drawdowns", type=int, default=5,
                    help="표시할 주요 하락 이벤트 수")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    funds = load_fund_list(args.fund_list)
    if not funds:
        raise SystemExit(f"No funds in {args.fund_list}")

    # Load benchmarks (optional — file may not exist)
    bench_path = Path(args.benchmark_list)
    benchmarks: list[dict] = []
    fund_currency: dict[str, str] = {}  # fundCd → currency (USD, JPY, etc.)
    if bench_path.exists():
        with open(bench_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("fundCd"):
                    benchmarks.append({"memberCd": "BENCH", "fundCd": row["fundCd"],
                                       "name": row.get("name") or row["fundCd"],
                                       "region": row.get("region", ""),
                                       "priceOnly": (row.get("priceOnly") or "").strip()})
                    ccy = row.get("currency", "").upper()
                    if ccy and ccy != "KRW":
                        fund_currency[row["fundCd"]] = ccy

    conn = get_conn(args.db)
    risk_free = args.risk_free / 100.0

    # Load FX rates
    fx_rates: dict[str, pd.Series] = {}
    fx_map = {"USDKRW": "USDKRW", "JPYKRW": "JPYKRW", "USDJPY": "USDJPY"}
    for code in fx_map.values():
        try:
            fx_rates[code] = load_nav_series(conn, "BENCH", code)
        except Exception:
            pass

    all_funds = funds + benchmarks
    results = []
    for f in all_funds:
        label = f.get("name") or f["fundCd"]
        print(f"Analyzing [{label}] ...")

        ccy = fund_currency.get(f["fundCd"])
        krw_nav = None
        jpy_nav = None
        usd_nav = None

        if ccy == "USD":
            foreign_nav = load_nav_series(conn, f["memberCd"], f["fundCd"])
            # USD → KRW
            if "USDKRW" in fx_rates:
                fx = fx_rates["USDKRW"].reindex(foreign_nav.index, method="ffill")
                krw_nav = (foreign_nav * fx).dropna()
            # USD → JPY
            if "USDJPY" in fx_rates:
                fx = fx_rates["USDJPY"].reindex(foreign_nav.index, method="ffill")
                jpy_nav = (foreign_nav * fx).dropna()
        elif ccy == "JPY":
            foreign_nav = load_nav_series(conn, f["memberCd"], f["fundCd"])
            # JPY → KRW
            if "JPYKRW" in fx_rates:
                fx = fx_rates["JPYKRW"].reindex(foreign_nav.index, method="ffill")
                krw_nav = (foreign_nav * fx).dropna()
        else:
            # KRW assets (insurance funds, KS200) → USD and JPY conversion
            krw_asset_nav = load_nav_series(conn, f["memberCd"], f["fundCd"])
            if "USDKRW" in fx_rates:
                fx = fx_rates["USDKRW"].reindex(krw_asset_nav.index, method="ffill")
                usd_nav = (krw_asset_nav / fx).dropna()
            if "JPYKRW" in fx_rates:
                fx = fx_rates["JPYKRW"].reindex(krw_asset_nav.index, method="ffill")
                jpy_nav = (krw_asset_nav / fx).dropna()

        result = analyze_fund(
            conn, f["memberCd"], f["fundCd"], label,
            risk_free, args.top_drawdowns, krw_nav=krw_nav,
        )
        if result:
            result["currency_label"] = ccy or "KRW"
            result["region"] = f.get("region", "")
            # Wrapper + underlying fee, already deducted from the published NAV.
            fee_raw = (f.get("totalFee") or "").strip()
            if fee_raw:
                result["total_fee"] = float(fee_raw)
            bench_raw = (f.get("benchmark") or "").strip()
            if bench_raw:
                result["benchmark"] = bench_raw
            # Indices pay no dividends, so their series is price-only however it is adjusted.
            if f.get("priceOnly"):
                result["price_only"] = True
            # Add USD analysis (for KRW assets)
            if usd_nav is not None and len(usd_nav) >= 30:
                result["has_usd"] = True
                result["usd"] = _build_series_data(usd_nav, risk_free, args.top_drawdowns)
            # Add JPY analysis
            if jpy_nav is not None and len(jpy_nav) >= 30:
                result["has_jpy"] = True
                result["jpy"] = _build_series_data(jpy_nav, risk_free, args.top_drawdowns)
            results.append(result)

    if not results:
        raise SystemExit("No fund data to analyze.")

    html = render_html(results, args.risk_free)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nDashboard saved to {out} ({len(html):,} bytes)")
    print(f"Analyzed {len(results)} fund(s).")


if __name__ == "__main__":
    main()
