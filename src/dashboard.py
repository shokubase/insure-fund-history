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

    daily_returns = nav.pct_change().dropna()
    annual_factor = len(daily_returns) / total_years
    vol = daily_returns.std() * sqrt(annual_factor)

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
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>
<style>
  :root { --bg: #f8f9fa; --card: #fff; --border: #dee2e6; --accent: #2563eb; --red: #dc2626; --green: #16a34a; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: #1a1a1a; line-height: 1.6; padding: 2rem; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 1.8rem; margin-bottom: 0.3rem; }
  .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 2rem; }
  .fund-section { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
                  padding: 2rem; margin-bottom: 2rem; }
  .fund-section h2 { font-size: 1.4rem; margin-bottom: 0.2rem; color: var(--accent); }
  .fund-meta { font-size: 0.85rem; color: #888; margin-bottom: 1.5rem; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                  gap: 1rem; margin-bottom: 1.5rem; }
  .metric-card { background: var(--bg); border-radius: 8px; padding: 0.8rem 1rem; text-align: center; }
  .metric-card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-card .value { font-size: 1.3rem; font-weight: 700; margin-top: 0.2rem; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .chart-container { position: relative; height: 300px; margin-bottom: 1.5rem; }
  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }
  @media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }
  h3 { font-size: 1.1rem; margin: 1.5rem 0 0.8rem; border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 1rem; }
  th, td { padding: 0.5rem 0.8rem; text-align: right; border-bottom: 1px solid var(--border); }
  th { background: var(--bg); font-weight: 600; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  tr:hover { background: #f0f4ff; }
  .ongoing { color: var(--red); font-style: italic; }

  /* Asset filter */
  .asset-filter { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
                  padding: 1.5rem 2rem; margin-bottom: 2rem; }
  .asset-filter h2 { font-size: 1.2rem; margin-bottom: 0.8rem; color: var(--accent); }
  .filter-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
  .filter-chip { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.4rem 0.8rem;
                 border: 1.5px solid var(--border); border-radius: 20px; font-size: 0.85rem;
                 cursor: pointer; transition: all 0.15s; user-select: none; background: var(--bg); }
  .filter-chip:hover { border-color: var(--accent); }
  .filter-chip.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .filter-chip input { display: none; }
  .filter-actions { margin-top: 0.8rem; display: flex; gap: 0.5rem; }
  .filter-actions button { background: none; border: 1px solid var(--border); border-radius: 6px;
                           padding: 0.3rem 0.8rem; font-size: 0.8rem; cursor: pointer; color: #666; }
  .filter-actions button:hover { border-color: var(--accent); color: var(--accent); }
  .fund-section.hidden { display: none; }

  /* Two-column selector layout */
  .selector-columns { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-bottom: 0.8rem; }
  .selector-column h4 { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em;
                         margin-bottom: 0.5rem; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border); }
  .selector-column .filter-chips { margin-bottom: 0; }
  @media (max-width: 640px) { .selector-columns { grid-template-columns: 1fr; } }

  /* Currency toggle */
  .currency-toggle { display: flex; gap: 0; }
  .btn-currency { background: var(--bg); border: 1px solid var(--border); padding: 0.4rem 1rem;
                   font-size: 0.85rem; cursor: pointer; color: #666; transition: all 0.15s; }
  .btn-currency:first-child { border-radius: 6px 0 0 6px; }
  .btn-currency:last-child { border-radius: 0 6px 6px 0; border-left: none; }
  .btn-currency.active { background: var(--accent); color: #fff; border-color: var(--accent); }

  /* Correlation matrix */
  .corr-table { width: auto; margin: 0 auto 1rem; }
  .corr-table th, .corr-table td { text-align: center; min-width: 80px; padding: 0.6rem; font-size: 0.85rem; }
  .corr-table th { background: var(--bg); font-size: 0.75rem; max-width: 120px;
                    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Portfolio analyzer */
  .portfolio-controls { margin-bottom: 1.5rem; }
  .fund-row { display: flex; align-items: center; gap: 0.8rem; padding: 0.5rem 0;
              border-bottom: 1px solid var(--border); }
  .fund-row label { flex: 1; min-width: 0; cursor: pointer; display: flex; align-items: center; gap: 0.5rem; }
  .fund-row label span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .fund-row input[type=number] { width: 70px; padding: 0.3rem 0.5rem; border: 1px solid var(--border);
                                  border-radius: 4px; text-align: right; font-size: 0.9rem; }
  .fund-row input[type=range] { width: 120px; }
  .weight-sum { margin: 0.8rem 0; font-size: 0.9rem; }
  .weight-sum.warn { color: var(--red); font-weight: 600; }
  .btn-analyze { background: var(--accent); color: #fff; border: none; border-radius: 8px;
                 padding: 0.6rem 1.5rem; font-size: 1rem; cursor: pointer; }
  .btn-analyze:hover { opacity: 0.9; }
  .btn-analyze:disabled { background: #aaa; cursor: not-allowed; }
  #portfolio-results { margin-top: 1.5rem; }
</style>
</head>
<body>
<h1>펀드 분석 대시보드</h1>
<p class="subtitle">생성일: %%GENERATED_AT%% | 무위험수익률: %%RISK_FREE%%%</p>

<!-- Asset Filter -->
<div class="asset-filter">
  <h2>자산 선택</h2>
  <div class="selector-columns">
    <div class="selector-column"><h4>보험펀드</h4><div class="filter-chips" id="filter-chips-insurance"></div></div>
    <div class="selector-column"><h4>미국</h4><div class="filter-chips" id="filter-chips-us"></div></div>
    <div class="selector-column"><h4>일본</h4><div class="filter-chips" id="filter-chips-jp"></div></div>
  </div>
  <div class="filter-actions">
    <button id="filter-all">전체 선택</button>
    <button id="filter-none">전체 해제</button>
  </div>
</div>

<!-- Asset Comparison -->
<section class="fund-section" id="comparison-section" style="display:none;">
  <h2>자산 비교</h2>
  <p class="fund-meta" id="comparison-meta"></p>
  <div class="chart-container" style="height:400px;"><canvas id="comparison-chart"></canvas></div>
</section>

%%FUND_SECTIONS%%

<!-- Correlation Matrix Analyzer -->
<section class="fund-section" id="corr-section">
  <h2>상관행렬 분석</h2>
  <p class="fund-meta">자산을 선택하면 바로 상관행렬이 표시됩니다</p>
  <div class="portfolio-controls">
    <div class="selector-columns">
      <div class="selector-column"><h4>보험펀드</h4><div class="filter-chips" id="corr-selector-insurance"></div></div>
      <div class="selector-column"><h4>미국</h4><div class="filter-chips" id="corr-selector-us"></div></div>
      <div class="selector-column"><h4>일본</h4><div class="filter-chips" id="corr-selector-jp"></div></div>
    </div>
  </div>
  <div id="corr-result"></div>
</section>

<!-- Portfolio Analyzer -->
<section class="fund-section" id="portfolio">
  <h2>포트폴리오 분석</h2>
  <p class="fund-meta">펀드를 선택하고 비중을 입력한 뒤 분석 버튼을 클릭하세요</p>
  <div class="portfolio-controls">
    <div class="selector-columns" id="fund-selector">
      <div class="selector-column"><h4>보험펀드</h4><div id="fund-selector-insurance"></div></div>
      <div class="selector-column"><h4>미국</h4><div id="fund-selector-us"></div></div>
      <div class="selector-column"><h4>일본</h4><div id="fund-selector-jp"></div></div>
    </div>
    <div style="margin:0.8rem 0;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
      <div class="weight-sum" id="weight-sum" style="margin:0;">비중 합계: 0%</div>
      <span style="color:#888;">|</span>
      <div style="font-size:0.85rem;display:flex;align-items:center;gap:0.5rem;">
        <label>시작일 <input type="date" id="pf-start" style="padding:0.3rem;border:1px solid var(--border);border-radius:4px;font-size:0.85rem;"></label>
        <label>종료일 <input type="date" id="pf-end" style="padding:0.3rem;border:1px solid var(--border);border-radius:4px;font-size:0.85rem;"></label>
      </div>
      <span id="pf-date-info" style="font-size:0.8rem;color:#888;"></span>
    </div>
    <button class="btn-analyze" id="btn-analyze" disabled>포트폴리오 분석</button>
  </div>
  <div id="portfolio-results" style="display:none">
    <div class="metrics-grid" id="pf-metrics"></div>
    <div class="chart-row">
      <div class="chart-container" style="position:relative;">
        <canvas id="pf-nav-chart"></canvas>
        <div id="pf-selection-overlay" style="display:none;position:absolute;top:0;height:100%;background:rgba(37,99,235,0.1);border-left:1px dashed var(--accent);border-right:1px dashed var(--accent);pointer-events:none;"></div>
        <div id="pf-selection-stats" style="display:none;position:absolute;top:8px;right:8px;background:rgba(255,255,255,0.95);border:1px solid var(--border);border-radius:8px;padding:0.5rem 0.8rem;font-size:0.8rem;line-height:1.5;box-shadow:0 2px 8px rgba(0,0,0,0.1);z-index:10;"></div>
      </div>
      <div class="chart-container"><canvas id="pf-dd-chart"></canvas></div>
    </div>
    <p style="font-size:0.75rem;color:#999;margin-top:-1rem;margin-bottom:1rem;">NAV 차트에서 드래그하여 구간 분석 (클릭하면 해제)</p>
    <div id="pf-yearly"></div>
    <div id="pf-trailing"></div>
    <div id="pf-dd-table"></div>
    <div id="pf-ls-table"></div>
    <div id="pf-corr-table"></div>
  </div>
</section>

<script>
const FUNDS = %%FUND_JSON%%;
const RISK_FREE = %%RISK_FREE_DECIMAL%%;
const CCY_SYMBOL = { USD: '$', JPY: '¥', KRW: '₩' };
function ccySym(fund) { return CCY_SYMBOL[fund.currency] || fund.currency; }

// Generic data getter respecting currency mode
function getDataByMode(fund, mode, key) {
  if (mode === 'krw' && fund.krw && fund.krw[key]) return fund.krw[key];
  if (mode === 'jpy' && fund.jpy && fund.jpy[key]) return fund.jpy[key];
  return fund[key];
}

// Build currency toggle buttons for a fund
function buildCcyToggle(fund, idx, style, stateObj, onChange) {
  if (!fund.hasKrw && !fund.hasJpy) return null;
  const span = document.createElement('span');
  span.className = 'currency-toggle';
  span.style.cssText = style;
  const defaultMode = fund.currency === 'JPY' ? 'orig' : 'krw';
  let btns = `<button class="btn-currency${defaultMode==='orig'?' active':''}" data-mode="orig" style="padding:0.1rem 0.4rem;font-size:0.7rem;border-radius:4px 0 0 4px;">${ccySym(fund)}</button>`;
  btns += `<button class="btn-currency${defaultMode==='krw'?' active':''}" data-mode="krw" style="padding:0.1rem 0.4rem;font-size:0.7rem;border-left:none;">₩</button>`;
  if (fund.hasJpy) {
    btns += `<button class="btn-currency" data-mode="jpy" style="padding:0.1rem 0.4rem;font-size:0.7rem;border-radius:0 4px 4px 0;border-left:none;">¥</button>`;
  } else {
    // close border-radius on last button
    btns = btns.replace('border-left:none;">', 'border-left:none;border-radius:0 4px 4px 0;">');
  }
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

// Fund region helper
function fundRegion(fund) {
  if (!fund.isBench) return 'insurance';
  if (fund.currency === 'JPY') return 'jp';
  return 'us';
}
function chipLabel(fund) {
  if (fund.currency === 'JPY' && fund.isBench) return fund.name;  // JPY: show full name (숫자 코드만으로는 식별 어려움)
  return fund.shortName || fund.name;
}

// ── Asset Filter ──
const filterCurrencyState = {};
FUNDS.forEach((f, i) => { if (f.hasKrw) filterCurrencyState[i] = f.currency === 'JPY' ? 'orig' : 'krw'; });

(function buildFilter() {
  const filterContainers = {
    insurance: document.getElementById('filter-chips-insurance'),
    us: document.getElementById('filter-chips-us'),
    jp: document.getElementById('filter-chips-jp'),
  };
  FUNDS.forEach((fund, idx) => {
    const chip = document.createElement('label');
    chip.className = 'filter-chip';
    chip.innerHTML = `<input type="checkbox" data-idx="${idx}">${chipLabel(fund)}`;

    const fToggle = buildCcyToggle(fund, idx, 'margin:0 0 0 0.3rem;display:inline-flex;', filterCurrencyState, updateComparison);
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

  const allFilterChips = () => document.querySelectorAll('#filter-chips-insurance .filter-chip, #filter-chips-us .filter-chip, #filter-chips-jp .filter-chip');

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
const COMPARISON_COLORS = ['#2563eb','#dc2626','#16a34a','#f59e0b','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1','#14b8a6'];
let comparisonChart = null;

function updateComparison() {
  const section = document.getElementById('comparison-section');
  const selected = [];
  document.querySelectorAll('#filter-chips-insurance input:checked, #filter-chips-us input:checked, #filter-chips-jp input:checked').forEach(cb => {
    selected.push(+cb.dataset.idx);
  });

  if (selected.length < 2) { section.style.display = 'none'; return; }
  section.style.display = '';

  // Find common date range (respecting per-asset currency toggle)
  const dailySets = selected.map(idx => {
    const fund = FUNDS[idx];
    return getDataByMode(fund, filterCurrencyState[idx] || 'krw', 'daily');
  });
  const dateSets = dailySets.map(d => new Set(d.dates));
  const common = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d))).sort();

  if (common.length < 2) { section.style.display = 'none'; return; }

  // Build NAV series normalized to 100 at start
  const datasets = selected.map((idx, si) => {
    const lookup = {};
    dailySets[si].dates.forEach((d, i) => { lookup[d] = dailySets[si].returns[i]; });
    const nav = [100];
    for (let i = 0; i < common.length; i++) {
      nav.push(nav[nav.length - 1] * (1 + (lookup[common[i]] || 0)));
    }
    // nav has length common.length+1, dates need synthetic first date
    return nav;
  });

  const firstDate = new Date(common[0]);
  firstDate.setDate(firstDate.getDate() - 1);
  const chartDates = [firstDate.toISOString().slice(0, 10), ...common];

  // Downsample
  const step = Math.max(1, Math.floor(chartDates.length / 600));
  const dsDates = chartDates.filter((_, i) => i % step === 0);

  document.getElementById('comparison-meta').textContent =
    `공통 기간: ${common[0]} ~ ${common[common.length-1]} | 시작점 = 100으로 정규화`;

  const chartDatasets = selected.map((idx, si) => {
    const dsNav = datasets[si].filter((_, i) => i % step === 0);
    return {
      label: FUNDS[idx].shortName || FUNDS[idx].name,
      data: dsNav.map(v => +v.toFixed(2)),
      borderColor: COMPARISON_COLORS[si % COMPARISON_COLORS.length],
      backgroundColor: 'transparent',
      fill: false, pointRadius: 0, borderWidth: 1.8,
    };
  });

  if (comparisonChart) comparisonChart.destroy();
  comparisonChart = new Chart(document.getElementById('comparison-chart'), {
    type: 'line',
    data: { labels: dsDates, datasets: chartDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: 'time', time: { unit: 'year' }, ticks: { maxTicksLimit: 10 } },
        y: { beginAtZero: false }
      },
      plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 14, font: { size: 11 } } } }
    }
  });
}

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
        {label:wy+'Y CAGR (%)',data:cr.map(v=>+v.toFixed(2)),borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,0.08)',fill:true,pointRadius:0,borderWidth:1.5},
        {label:'평균',data:cd.map(()=>+avg.toFixed(2)),borderColor:'#888',borderDash:[5,5],pointRadius:0,borderWidth:1},
        {label:'0%',data:cd.map(()=>0),borderColor:'#dc2626',borderDash:[3,3],pointRadius:0,borderWidth:1},
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
  const canvas = document.getElementById(canvasId);
  const overlay = document.getElementById(overlayId);
  const statsBox = document.getElementById(statsId);
  if (!canvas || !overlay || !statsBox) return;

  let dragStart = null, dragging = false;

  function clear() { overlay.style.display = 'none'; statsBox.style.display = 'none'; dragStart = null; dragging = false; }

  function showStats(d1, d2) {
    const fullDates = typeof fullDatesOrFn === 'function' ? fullDatesOrFn() : fullDatesOrFn;
    const fullNav = typeof fullNavOrFn === 'function' ? fullNavOrFn() : fullNavOrFn;
    let si = fullDates.findIndex(d => d >= d1);
    let ei = fullDates.length - 1;
    for (let i = fullDates.length - 1; i >= 0; i--) { if (fullDates[i] <= d2) { ei = i; break; } }
    if (si < 0 || si >= ei || ei - si < 2) { statsBox.style.display = 'none'; return; }
    const dates = fullDates.slice(si, ei+1), nav = fullNav.slice(si, ei+1), n = nav.length;
    const totalDays = (new Date(dates[n-1]) - new Date(dates[0])) / 86400000;
    const totalYears = totalDays / 365.25;
    const totalReturn = ((nav[n-1]/nav[0]-1)*100).toFixed(2);
    const cagr = totalYears > 0 ? ((Math.pow(nav[n-1]/nav[0],1/totalYears)-1)*100).toFixed(2) : '-';
    const dr = []; for (let i=1;i<n;i++) dr.push(nav[i]/nav[i-1]-1);
    const mean = dr.reduce((s,v)=>s+v,0)/dr.length;
    const vari = dr.reduce((s,v)=>s+(v-mean)**2,0)/(dr.length-1);
    const af = totalYears > 0 ? dr.length/totalYears : 252;
    const vol = (Math.sqrt(vari)*Math.sqrt(af)*100).toFixed(2);
    let peak = nav[0], mdd = 0;
    for (const v of nav) { peak = Math.max(peak,v); mdd = Math.min(mdd,(v-peak)/peak); }
    const pc = v => +v>0?'positive':+v<0?'negative':'';
    statsBox.innerHTML =
      `<div style="font-weight:600;margin-bottom:0.3rem;">${dates[0]} ~ ${dates[n-1]}</div>`+
      `<div>수익률: <b class="${pc(totalReturn)}">${+totalReturn>0?'+':''}${totalReturn}%</b></div>`+
      `<div>CAGR: <b class="${pc(cagr)}">${+cagr>0?'+':''}${cagr}%</b></div>`+
      `<div>변동성: <b>${vol}%</b></div>`+
      `<div>MDD: <b class="negative">${(mdd*100).toFixed(2)}%</b></div>`;
    statsBox.style.display = 'block';
  }

  canvas.addEventListener('mousedown', (e) => {
    const chart = chartRef();
    if (!chart) return;
    const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left;
    if (x < chart.chartArea.left || x > chart.chartArea.right) return;
    dragStart = x; dragging = true;
    overlay.style.display = 'block'; overlay.style.left = x+'px'; overlay.style.width = '0px';
    statsBox.style.display = 'none';
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const chart = chartRef();
    if (!chart) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(chart.chartArea.left, Math.min(e.clientX-rect.left, chart.chartArea.right));
    overlay.style.left = Math.min(dragStart,x)+'px'; overlay.style.width = Math.abs(x-dragStart)+'px';
  });
  canvas.addEventListener('mouseup', (e) => {
    if (!dragging) return;
    dragging = false;
    const chart = chartRef();
    if (!chart) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(chart.chartArea.left, Math.min(e.clientX-rect.left, chart.chartArea.right));
    if (Math.abs(x-dragStart) < 5) { clear(); return; }
    const scale = chart.scales.x;
    const d1 = new Date(scale.getValueForPixel(Math.min(dragStart,x))).toISOString().slice(0,10);
    const d2 = new Date(scale.getValueForPixel(Math.max(dragStart,x))).toISOString().slice(0,10);
    showStats(d1, d2);
  });
  canvas.addEventListener('mouseleave', () => { if (dragging) dragging = false; });
}

const fundCharts = {};

function rebuildNav(dailyData) {
  const nav = [1000];
  for (let i = 0; i < dailyData.returns.length; i++) nav.push(nav[nav.length-1] * (1 + dailyData.returns[i]));
  const d0 = new Date(dailyData.dates[0]); d0.setDate(d0.getDate()-1);
  return { dates: [d0.toISOString().slice(0,10), ...dailyData.dates], nav };
}

function createSingleChart(idx) {
  const fund = FUNDS[idx];

  // USD charts + drag-select
  const usdNav = renderChart(`chart-${idx}-usd-nav`, fund.chart.dates, fund.chart.nav,
    '#2563eb', { label: '기준가', bg: 'rgba(37,99,235,0.08)' });
  renderChart(`chart-${idx}-usd-dd`, fund.chart.dates, fund.chart.drawdown,
    '#dc2626', { label: '드로다운 (%)', bg: 'rgba(220,38,38,0.15)', yOpts: { max: 0 } });
  fundCharts[`${idx}-usd`] = usdNav;
  const usdFull = rebuildNav(fund.daily);
  attachDragSelect(`chart-${idx}-usd-nav`, `chart-${idx}-usd-overlay`, `chart-${idx}-usd-stats`,
    () => fundCharts[`${idx}-usd`], usdFull.dates, usdFull.nav);

  // KRW charts + drag-select
  if (fund.krw) {
    const krwNav = renderChart(`chart-${idx}-krw-nav`, fund.krw.chart.dates, fund.krw.chart.nav,
      '#2563eb', { label: '기준가 (KRW)', bg: 'rgba(37,99,235,0.08)' });
    renderChart(`chart-${idx}-krw-dd`, fund.krw.chart.dates, fund.krw.chart.drawdown,
      '#dc2626', { label: '드로다운 (%)', bg: 'rgba(220,38,38,0.15)', yOpts: { max: 0 } });
    fundCharts[`${idx}-krw`] = krwNav;
    const krwFull = rebuildNav(fund.krw.daily);
    attachDragSelect(`chart-${idx}-krw-nav`, `chart-${idx}-krw-overlay`, `chart-${idx}-krw-stats`,
      () => fundCharts[`${idx}-krw`], krwFull.dates, krwFull.nav);
  }

  // JPY charts + drag-select (if available)
  if (fund.jpy) {
    const jpyNav = renderChart(`chart-${idx}-jpy-nav`, fund.jpy.chart.dates, fund.jpy.chart.nav,
      '#2563eb', { label: '기준가 (JPY)', bg: 'rgba(37,99,235,0.08)' });
    renderChart(`chart-${idx}-jpy-dd`, fund.jpy.chart.dates, fund.jpy.chart.drawdown,
      '#dc2626', { label: '드로다운 (%)', bg: 'rgba(220,38,38,0.15)', yOpts: { max: 0 } });
    fundCharts[`${idx}-jpy`] = jpyNav;
    const jpyFull = rebuildNav(fund.jpy.daily);
    attachDragSelect(`chart-${idx}-jpy-nav`, `chart-${idx}-jpy-overlay`, `chart-${idx}-jpy-stats`,
      () => fundCharts[`${idx}-jpy`], jpyFull.dates, jpyFull.nav);
  }

  // Init trailing returns
  initFundTrailing(`chart-${idx}-usd`, fund.daily);
  if (fund.krw) initFundTrailing(`chart-${idx}-krw`, fund.krw.daily);
  if (fund.jpy) initFundTrailing(`chart-${idx}-jpy`, fund.jpy.daily);
}

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

// ── Correlation Matrix Analyzer ──
(function buildCorrSelector() {
  const corrContainers = {
    insurance: document.getElementById('corr-selector-insurance'),
    us: document.getElementById('corr-selector-us'),
    jp: document.getElementById('corr-selector-jp'),
  };
  const corrCurrencyState = {};
  FUNDS.forEach((f, i) => { if (f.hasKrw) corrCurrencyState[i] = f.currency === 'JPY' ? 'orig' : 'krw'; });

  FUNDS.forEach((fund, idx) => {
    const chip = document.createElement('label');
    chip.className = 'filter-chip';
    chip.dataset.idx = idx;
    chip.innerHTML = `<input type="checkbox" data-idx="${idx}">${chipLabel(fund)}`;

    const cToggle = buildCcyToggle(fund, idx, 'margin:0 0 0 0.3rem;display:inline-flex;', corrCurrencyState, updateCorrMatrix);
    if (cToggle) chip.appendChild(cToggle);

    chip.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-currency')) return;
      setTimeout(() => {
        const checked = chip.querySelector('input').checked;
        chip.classList.toggle('active', checked);
        updateCorrMatrix();
      }, 0);
    });
    corrContainers[fundRegion(fund)].appendChild(chip);
  });

  function getCorrData(fund, idx, key) {
    return getDataByMode(fund, corrCurrencyState[idx] || 'krw', key);
  }

  function updateCorrMatrix() {
    const el = document.getElementById('corr-result');
    const selected = [];
    document.querySelectorAll('#corr-selector-insurance input:checked, #corr-selector-us input:checked, #corr-selector-jp input:checked').forEach(cb => {
      selected.push(+cb.dataset.idx);
    });

    if (selected.length < 2) { el.innerHTML = '<p style="color:#888;">2개 이상 선택하세요</p>'; return; }

    // Build monthly return data
    const fundData = selected.map(idx => {
      const f = getCorrData(FUNDS[idx], idx, 'monthly');
      const m = {};
      f.dates.forEach((d, i) => { m[d] = f.returns[i]; });
      return { name: FUNDS[idx].shortName, map: m, dates: new Set(f.dates) };
    });

    let common = [...fundData[0].dates].filter(d => fundData.every(f => f.dates.has(d))).sort();
    if (common.length < 6) { el.innerHTML = '<p style="color:#888;">공통 기간 부족 (최소 6개월 필요)</p>'; return; }

    const arrays = fundData.map(f => common.map(d => f.map[d]));
    const n = common.length;
    const names = fundData.map(f => f.name);
    const means = arrays.map(arr => arr.reduce((s, v) => s + v, 0) / n);

    const matrix = [];
    for (let i = 0; i < arrays.length; i++) {
      const row = [];
      for (let j = 0; j < arrays.length; j++) {
        let sumXY = 0, sumX2 = 0, sumY2 = 0;
        for (let k = 0; k < n; k++) {
          const dx = arrays[i][k] - means[i];
          const dy = arrays[j][k] - means[j];
          sumXY += dx * dy; sumX2 += dx * dx; sumY2 += dy * dy;
        }
        const denom = Math.sqrt(sumX2 * sumY2);
        row.push(denom > 0 ? sumXY / denom : 0);
      }
      matrix.push(row);
    }

    function cellStyle(v) {
      if (v >= 1) return 'background:#1d4ed8;color:#fff;';
      if (v >= 0) return `background:rgba(37,99,235,${(v*0.5).toFixed(2)});color:${v>0.7?'#fff':'#1a1a1a'};`;
      return `background:rgba(220,38,38,${(Math.abs(v)*0.5).toFixed(2)});color:${v<-0.7?'#fff':'#1a1a1a'};`;
    }

    const header = '<tr><th></th>' + names.map(n => `<th>${n}</th>`).join('') + '</tr>';
    const rows = matrix.map((row, i) =>
      '<tr><th>' + names[i] + '</th>' + row.map(v => `<td style="${cellStyle(v)}">${v.toFixed(2)}</td>`).join('') + '</tr>'
    ).join('');

    el.innerHTML = `
      <p class="fund-meta">월간 수익률 기준 | 공통 기간: ${common[0]} ~ ${common[common.length-1]} (${n}개월)</p>
      <table class="corr-table">${header}${rows}</table>`;
  }
})();

// ── Portfolio Analyzer ──

// Per-fund currency mode for portfolio
const pfFundCurrency = {};
FUNDS.forEach((f, i) => { if (f.hasKrw) pfFundCurrency[i] = f.currency === 'JPY' ? 'orig' : 'krw'; });

// Build fund selector UI
(function buildSelector() {
  const pfContainers = {
    insurance: document.getElementById('fund-selector-insurance'),
    us: document.getElementById('fund-selector-us'),
    jp: document.getElementById('fund-selector-jp'),
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
      if (!cb.checked) { num.value = ''; }
      else if (!num.value || +num.value === 0) { num.value = 20; }
      updateWeightSum();
    });
    num.addEventListener('input', () => { cb.checked = +num.value > 0; updateWeightSum(); });
  });
})();

function togglePfCurrency(btn) {
  const idx = btn.dataset.idx;
  const mode = btn.dataset.mode;
  pfFundCurrency[idx] = mode;
  btn.parentElement.querySelectorAll('.btn-currency').forEach(b => b.classList.toggle('active', b === btn));
}

function getPfFundData(fund, idx, key) {
  return getDataByMode(fund, pfFundCurrency[idx] || 'krw', key);
}

function getSelections() {
  const rows = document.querySelectorAll('#fund-selector-insurance .fund-row, #fund-selector-us .fund-row, #fund-selector-jp .fund-row');
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

  // Update common date range info
  const info = document.getElementById('pf-date-info');
  const startInput = document.getElementById('pf-start');
  const endInput = document.getElementById('pf-end');
  if (sel.length === 0) {
    info.textContent = '';
    startInput.value = ''; endInput.value = '';
    startInput.min = ''; startInput.max = '';
    endInput.min = ''; endInput.max = '';
    return;
  }
  const dailySets = sel.map(s => getPfFundData(FUNDS[s.idx], s.idx, 'daily'));
  const dateSets = dailySets.map(d => new Set(d.dates));
  const common = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d))).sort();
  if (common.length === 0) { info.textContent = '공통 기간 없음'; return; }
  const earliest = common[0];
  const latest = common[common.length - 1];
  // Find which asset(s) determined the start date
  const bottlenecks = sel.filter(s => {
    const d = getPfFundData(FUNDS[s.idx], s.idx, 'daily');
    return d.dates[0] >= earliest;
  }).map(s => chipLabel(FUNDS[s.idx]));
  const bottleneckStr = bottlenecks.length > 0 ? ` (${bottlenecks.join(', ')})` : '';
  info.textContent = `공통 기간: ${earliest} ~ ${latest}${bottleneckStr}`;
  startInput.min = earliest; startInput.max = latest;
  endInput.min = earliest; endInput.max = latest;
  if (!startInput.value || startInput.value < earliest) startInput.value = earliest;
  if (!endInput.value || endInput.value > latest) endInput.value = latest;
}

// Build portfolio NAV from weighted daily returns
function buildPortfolio(selections) {
  // Find common date range (respecting per-fund currency mode)
  const dailySets = selections.map(s => getPfFundData(FUNDS[s.idx], s.idx, 'daily'));
  const dateSets = dailySets.map(d => new Set(d.dates));
  let commonDates = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d))).sort();

  // Apply user date range filter
  const startDate = document.getElementById('pf-start').value;
  const endDate = document.getElementById('pf-end').value;
  if (startDate) commonDates = commonDates.filter(d => d >= startDate);
  if (endDate) commonDates = commonDates.filter(d => d <= endDate);

  // Build date→return lookup per fund
  const lookups = dailySets.map(f => {
    const m = {};
    f.dates.forEach((d, i) => { m[d] = f.returns[i]; });
    return m;
  });

  const dates = commonDates;
  const returns = dates.map(d => {
    let r = 0;
    selections.forEach((s, si) => { r += lookups[si][d] * s.weight; });
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
function calcMetrics(dates, nav) {
  const n = nav.length;
  const firstDate = new Date(dates[0]);
  const lastDate = new Date(dates[n - 1]);
  const totalYears = (lastDate - firstDate) / (365.25 * 86400000);
  if (totalYears <= 0) return null;

  const totalReturn = (nav[n - 1] / nav[0] - 1) * 100;
  const cagr = (Math.pow(nav[n - 1] / nav[0], 1 / totalYears) - 1);

  // Daily returns (from NAV)
  const dr = [];
  for (let i = 1; i < n; i++) dr.push(nav[i] / nav[i - 1] - 1);
  const mean = dr.reduce((s, v) => s + v, 0) / dr.length;
  const variance = dr.reduce((s, v) => s + (v - mean) ** 2, 0) / (dr.length - 1);
  const annualFactor = dr.length / totalYears;
  const vol = Math.sqrt(variance) * Math.sqrt(annualFactor);
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
          borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.08)',
          fill: true, pointRadius: 0, borderWidth: 1.5 },
        { label: '평균', data: cDates.map(() => +avg.toFixed(2)),
          borderColor: '#888', borderDash: [5, 5], pointRadius: 0, borderWidth: 1 },
        { label: '0%', data: cDates.map(() => 0),
          borderColor: '#dc2626', borderDash: [3, 3], pointRadius: 0, borderWidth: 1 },
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
    <div class="metric-card"><div class="label">변동성</div><div class="value">${m.volatility}%</div></div>
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
      borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.08)',
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
      borderColor: '#dc2626', backgroundColor: 'rgba(220,38,38,0.15)',
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
    document.getElementById('pf-ls-table').innerHTML = '<p style="color:#888;">데이터 부족으로 LS vs DCA 분석 불가</p>';
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

  let common = [...fundData[0].dates].filter(d => fundData.every(f => f.dates.has(d))).sort();
  if (common.length < 30) return null;

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
    if (v >= 0) return `background:rgba(37,99,235,${(v*0.5).toFixed(2)});color:${v>0.7?'#fff':'#1a1a1a'};`;
    return `background:rgba(220,38,38,${(Math.abs(v)*0.5).toFixed(2)});color:${v<-0.7?'#fff':'#1a1a1a'};`;
  }

  const header = '<tr><th></th>' + corr.names.map(n => `<th>${n}</th>`).join('') + '</tr>';
  const rows = corr.matrix.map((row, i) =>
    '<tr><th>' + corr.names[i] + '</th>' +
    row.map(v => `<td style="${cellStyle(v)}">${v.toFixed(2)}</td>`).join('') + '</tr>'
  ).join('');

  el.innerHTML = `
    <h3>상관행렬 (Correlation Matrix)</h3>
    <p class="fund-meta">월간 수익률 기준 | 공통 기간 관측수: ${corr.obs.toLocaleString()}개월</p>
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

  // For each asset: build NAV from daily returns, compute yearly returns
  const assetData = selections.map(s => {
    const daily = getPfFundData(FUNDS[s.idx], s.idx, 'daily');
    const nav = [1000];
    const dates = [];
    // Synthetic first date
    const d0 = new Date(daily.dates[0]); d0.setDate(d0.getDate() - 1);
    dates.push(d0.toISOString().slice(0, 10));
    for (let i = 0; i < daily.returns.length; i++) {
      nav.push(nav[nav.length - 1] * (1 + daily.returns[i]));
      dates.push(daily.dates[i]);
    }
    return { name: chipLabel(FUNDS[s.idx]), weight: s.weight, dates, nav };
  });

  // Compute yearly return for a (dates, nav) series
  function yearlyReturn(dates, nav, year) {
    // Find first and last index in this year
    let si = -1, ei = -1;
    // Find last date of previous year or first date of this year
    for (let i = 0; i < dates.length; i++) {
      const y = +dates[i].slice(0, 4);
      if (y === year && si < 0) si = Math.max(0, i - 1); // use prev day as start
      if (y === year) ei = i;
    }
    if (si < 0 || ei <= si) return null;
    return (nav[ei] / nav[si] - 1) * 100;
  }

  // Build table
  const names = assetData.map(a => a.name);
  const weights = assetData.map(a => a.weight);

  let header = '<tr><th>연도</th>';
  names.forEach((n, i) => { header += `<th>${n}<br><span style="font-weight:400;color:#888;">${(weights[i]*100).toFixed(0)}%</span></th>`; });
  header += '<th style="border-left:2px solid var(--border);">포트폴리오</th></tr>';

  let rows = '';
  for (const year of years) {
    const assetReturns = assetData.map(a => yearlyReturn(a.dates, a.nav, year));
    const pfReturn = yearlyReturn(pf.dates, pf.nav, year);

    // Contribution = asset return × weight
    let row = `<tr><td><b>${year}</b></td>`;
    assetReturns.forEach((r, i) => {
      if (r === null) { row += '<td>-</td>'; return; }
      const cls = r > 0 ? 'positive' : r < 0 ? 'negative' : '';
      const contrib = (r * weights[i]).toFixed(2);
      row += `<td class="${cls}">${r > 0 ? '+' : ''}${r.toFixed(2)}%<br><span style="font-size:0.75rem;color:#888;">기여 ${+contrib > 0 ? '+' : ''}${contrib}%p</span></td>`;
    });
    // Portfolio total
    if (pfReturn !== null) {
      const cls = pfReturn > 0 ? 'positive' : pfReturn < 0 ? 'negative' : '';
      row += `<td style="border-left:2px solid var(--border);" class="${cls}"><b>${pfReturn > 0 ? '+' : ''}${pfReturn.toFixed(2)}%</b></td>`;
    } else {
      row += '<td style="border-left:2px solid var(--border);">-</td>';
    }
    row += '</tr>';
    rows += row;
  }

  el.innerHTML = `
    <h3>연도별 자산 수익률 및 기여도</h3>
    <div style="overflow-x:auto;">
    <table style="min-width:100%;">
      ${header}
      ${rows}
    </table>
    </div>`;
}

document.getElementById('btn-analyze').addEventListener('click', () => {
  const sel = getSelections();
  if (sel.length === 0) return;
  const pf = buildPortfolio(sel);
  renderPortfolio(pf);
  renderYearlyBreakdown(sel, pf);
  renderCorrelation(sel);
  clearSelection();
});

// ── Drag-to-select on portfolio NAV chart ──
attachDragSelect('pf-nav-chart', 'pf-selection-overlay', 'pf-selection-stats',
  () => pfNavChart, () => pfFullDates, () => pfFullNav);

function clearSelection() {
  document.getElementById('pf-selection-overlay').style.display = 'none';
  document.getElementById('pf-selection-stats').style.display = 'none';
}
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
    <div class="metrics-grid">
      <div class="metric-card"><div class="label">기간</div><div class="value">{b['total_years']}년</div></div>
      <div class="metric-card"><div class="label">총 수익률</div><div class="value {_pct_class(b['total_return'])}">{_fmt_pct(b['total_return'])}</div></div>
      <div class="metric-card"><div class="label">CAGR</div><div class="value {_pct_class(b['cagr'])}">{_fmt_pct(b['cagr'])}</div></div>
      <div class="metric-card"><div class="label">변동성</div><div class="value">{b['volatility']:.2f}%</div></div>
      <div class="metric-card"><div class="label">샤프비율</div><div class="value">{b['sharpe']:.2f}</div></div>
      <div class="metric-card"><div class="label">MDD</div><div class="value negative">{_fmt_pct(b['mdd'], False)}</div></div>
      <div class="metric-card"><div class="label">평균 하락폭</div><div class="value negative">-{ds['avg_drawdown']:.2f}%</div></div>
      <div class="metric-card"><div class="label">최장 하락 기간</div><div class="value">{ds['longest_days']}일</div></div>
    </div>"""

    charts = f"""\
    <div class="chart-row">
      <div class="chart-container" style="position:relative;">
        <canvas id="{canvas_id_prefix}-nav"></canvas>
        <div class="drag-overlay" id="{canvas_id_prefix}-overlay" style="display:none;position:absolute;top:0;height:100%;background:rgba(37,99,235,0.1);border-left:1px dashed var(--accent);border-right:1px dashed var(--accent);pointer-events:none;"></div>
        <div class="drag-stats" id="{canvas_id_prefix}-stats" style="display:none;position:absolute;top:8px;right:8px;background:rgba(255,255,255,0.95);border:1px solid var(--border);border-radius:8px;padding:0.5rem 0.8rem;font-size:0.8rem;line-height:1.5;box-shadow:0 2px 8px rgba(0,0,0,0.1);z-index:10;"></div>
      </div>
      <div class="chart-container"><canvas id="{canvas_id_prefix}-dd"></canvas></div>
    </div>
    <p style="font-size:0.75rem;color:#999;margin-top:-1rem;margin-bottom:1rem;">차트에서 드래그하여 구간 분석</p>"""

    events = data["top_events"]
    if events:
        rows = ""
        for i, e in enumerate(events, 1):
            end_str = e["end"] if e["end"] else '<span class="ongoing">진행중</span>'
            rows += f"<tr><td>{i}</td><td>{e['start']}</td><td>{e['trough']}</td>"
            rows += f"<td>{end_str}</td><td class='negative'>{e['depth']:.2f}%</td>"
            rows += f"<td>{e['duration_days']:,}일</td></tr>\n"
        dd_table = f"""\
    <h3>주요 하락 이벤트 (Top {len(events)})</h3>
    <table><tr><th>#</th><th>시작</th><th>저점</th><th>회복</th><th>하락폭</th><th>기간</th></tr>
      {rows}</table>"""
    else:
        dd_table = ""

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
        ls_table = f"""\
    <h3>LS vs DCA 분석</h3>
    <table><tr><th>기간</th><th>관측수</th><th>LS 승률</th><th>MLSA</th><th>MLSD</th></tr>
      {ls_rows}</table>"""
    else:
        ls_table = '<p style="color:#888;">데이터 부족으로 LS vs DCA 분석 불가</p>'

    trailing = f'<div class="trailing-section" data-prefix="{canvas_id_prefix}"></div>'

    return f"{metrics}\n{charts}\n{trailing}\n{dd_table}\n{ls_table}"


def render_fund_section(fund: dict, idx: int) -> str:
    b = fund["basic"]
    has_krw = fund.get("has_krw", False) and "krw" in fund

    # Currency toggle for foreign currency assets
    has_jpy = fund.get("has_jpy", False) and "jpy" in fund
    ccy_label = fund.get("currency_label", "USD")

    is_jpy_asset = ccy_label == "JPY"
    default_view = "orig" if is_jpy_asset else "krw"

    toggle_html = ""
    if has_krw or has_jpy:
        orig_active = " active" if default_view == "orig" else ""
        krw_active = " active" if default_view == "krw" else ""
        btns = f'<button class="btn-currency{orig_active}" data-view="fund-{idx}-orig" onclick="toggleFundView(this)">{ccy_label}</button>'
        btns += f'<button class="btn-currency{krw_active}" data-view="fund-{idx}-krw" onclick="toggleFundView(this)">KRW</button>'
        if has_jpy:
            btns += f'<button class="btn-currency" data-view="fund-{idx}-jpy" onclick="toggleFundView(this)">JPY</button>'
        toggle_html = f'<div class="currency-toggle" style="margin-bottom:1rem;" data-group="fund-{idx}">{btns}</div>'

    orig_block = _render_analysis_block(fund, f"chart-{idx}-usd")
    orig_hide = '' if (default_view == "orig" or not (has_krw or has_jpy)) else ' style="display:none"'
    orig_div = f'<div id="fund-{idx}-orig"{orig_hide}>{orig_block}</div>'

    krw_div = ""
    if has_krw:
        krw_hide = '' if default_view == "krw" else ' style="display:none"'
        krw_block = _render_analysis_block(fund["krw"], f"chart-{idx}-krw")
        krw_div = f'<div id="fund-{idx}-krw"{krw_hide}>{krw_block}</div>'

    jpy_div = ""
    if has_jpy:
        jpy_block = _render_analysis_block(fund["jpy"], f"chart-{idx}-jpy")
        jpy_div = f'<div id="fund-{idx}-jpy" style="display:none">{jpy_block}</div>'

    return f"""\
<section class="fund-section hidden" id="fund-{idx}">
  <h2>{fund['name']}</h2>
  <p class="fund-meta">{fund['member_cd']} / {fund['fund_cd']} | {b['first_date']} ~ {b['last_date']}</p>
  {toggle_html}
  {orig_div}
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
            return f"background: rgba(37,99,235,{opacity * 0.5:.2f}); color: {'#fff' if val > 0.7 else '#1a1a1a'};"
        else:
            opacity = abs(val)
            return f"background: rgba(220,38,38,{opacity * 0.5:.2f}); color: {'#fff' if val < -0.7 else '#1a1a1a'};"

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
            "currency": f.get("currency_label", "KRW"),
        }
        if f.get("krw"):
            entry["krw"] = {
                "chart": f["krw"]["chart"],
                "daily": f["krw"]["daily"],
                "monthly": f["krw"]["monthly"],
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
                                       "name": row.get("name") or row["fundCd"]})
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

        result = analyze_fund(
            conn, f["memberCd"], f["fundCd"], label,
            risk_free, args.top_drawdowns, krw_nav=krw_nav,
        )
        if result:
            result["currency_label"] = ccy or "KRW"
            # Add JPY analysis for USD assets
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
