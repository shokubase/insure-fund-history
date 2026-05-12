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
  <div class="filter-chips" id="filter-chips"></div>
  <div class="filter-actions">
    <button id="filter-all">전체 선택</button>
    <button id="filter-none">전체 해제</button>
  </div>
</div>

%%FUND_SECTIONS%%

<!-- Correlation Matrix Analyzer -->
<section class="fund-section" id="corr-section">
  <h2>상관행렬 분석</h2>
  <p class="fund-meta">자산을 선택하면 바로 상관행렬이 표시됩니다</p>
  <div class="portfolio-controls">
    <div class="filter-chips" id="corr-selector"></div>
  </div>
  <div id="corr-result"></div>
</section>

<!-- Portfolio Analyzer -->
<section class="fund-section" id="portfolio">
  <h2>포트폴리오 분석</h2>
  <p class="fund-meta">펀드를 선택하고 비중을 입력한 뒤 분석 버튼을 클릭하세요</p>
  <div class="portfolio-controls">
    <div id="fund-selector"></div>
    <div class="weight-sum" id="weight-sum">비중 합계: 0%</div>
    <button class="btn-analyze" id="btn-analyze" disabled>포트폴리오 분석</button>
  </div>
  <div id="portfolio-results" style="display:none">
    <div class="metrics-grid" id="pf-metrics"></div>
    <div class="chart-row">
      <div class="chart-container"><canvas id="pf-nav-chart"></canvas></div>
      <div class="chart-container"><canvas id="pf-dd-chart"></canvas></div>
    </div>
    <div id="pf-dd-table"></div>
    <div id="pf-ls-table"></div>
    <div id="pf-corr-table"></div>
  </div>
</section>

<script>
const FUNDS = %%FUND_JSON%%;
const RISK_FREE = %%RISK_FREE_DECIMAL%%;

// ── Asset Filter ──
(function buildFilter() {
  const container = document.getElementById('filter-chips');
  FUNDS.forEach((fund, idx) => {
    const chip = document.createElement('label');
    chip.className = 'filter-chip';
    chip.innerHTML = `<input type="checkbox" data-idx="${idx}">${fund.shortName || fund.name}`;
    chip.addEventListener('click', () => {
      setTimeout(() => {
        const checked = chip.querySelector('input').checked;
        chip.classList.toggle('active', checked);
        const section = document.getElementById('fund-' + idx);
        if (section) section.classList.toggle('hidden', !checked);
        // Lazy-init charts on first show
        if (checked && !section._chartsCreated) {
          createSingleChart(idx);
          section._chartsCreated = true;
        }
      }, 0);
    });
    container.appendChild(chip);
  });

  document.getElementById('filter-all').addEventListener('click', () => {
    container.querySelectorAll('.filter-chip').forEach(chip => {
      const cb = chip.querySelector('input');
      if (!cb.checked) { cb.checked = true; chip.classList.add('active'); }
      const idx = cb.dataset.idx;
      const section = document.getElementById('fund-' + idx);
      if (section) { section.classList.remove('hidden');
        if (!section._chartsCreated) { createSingleChart(+idx); section._chartsCreated = true; }
      }
    });
  });

  document.getElementById('filter-none').addEventListener('click', () => {
    container.querySelectorAll('.filter-chip').forEach(chip => {
      const cb = chip.querySelector('input');
      cb.checked = false; chip.classList.remove('active');
      const section = document.getElementById('fund-' + cb.dataset.idx);
      if (section) section.classList.add('hidden');
    });
  });
})();

function renderChart(canvasId, labels, data, color, opts) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  new Chart(ctx, {
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

function createSingleChart(idx) {
  const fund = FUNDS[idx];
  // USD charts
  renderChart(`chart-${idx}-usd-nav`, fund.chart.dates, fund.chart.nav,
    '#2563eb', { label: '기준가', bg: 'rgba(37,99,235,0.08)' });
  renderChart(`chart-${idx}-usd-dd`, fund.chart.dates, fund.chart.drawdown,
    '#dc2626', { label: '드로다운 (%)', bg: 'rgba(220,38,38,0.15)', yOpts: { max: 0 } });
  // KRW charts (if available)
  if (fund.krw) {
    renderChart(`chart-${idx}-krw-nav`, fund.krw.chart.dates, fund.krw.chart.nav,
      '#2563eb', { label: '기준가 (KRW)', bg: 'rgba(37,99,235,0.08)' });
    renderChart(`chart-${idx}-krw-dd`, fund.krw.chart.dates, fund.krw.chart.drawdown,
      '#dc2626', { label: '드로다운 (%)', bg: 'rgba(220,38,38,0.15)', yOpts: { max: 0 } });
  }
}

function toggleCurrency(btn) {
  const target = document.getElementById(btn.dataset.target);
  const pair = document.getElementById(btn.dataset.pair);
  if (!target || !pair) return;
  target.style.display = '';
  pair.style.display = 'none';
  btn.classList.add('active');
  btn.parentElement.querySelectorAll('.btn-currency').forEach(b => { if (b !== btn) b.classList.remove('active'); });
}

// ── Correlation Matrix Analyzer ──
(function buildCorrSelector() {
  const container = document.getElementById('corr-selector');
  const corrCurrencyState = {};

  FUNDS.forEach((fund, idx) => {
    const chip = document.createElement('label');
    chip.className = 'filter-chip';
    chip.dataset.idx = idx;
    chip.innerHTML = `<input type="checkbox" data-idx="${idx}">${fund.shortName || fund.name}`;

    // USD/KRW sub-toggle for USD assets
    if (fund.hasKrw) {
      const toggle = document.createElement('span');
      toggle.className = 'currency-toggle';
      toggle.style.cssText = 'margin:0 0 0 0.3rem;display:inline-flex;';
      toggle.innerHTML =
        `<button class="btn-currency active" data-idx="${idx}" data-mode="usd" style="padding:0.1rem 0.4rem;font-size:0.7rem;border-radius:4px 0 0 4px;">$</button>` +
        `<button class="btn-currency" data-idx="${idx}" data-mode="krw" style="padding:0.1rem 0.4rem;font-size:0.7rem;border-radius:0 4px 4px 0;border-left:none;">₩</button>`;
      toggle.querySelectorAll('.btn-currency').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.preventDefault(); e.stopPropagation();
          corrCurrencyState[idx] = btn.dataset.mode;
          toggle.querySelectorAll('.btn-currency').forEach(b => b.classList.toggle('active', b === btn));
          updateCorrMatrix();
        });
      });
      chip.appendChild(toggle);
    }

    chip.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-currency')) return;
      setTimeout(() => {
        const checked = chip.querySelector('input').checked;
        chip.classList.toggle('active', checked);
        updateCorrMatrix();
      }, 0);
    });
    container.appendChild(chip);
  });

  function getCorrData(fund, idx, key) {
    if (corrCurrencyState[idx] === 'krw' && fund.krw && fund.krw[key]) return fund.krw[key];
    return fund[key];
  }

  function updateCorrMatrix() {
    const el = document.getElementById('corr-result');
    const selected = [];
    container.querySelectorAll('input[type=checkbox]:checked').forEach(cb => {
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

// Build fund selector UI
(function buildSelector() {
  const container = document.getElementById('fund-selector');
  FUNDS.forEach((fund, idx) => {
    const row = document.createElement('div');
    row.className = 'fund-row';
    const krwToggle = fund.hasKrw
      ? `<span class="currency-toggle" style="margin:0;">` +
        `<button class="btn-currency active" data-idx="${idx}" data-mode="usd" onclick="togglePfCurrency(this)" style="padding:0.2rem 0.5rem;font-size:0.75rem;">USD</button>` +
        `<button class="btn-currency" data-idx="${idx}" data-mode="krw" onclick="togglePfCurrency(this)" style="padding:0.2rem 0.5rem;font-size:0.75rem;">KRW</button></span>`
      : '';
    row.innerHTML = `
      <label><input type="checkbox" data-idx="${idx}">
        <span>${fund.name}</span></label>
      ${krwToggle}
      <input type="range" min="0" max="100" value="0" data-idx="${idx}">
      <input type="number" min="0" max="100" value="0" data-idx="${idx}" style="width:70px"> %`;
    container.appendChild(row);

    const cb = row.querySelector('input[type=checkbox]');
    const slider = row.querySelector('input[type=range]');
    const num = row.querySelector('input[type=number]');

    cb.addEventListener('change', () => {
      if (!cb.checked) { slider.value = 0; num.value = 0; }
      updateWeightSum();
    });
    slider.addEventListener('input', () => { num.value = slider.value; cb.checked = +slider.value > 0; updateWeightSum(); });
    num.addEventListener('input', () => { slider.value = num.value; cb.checked = +num.value > 0; updateWeightSum(); });
  });
})();

// Per-fund currency mode for portfolio
const pfFundCurrency = {};
function togglePfCurrency(btn) {
  const idx = btn.dataset.idx;
  const mode = btn.dataset.mode;
  pfFundCurrency[idx] = mode;
  btn.parentElement.querySelectorAll('.btn-currency').forEach(b => b.classList.toggle('active', b === btn));
}

function getPfFundData(fund, idx, key) {
  if (pfFundCurrency[idx] === 'krw' && fund.krw && fund.krw[key]) return fund.krw[key];
  return fund[key];
}

function getSelections() {
  const rows = document.querySelectorAll('#fund-selector .fund-row');
  const sel = [];
  rows.forEach((row, idx) => {
    const cb = row.querySelector('input[type=checkbox]');
    const w = +row.querySelector('input[type=number]').value;
    if (cb.checked && w > 0) sel.push({ idx, weight: w / 100 });
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
}

// Build portfolio NAV from weighted daily returns
function buildPortfolio(selections) {
  // Find common date range (respecting per-fund currency mode)
  const dailySets = selections.map(s => getPfFundData(FUNDS[s.idx], s.idx, 'daily'));
  const dateSets = dailySets.map(d => new Set(d.dates));
  let commonDates = [...dateSets[0]].filter(d => dateSets.every(ds => ds.has(d))).sort();

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

// Render portfolio results
let pfNavChart = null, pfDdChart = null;

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

document.getElementById('btn-analyze').addEventListener('click', () => {
  const sel = getSelections();
  if (sel.length === 0) return;
  const pf = buildPortfolio(sel);
  renderPortfolio(pf);
  renderCorrelation(sel);
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
      <div class="chart-container"><canvas id="{canvas_id_prefix}-nav"></canvas></div>
      <div class="chart-container"><canvas id="{canvas_id_prefix}-dd"></canvas></div>
    </div>"""

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

    return f"{metrics}\n{charts}\n{dd_table}\n{ls_table}"


def render_fund_section(fund: dict, idx: int) -> str:
    b = fund["basic"]
    has_krw = fund.get("has_krw", False) and "krw" in fund

    # Currency toggle for USD assets
    toggle_html = ""
    if has_krw:
        toggle_html = f"""\
  <div class="currency-toggle" style="margin-bottom:1rem;">
    <button class="btn-currency active" data-target="fund-{idx}-usd" data-pair="fund-{idx}-krw" onclick="toggleCurrency(this)">USD</button>
    <button class="btn-currency" data-target="fund-{idx}-krw" data-pair="fund-{idx}-usd" onclick="toggleCurrency(this)">KRW 환산</button>
  </div>"""

    usd_block = _render_analysis_block(fund, f"chart-{idx}-usd")
    usd_div = f'<div id="fund-{idx}-usd">{usd_block}</div>'

    krw_div = ""
    if has_krw:
        krw_block = _render_analysis_block(fund["krw"], f"chart-{idx}-krw")
        krw_div = f'<div id="fund-{idx}-krw" style="display:none">{krw_block}</div>'

    return f"""\
<section class="fund-section hidden" id="fund-{idx}">
  <h2>{fund['name']}</h2>
  <p class="fund-meta">{fund['member_cd']} / {fund['fund_cd']} | {b['first_date']} ~ {b['last_date']}</p>
  {toggle_html}
  {usd_div}
  {krw_div}
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
        }
        if f.get("krw"):
            entry["krw"] = {
                "chart": f["krw"]["chart"],
                "daily": f["krw"]["daily"],
                "monthly": f["krw"]["monthly"],
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
    usd_fund_cds: set[str] = set()
    if bench_path.exists():
        with open(bench_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("fundCd"):
                    benchmarks.append({"memberCd": "BENCH", "fundCd": row["fundCd"],
                                       "name": row.get("name") or row["fundCd"]})
                    if row.get("currency", "").upper() == "USD":
                        usd_fund_cds.add(row["fundCd"])

    conn = get_conn(args.db)
    risk_free = args.risk_free / 100.0

    # Load USD/KRW for KRW adjustment of USD assets
    usdkrw: pd.Series | None = None
    if usd_fund_cds:
        try:
            usdkrw = load_nav_series(conn, "BENCH", "USDKRW")
        except Exception:
            logger.warning("USD/KRW data not found; KRW adjustment disabled")

    all_funds = funds + benchmarks
    results = []
    for f in all_funds:
        label = f.get("name") or f["fundCd"]
        print(f"Analyzing [{label}] ...")

        # Build KRW-adjusted NAV for USD benchmarks
        krw_nav = None
        if f["fundCd"] in usd_fund_cds and usdkrw is not None:
            usd_nav = load_nav_series(conn, f["memberCd"], f["fundCd"])
            fx = usdkrw.reindex(usd_nav.index, method="ffill")
            krw_nav = (usd_nav * fx).dropna()

        result = analyze_fund(
            conn, f["memberCd"], f["fundCd"], label,
            risk_free, args.top_drawdowns, krw_nav=krw_nav,
        )
        if result:
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
