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
from string import Template

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
# Analyze one fund
# ---------------------------------------------------------------------------

def analyze_fund(conn, member_cd: str, fund_cd: str, name: str,
                 risk_free: float, top_n: int) -> dict | None:
    nav = load_nav_series(conn, member_cd, fund_cd)
    if len(nav) < 30:
        logger.warning("Skipping %s: only %d data points", fund_cd, len(nav))
        return None

    basic = compute_basic_metrics(nav, risk_free)
    events = find_drawdown_events(nav)
    dd_summary = drawdown_summary(events)
    top_events = events[:top_n]

    ls_dca = []
    for w in [3, 12, 36]:
        result = compute_ls_vs_dca(nav, w)
        if result:
            ls_dca.append(result)

    # Chart data (downsample to ~500 points)
    step = max(1, len(nav) // 500)
    chart_nav = nav.iloc[::step]
    cummax = nav.cummax()
    dd_series = ((nav - cummax) / cummax).iloc[::step]

    chart_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in chart_nav.index],
        "nav": [round(v, 2) for v in chart_nav.values],
        "drawdown": [round(v * 100, 2) for v in dd_series.values],
    }

    return {
        "name": name,
        "member_cd": member_cd,
        "fund_cd": fund_cd,
        "basic": basic,
        "dd_summary": dd_summary,
        "top_events": top_events,
        "ls_dca": ls_dca,
        "chart": chart_data,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

HTML_TEMPLATE = Template("""\
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
</style>
</head>
<body>
<h1>펀드 분석 대시보드</h1>
<p class="subtitle">생성일: $generated_at | 무위험수익률: $risk_free%</p>

$fund_sections

<script>
const FUNDS = $fund_json;

function createCharts() {
  FUNDS.forEach((fund, idx) => {
    const navCtx = document.getElementById('nav-chart-' + idx);
    const ddCtx = document.getElementById('dd-chart-' + idx);
    if (!navCtx || !ddCtx) return;

    new Chart(navCtx, {
      type: 'line',
      data: {
        labels: fund.chart.dates,
        datasets: [{
          label: '기준가 (원)',
          data: fund.chart.nav,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37,99,235,0.08)',
          fill: true,
          pointRadius: 0,
          borderWidth: 1.5,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { type: 'time', time: { unit: 'year' }, ticks: { maxTicksLimit: 8 } },
          y: { beginAtZero: false }
        },
        plugins: { legend: { display: false } }
      }
    });

    new Chart(ddCtx, {
      type: 'line',
      data: {
        labels: fund.chart.dates,
        datasets: [{
          label: '드로다운 (%)',
          data: fund.chart.drawdown,
          borderColor: '#dc2626',
          backgroundColor: 'rgba(220,38,38,0.15)',
          fill: true,
          pointRadius: 0,
          borderWidth: 1.5,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { type: 'time', time: { unit: 'year' }, ticks: { maxTicksLimit: 8 } },
          y: { max: 0 }
        },
        plugins: { legend: { display: false } }
      }
    });
  });
}

createCharts();
</script>
</body>
</html>
""")


def _fmt_pct(val: float, with_sign: bool = True) -> str:
    sign = "+" if val > 0 and with_sign else ""
    return f"{sign}{val:.2f}%"


def _pct_class(val: float) -> str:
    if val > 0:
        return "positive"
    elif val < 0:
        return "negative"
    return ""


def render_fund_section(fund: dict, idx: int) -> str:
    b = fund["basic"]
    ds = fund["dd_summary"]

    metrics_html = f"""\
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="label">기간</div>
        <div class="value">{b['total_years']}년</div>
      </div>
      <div class="metric-card">
        <div class="label">총 수익률</div>
        <div class="value {_pct_class(b['total_return'])}">{_fmt_pct(b['total_return'])}</div>
      </div>
      <div class="metric-card">
        <div class="label">CAGR</div>
        <div class="value {_pct_class(b['cagr'])}">{_fmt_pct(b['cagr'])}</div>
      </div>
      <div class="metric-card">
        <div class="label">변동성</div>
        <div class="value">{b['volatility']:.2f}%</div>
      </div>
      <div class="metric-card">
        <div class="label">샤프비율</div>
        <div class="value">{b['sharpe']:.2f}</div>
      </div>
      <div class="metric-card">
        <div class="label">MDD</div>
        <div class="value negative">{_fmt_pct(b['mdd'], False)}</div>
      </div>
      <div class="metric-card">
        <div class="label">평균 하락폭</div>
        <div class="value negative">-{ds['avg_drawdown']:.2f}%</div>
      </div>
      <div class="metric-card">
        <div class="label">최장 하락 기간</div>
        <div class="value">{ds['longest_days']}일</div>
      </div>
    </div>"""

    charts_html = f"""\
    <div class="chart-row">
      <div class="chart-container"><canvas id="nav-chart-{idx}"></canvas></div>
      <div class="chart-container"><canvas id="dd-chart-{idx}"></canvas></div>
    </div>"""

    # Drawdown events table
    events = fund["top_events"]
    if events:
        rows = ""
        for i, e in enumerate(events, 1):
            end_str = e["end"] if e["end"] else '<span class="ongoing">진행중</span>'
            rows += f"<tr><td>{i}</td><td>{e['start']}</td><td>{e['trough']}</td>"
            rows += f"<td>{end_str}</td><td class='negative'>{e['depth']:.2f}%</td>"
            rows += f"<td>{e['duration_days']:,}일</td></tr>\n"
        dd_table = f"""\
    <h3>주요 하락 이벤트 (Top {len(events)})</h3>
    <table>
      <tr><th>#</th><th>시작</th><th>저점</th><th>회복</th><th>하락폭</th><th>기간</th></tr>
      {rows}
    </table>"""
    else:
        dd_table = ""

    # LS vs DCA table
    ls_dca = fund["ls_dca"]
    if ls_dca:
        ls_rows = ""
        for r in ls_dca:
            win_cls = "positive" if r["win_rate"] > 50 else "negative"
            mlsa_cls = _pct_class(r["mlsa"])
            ls_rows += f"<tr><td>{r['window']}개월</td>"
            ls_rows += f"<td>{r['observations']:,}</td>"
            ls_rows += f"<td class='{win_cls}'>{r['win_rate']:.1f}%</td>"
            ls_rows += f"<td class='{mlsa_cls}'>{_fmt_pct(r['mlsa'])}</td>"
            ls_rows += f"<td class='negative'>{_fmt_pct(r['mlsd'], False)}</td></tr>\n"
        ls_table = f"""\
    <h3>LS vs DCA 분석</h3>
    <table>
      <tr><th>기간</th><th>관측수</th><th>LS 승률</th><th>MLSA</th><th>MLSD</th></tr>
      {ls_rows}
    </table>"""
    else:
        ls_table = '<p style="color:#888;">데이터 부족으로 LS vs DCA 분석 불가</p>'

    return f"""\
<section class="fund-section" id="fund-{idx}">
  <h2>{fund['name']}</h2>
  <p class="fund-meta">{fund['member_cd']} / {fund['fund_cd']} | {b['first_date']} ~ {b['last_date']}</p>
  {metrics_html}
  {charts_html}
  {dd_table}
  {ls_table}
</section>"""


def render_html(fund_results: list[dict], risk_free: float) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = "\n".join(
        render_fund_section(f, i) for i, f in enumerate(fund_results)
    )
    # Chart data for JS — strip non-chart fields
    chart_payload = [{"chart": f["chart"]} for f in fund_results]
    return HTML_TEMPLATE.substitute(
        generated_at=generated_at,
        risk_free=risk_free,
        fund_sections=sections,
        fund_json=json.dumps(chart_payload, ensure_ascii=False),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="펀드 분석 대시보드 생성")
    ap.add_argument("--fund-list", default="fund_list.csv")
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

    conn = get_conn(args.db)
    risk_free = args.risk_free / 100.0

    results = []
    for f in funds:
        label = f.get("name") or f["fundCd"]
        print(f"Analyzing [{label}] ...")
        result = analyze_fund(
            conn, f["memberCd"], f["fundCd"], label,
            risk_free, args.top_drawdowns,
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
