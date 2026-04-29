"""Stellar-Quant — Interactive Streamlit Dashboard (Cosmic Edition)."""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_utils import (
    GbmParams,
    compute_risk_metrics,
    currency_symbol,
    estimate_gbm_params,
    estimate_jump_params,
    fetch_prices,
    fmt_price,
)
from loader import import_simulator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stellar-Quant · 이준협",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette — enterprise dark (corporate / 대기업 스타일 톤)
# ---------------------------------------------------------------------------
BG = "#0b0d10"
CARD = "rgba(255,255,255,0.045)"
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#e8eaee"
MUTED = "#8f96a3"
ACCENT = "#3b82f6"
CYAN = "#38bdf8"
TEAL = "#22c55e"
PINK = "#a78bfa"
ORANGE = "#f59e0b"
RED = "#ef4444"
GREEN = "#10b981"

# ---------------------------------------------------------------------------
# Typography — Pretendard + mono for figures
# ---------------------------------------------------------------------------
GLOBAL_CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');

:root {{
  --sans: 'Pretendard Variable', Pretendard, -apple-system, BlinkMacSystemFont,
          'Segoe UI', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
  --mono: 'SF Mono', 'Consolas', 'JetBrains Mono', ui-monospace, monospace;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
  background: {BG} !important;
  color: {TEXT} !important;
  font-family: var(--sans) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

[data-testid="stSidebar"] {{
  background: #0e1117 !important;
  border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ font-family: var(--sans) !important; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
  border: none !important;
  background: none !important;
}}
section[data-testid="stSidebar"] hr {{ display: none !important; }}

section[data-testid="stSidebar"] .stTextInput > div > div {{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid {BORDER} !important;
  border-radius: 6px !important;
}}
section[data-testid="stSidebar"] .stTextInput > div > div:focus-within {{
  border-color: rgba(59,130,246,0.45) !important;
  box-shadow: 0 0 0 1px rgba(59,130,246,0.25) !important;
}}

/* Collapsed labels: Streamlit 기본 라벨 숨김 (key_double·중복 라벨 방지) */
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
  display: none !important;
}}

/* Expander: <summary> 기본 삼각(▶) + Streamlit chevron(→/↓) 겹침 방지 */
[data-testid="stExpander"] summary {{
  list-style: none !important;
  list-style-type: none !important;
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 0.35rem !important;
  min-height: 2.25rem !important;
  padding-inline: 0.15rem 0.35rem !important;
}}
[data-testid="stExpander"] summary::-webkit-details-marker,
[data-testid="stExpander"] summary::marker {{
  display: none !important;
  content: "" !important;
  width: 0 !important;
  height: 0 !important;
}}
[data-testid="stExpander"] summary::-moz-list-bullet {{
  list-style: none !important;
}}
/* 아이콘 한 줄 정렬 */
[data-testid="stExpander"] summary > div,
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] {{
  display: inline-flex !important;
  align-items: center !important;
}}
[data-testid="stExpander"] summary svg {{
  flex-shrink: 0 !important;
  display: block !important;
}}

.sb-label {{
  font-size: 0.75rem;
  font-weight: 600;
  color: {MUTED};
  letter-spacing: -0.01em;
  padding: 16px 0 8px 0;
  margin: 0;
}}

.brand-bar {{
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 6px 10px;
  padding-bottom: 16px;
  margin-bottom: 8px;
  border-bottom: 1px solid {BORDER};
}}
.brand-title {{
  font-size: 1.125rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: {TEXT};
}}
.brand-sep {{
  color: {MUTED};
  font-weight: 400;
  font-size: 1rem;
}}
.brand-author {{
  font-size: 0.9375rem;
  font-weight: 500;
  color: {MUTED};
  letter-spacing: -0.02em;
}}

section[data-testid="stSidebar"] button[kind="primary"] {{
  background: {ACCENT} !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
  border-radius: 6px !important;
  padding: 10px !important;
  font-size: 0.875rem !important;
  letter-spacing: -0.01em !important;
}}
section[data-testid="stSidebar"] button[kind="primary"]:hover {{
  background: #2563eb !important;
}}

.sb-foot {{
  margin-top: 14px;
  font-size: 0.6875rem;
  color: #6b7280;
  line-height: 1.55;
  letter-spacing: -0.01em;
}}

.mc {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 8px;
  padding: 16px 18px;
}}
.mc-lbl {{
  font-size: 0.75rem;
  font-weight: 500;
  color: {MUTED};
  letter-spacing: -0.01em;
  margin-bottom: 6px;
}}
.mc-val {{
  font-size: 1.375rem;
  font-weight: 600;
  line-height: 1.25;
  font-family: var(--mono);
  color: {TEXT};
  letter-spacing: -0.02em;
}}
.mc-val-lg {{
  font-size: 1.625rem;
  font-weight: 600;
  line-height: 1.2;
  font-family: var(--mono);
  letter-spacing: -0.02em;
}}
.mc-delta {{
  font-size: 0.8125rem;
  font-weight: 500;
  margin-top: 4px;
  font-family: var(--mono);
  letter-spacing: -0.01em;
}}
.d-pos {{ color: {GREEN}; }}
.d-neg {{ color: {RED}; }}

.stitle {{
  font-size: 0.8125rem;
  font-weight: 600;
  color: {MUTED};
  letter-spacing: -0.01em;
  margin: 20px 0 12px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid {BORDER};
}}

.rtbl {{
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid {BORDER};
  font-size: 0.8125rem;
}}
.rtbl th {{
  background: rgba(255,255,255,0.04);
  color: {MUTED};
  font-weight: 600;
  text-align: left;
  padding: 10px 14px;
  font-size: 0.75rem;
  letter-spacing: -0.01em;
}}
.rtbl td {{
  background: {CARD};
  color: {TEXT};
  padding: 10px 14px;
  border-top: 1px solid {BORDER};
  font-family: var(--mono);
  font-size: 0.8125rem;
}}
.rtbl tr:hover td {{ background: rgba(255,255,255,0.03); }}

.math-panel {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 8px;
  padding: 22px 24px;
  margin-top: 8px;
}}
.math-panel h3 {{
  font-size: 0.875rem;
  font-weight: 600;
  color: {TEXT};
  margin: 0 0 12px 0;
  letter-spacing: -0.02em;
}}
.math-eq {{
  font-family: var(--mono);
  font-size: 0.875rem;
  color: #93c5fd;
  background: rgba(59,130,246,0.06);
  border-left: 2px solid rgba(59,130,246,0.35);
  padding: 12px 14px;
  border-radius: 0 6px 6px 0;
  margin: 8px 0;
  line-height: 1.65;
}}
.math-desc {{
  font-size: 0.8125rem;
  color: {MUTED};
  line-height: 1.65;
  letter-spacing: -0.01em;
}}
.math-desc b {{ color: {TEXT}; font-weight: 600; }}
.math-var {{
  display: inline-block;
  font-family: var(--mono);
  color: #7dd3fc;
  font-weight: 500;
  min-width: 24px;
}}

.hbar {{
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px 16px;
  margin-bottom: 10px;
  padding-bottom: 12px;
  border-bottom: 1px solid {BORDER};
}}
.hbar-brand {{
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}}
.hbar h1 {{
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0;
  color: {TEXT};
  letter-spacing: -0.03em;
}}
.hbar .hbar-author {{
  font-size: 0.875rem;
  font-weight: 500;
  color: {MUTED};
  letter-spacing: -0.02em;
}}
.hbar .badge {{
  background: rgba(59,130,246,0.2);
  color: #93c5fd;
  font-weight: 600;
  font-size: 0.75rem;
  padding: 4px 10px;
  border-radius: 4px;
  font-family: var(--mono);
  border: 1px solid rgba(59,130,246,0.25);
}}
.hbar .perf {{
  color: {MUTED};
  font-size: 0.75rem;
  margin-left: auto;
  font-family: var(--mono);
  letter-spacing: -0.01em;
}}

.disc {{
  font-size: 0.75rem;
  color: {MUTED};
  margin-top: 12px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
  border-radius: 6px;
  border: 1px solid {BORDER};
  line-height: 1.55;
  letter-spacing: -0.01em;
}}

/* 메인 메뉴·푸터만 숨김. `header` 전체를 숨기면 상단의 사이드바(≡) 버튼도 사라짐 */
#MainMenu, footer {{ visibility: hidden !important; }}
.block-container {{
  padding-top: 1rem !important;
  padding-bottom: 0.5rem !important;
  max-width: 100% !important;
}}
[data-testid="stPlotlyChart"] {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 8px;
  padding: 2px;
}}
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fp(v: float, cur: str) -> str:
    return fmt_price(v, cur)

def _dpct(v: float, s0: float) -> float:
    return (v - s0) / s0 * 100.0 if s0 else 0.0

def _dhtml(pct: float) -> str:
    c = "d-pos" if pct >= 0 else "d-neg"
    a = "▲" if pct >= 0 else "▼"
    return f'<div class="mc-delta {c}">{a} {pct:+.1f}%</div>'

def _outlook(up: float) -> tuple[str, str]:
    if up >= 70: return "Bullish (강세)", GREEN
    if up >= 50: return "Moderately Bullish (약간 강세)", GREEN
    if up >= 30: return "Moderately Bearish (약간 약세)", ORANGE
    return "Bearish (약세)", RED


def lognormal_pdf(x, mu, sig):
    if sig <= 0: return np.zeros_like(x)
    s = np.where(x > 0, x, 1.0)
    z = (np.log(s) - mu) / sig
    p = np.exp(-0.5 * z * z) / (s * sig * np.sqrt(2.0 * np.pi))
    return np.where(x > 0, p, 0.0)


def _mticks(yrs):
    tot = int(round(yrs * 12))
    pos, lbl = [0.0], ["0"]
    m = 3
    while m <= tot:
        pos.append(m / 12.0)
        lbl.append(f"{m // 12}Y" if m % 12 == 0 else f"{m}M")
        m += 3
    if pos[-1] < yrs - 0.01:
        pos.append(yrs); lbl.append(f"{tot}M")
    return pos, lbl


def _ds(arr, mx=120):
    n = len(arr)
    if n <= mx: idx = np.arange(n)
    else: idx = np.unique(np.linspace(0, n - 1, mx, dtype=int))
    return idx, arr[idx]


# ---------------------------------------------------------------------------
# Chart common — Plotly dark theme
# ---------------------------------------------------------------------------
_PLOT_BG = "#0e1218"
_PAPER_BG = "#0a0e14"
_GRID = "rgba(130,150,185,0.12)"
_AXLINE = "rgba(130,150,185,0.18)"

_AX = dict(
    gridcolor=_GRID,
    zerolinecolor=_AXLINE,
    gridwidth=0.85,
    zerolinewidth=0.85,
    tickfont=dict(
        size=11,
        family="SF Mono, Consolas, JetBrains Mono, monospace",
        color="#9ca3af",
    ),
    title_font=dict(
        size=12,
        color="#9ca3af",
        family="Pretendard Variable, Pretendard, sans-serif",
    ),
    showline=True,
    linecolor="rgba(100,120,160,0.25)",
    mirror=False,
)

_LAY = dict(
    paper_bgcolor=_PAPER_BG,
    plot_bgcolor=_PLOT_BG,
    font=dict(
        family="Pretendard Variable, Pretendard, Malgun Gothic, Apple SD Gothic Neo, sans-serif",
        color="#e8eaee",
        size=13,
    ),
    hoverlabel=dict(
        bgcolor="rgba(17,24,39,0.95)",
        bordercolor="rgba(148,163,184,0.35)",
        font_size=12,
        font_family="Pretendard Variable, Pretendard, sans-serif",
        font_color="#f3f4f6",
    ),
    hovermode="x unified",
)

CH = 440


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
def build_hist(terminal, s0, m, cur, ticker):
    sym = currency_symbol(cur)
    hf = ",.0f" if cur == "KRW" else ",.2f"
    p01, p99 = m["p01"], m["p99"]
    cl = terminal[(terminal >= p01) & (terminal <= p99)]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cl, nbinsx=120, histnorm="probability density",
        marker=dict(color="rgba(118,156,238,0.38)", line_width=0),
        opacity=0.82, name="분포 (Distribution)",
        hovertemplate=f"{sym}%{{x:{hf}}}<br>밀도: %{{y:.5f}}<extra></extra>",
    ))
    lp = np.log(terminal[terminal > 0])
    mu, sig = float(lp.mean()), float(lp.std(ddof=1))
    xp = np.linspace(p01, p99, 300)
    yp = lognormal_pdf(xp, mu, sig)
    pm = float(yp.max()) if len(yp) else 1.0

    fig.add_trace(go.Scatter(x=xp, y=yp, mode="lines",
        line=dict(color="#d4a853", width=2.8, shape="spline", smoothing=1.2),
        name="PDF (확률밀도)", hoverinfo="skip"))

    for v, c, d, w, n in [
        (s0, "rgba(230,236,252,0.55)", "dash", 1.6, "현재가 (Current)"),
        (m["p05"], "#e8a85c", "solid", 2.2, "VaR 5% (최대손실)"),
        (m["p50"], "#c9a0e8", "solid", 2.0, "Median (중앙값)"),
        (m["p95"], "#5ec9a8", "dash", 1.4, "95th (상한선)"),
    ]:
        fig.add_trace(go.Scatter(x=[v,v], y=[0, pm*1.02], mode="lines",
            line=dict(color=c, width=w, dash=d),
            name=f"{n} {_fp(v,cur)}", hoverinfo="skip"))

    fig.add_annotation(x=m["p05"], y=pm*0.88,
        text=f"<b>VaR 5%</b><br>{_fp(m['p05'],cur)}",
        showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#e8a85c",
        ax=50, ay=-30, font=dict(color="#fcd34d", size=11, family="Pretendard Variable, sans-serif"),
        bgcolor="rgba(14,18,28,0.92)", bordercolor="#c99a4a", borderwidth=1, borderpad=5)

    fig.update_layout(**_LAY,
        title=dict(
            text=f"<b>Terminal Price Distribution</b> <span style='color:#64748b;font-weight:500'>(최종가 분포)</span> — {ticker}",
            font=dict(size=14, color="#f1f5f9", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.5,
            y=0.97,
        ),
        margin=dict(l=55, r=15, t=70, b=50),
        legend=dict(
            bgcolor="rgba(15,23,42,0.92)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
            font=dict(size=10, color="#cbd5e1", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.98, y=0.98, xanchor="right", yanchor="top",
        ),
        xaxis=dict(**_AX, title=f"최종가 Final Price ({cur})",
                   range=[p01*0.95, p99*1.05], tickformat=hf, tickprefix=sym),
        yaxis=dict(**_AX, title="확률밀도 Density"),
        bargap=0.01, height=CH)
    return fig


# ---------------------------------------------------------------------------
# Fan chart
# ---------------------------------------------------------------------------
def build_fan(pm, s0, yrs, cur, ticker):
    sym = currency_symbol(cur)
    hf = ",.0f" if cur == "KRW" else ",.2f"
    np_, npt = pm.shape
    tf = np.linspace(0.0, yrs, npt)
    q05, q25, q50, q75, q95 = np.quantile(pm, [0.05,0.25,0.5,0.75,0.95], axis=0)

    di, _ = _ds(tf, 120)
    t = tf[di]
    d05, d25, d50, d75, d95 = q05[di], q25[di], q50[di], q75[di], q95[di]

    fig = go.Figure()
    rng = np.random.default_rng(123)
    bc = min(60, np_)
    bi = rng.choice(np_, size=bc, replace=False)
    step = max(1, npt // 80)
    tb = tf[::step]
    xs, ys = [], []
    for i in bi:
        xs.extend(tb.tolist() + [None])
        ys.extend(pm[i, ::step].tolist() + [None])
    fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines",
        line=dict(color="rgba(120,145,190,0.06)", width=0.45),
        showlegend=False, hoverinfo="skip"))

    tr = t[::-1]
    fig.add_trace(go.Scatter(x=np.concatenate([t,tr]),
        y=np.concatenate([d95, d05[::-1]]),
        fill="toself", fillcolor="rgba(118,156,238,0.10)",
        line_width=0, name="5%-95% (신뢰구간)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.concatenate([t,tr]),
        y=np.concatenate([d75, d25[::-1]]),
        fill="toself", fillcolor="rgba(118,156,238,0.18)",
        line_width=0, name="25%-75% (신뢰구간)", hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=t, y=d50, mode="lines",
        line=dict(color="#c9a0e8", width=3.0, shape="spline", smoothing=0.45), name="Median (중간값)",
        hovertemplate=f"%{{x:.2f}}Y — {sym}%{{y:{hf}}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=d05, mode="lines",
        line=dict(color="#5ec9a8", width=1.35, dash="dot"), name="5% (하한)",
        hovertemplate=f"%{{x:.2f}}Y — {sym}%{{y:{hf}}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=d95, mode="lines",
        line=dict(color="#7eb8da", width=1.35, dash="dot"), name="95% (상한)",
        hovertemplate=f"%{{x:.2f}}Y — {sym}%{{y:{hf}}}<extra></extra>"))

    fig.add_hline(y=s0, line_color="rgba(220,228,248,0.42)", line_dash="dash", line_width=1.35)
    fig.add_annotation(x=0.01, y=s0, xref="paper",
        text=f" 현재가 {_fp(s0,cur)} ", showarrow=False, yanchor="bottom",
        font=dict(color="#fff", size=10), bgcolor="rgba(6,8,15,0.7)", borderpad=3)

    tp, tl = _mticks(yrs)
    yl = min(float(q05.min()), s0) * 0.88
    yh = float(q95.max()) * 1.12

    fig.update_layout(**_LAY,
        title=dict(
            text=f"<b>Stock Price Simulation</b> <span style='color:#64748b;font-weight:500'>(주가 경로)</span> — {ticker}",
            font=dict(size=14, color="#f1f5f9", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.5,
            y=0.97,
        ),
        margin=dict(l=60, r=70, t=72, b=50),
        legend=dict(
            bgcolor="rgba(15,23,42,0.92)",
            bordercolor="rgba(148,163,184,0.2)",
            borderwidth=1,
            font=dict(size=10, color="#cbd5e1", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.02, y=0.98, xanchor="left", yanchor="top",
        ),
        xaxis=dict(**_AX, title="예측 기간 Period", tickvals=tp, ticktext=tl),
        yaxis=dict(**_AX, title=f"주가 Price ({cur})", range=[yl, yh],
                   tickformat=hf, tickprefix=sym),
        height=CH)

    for v, c in [(float(q50[-1]), "#c9a0e8"), (float(q05[-1]), "#5ec9a8"), (float(q95[-1]), "#7eb8da")]:
        fig.add_annotation(x=yrs, y=v, text=f" <b>{_fp(v,cur)}</b> ",
            showarrow=False, xanchor="left", font=dict(color=c, size=10.5),
            bgcolor="rgba(6,8,15,0.55)", borderpad=2)
    return fig


# ---------------------------------------------------------------------------
# Math formula section builder
# ---------------------------------------------------------------------------
def _build_math_section(
    params: GbmParams,
    cur: str,
    jump_lambda: float,
    jump_mu: float,
    jump_sigma: float,
) -> str:
    fp = lambda v: _fp(v, cur)
    jump_on = jump_lambda > 0.0
    return f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">

<div class="math-panel">
  <h3>✦ GBM 확률 미분 방정식 (Stochastic Differential Equation)</h3>
  <div class="math-eq">dS = μ · S · dt  +  σ · S · dW</div>
  <div class="math-desc">
    주가 <b>S</b>의 순간적 변화를 나타내는 기본 모델입니다.<br>
    왼쪽 항은 <b>추세(drift)</b>, 오른쪽 항은 <b>무작위 변동(diffusion)</b>입니다.
  </div>
</div>

<div class="math-panel">
  <h3>✦ 해석해 — Itô's Lemma (이토 보조정리)</h3>
  <div class="math-eq">S(T) = S₀ · exp( (μ − ½σ²)T + σ√T · Z )</div>
  <div class="math-desc">
    위 SDE의 정확한 해입니다.<br>
    <b>Z ~ N(0,1)</b> 표준정규분포 난수를 대입하면<br>
    미래 시점 <b>T</b>에서의 주가 <b>S(T)</b>를 한 번에 계산합니다.
  </div>
</div>

<div class="math-panel">
  <h3>✦ 이산화 경로 (Exact Discretization)</h3>
  <div class="math-eq">S(t+Δt) = S(t) · exp( (μ − ½σ²)Δt + σ√Δt · Zₜ + Σ jumps)</div>
  <div class="math-desc">
    Fan Chart의 각 시간 단계마다 적용되며, 점프 확산이 꺼져 있으면 Σ jumps = 0입니다.<br>
    확산 난수에는 <b>대칭 변수법 (antithetic Z / −Z)</b>을 적용했습니다.
  </div>
</div>

<div class="math-panel">
  <h3>✦ VaR — Value at Risk (최대 예상 손실)</h3>
  <div class="math-eq">VaR(95%) = S₀ − Q₀.₀₅(S_T)</div>
  <div class="math-desc">
    시뮬레이션 결과의 <b>하위 5%</b> 가격을 기준으로<br>
    현재가 대비 <b>95% 확률로 이 이상의 가격을 유지</b>한다는 의미입니다.
  </div>
</div>

</div>

<div class="math-panel" style="margin-top:16px;">
  <h3>✦ Merton Jump Diffusion — 점프 항</h3>
  <div class="math-eq">ln S_T − ln S₀ = (μ − ½σ²)T + σ√T · Z + Σ<sub>i=1</sub><sup>N_T</sup> Jᵢ,&nbsp;
  N_T ~ Poisson(λT),&nbsp; Jᵢ ~ N(μ_J, σ_J²)</div>
  <div class="math-desc">
    <b>dN_t</b>: 포아송 도약. <b>J</b>: 로그 점프 크기. 엔진은 스레드별 <code>std::mt19937</code>,
    <code>std::poisson_distribution</code>, 정규분포를 사용합니다.
    <br/>현재 설정: λ = <b>{jump_lambda:.4f}</b>, μ_J = <b>{jump_mu:.4f}</b>, σ_J = <b>{jump_sigma:.4f}</b>
    ({'점프 활성' if jump_on else 'λ = 0 → GBM만'}).
  </div>
</div>

<div class="math-panel" style="margin-top:16px;">
  <h3>✦ CVaR — Expected Shortfall (조건부 꼬리 손실)</h3>
  <div class="math-eq">CVaR₍₉₅₎ = 𝔼[ Loss | Loss ≥ VaR₍₉₅₎ ], &nbsp; Loss = max(0, S₀ − S_T)</div>
  <div class="math-desc">
    하위 5% 경로에서의 평균 손실로, VaR보다 꼬리 리스크를 더 보수적으로 요약합니다.
  </div>
</div>

<div class="math-panel" style="margin-top:16px;">
  <h3>✦ 입력 파라미터 (Input Parameters Used)</h3>
  <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px 24px;">
    <div class="math-desc"><span class="math-var">S₀</span> <b>현재 주가</b><br>Current Price = {fp(params.s0)}</div>
    <div class="math-desc"><span class="math-var">μ</span> <b>기대수익률 (연율)</b><br>Annual Drift = {params.mu:.6f}</div>
    <div class="math-desc"><span class="math-var">σ</span> <b>변동성 (연율)</b><br>Annual Volatility = {params.sigma:.6f}</div>
    <div class="math-desc"><span class="math-var">T</span> <b>예측 기간</b><br>Time Horizon (years)</div>
    <div class="math-desc"><span class="math-var">λ</span> <b>연간 점프 강도</b><br>Jump intensity = {jump_lambda:.6f}</div>
    <div class="math-desc"><span class="math-var">μ_J</span> <b>로그 점프 평균</b><br>Mean log jump = {jump_mu:.6f}</div>
    <div class="math-desc"><span class="math-var">σ_J</span> <b>로그 점프 변동성</b><br>Log-jump std = {jump_sigma:.6f}</div>
    <div class="math-desc"><span class="math-var">Z</span> <b>표준정규 난수</b><br>Antithetic pairs per step</div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Metric card
# ---------------------------------------------------------------------------
def _mc(label, value, delta="", lg=False, large=False, vc=TEXT):
    c = "mc-val-lg" if (lg or large) else "mc-val"
    return (f'<div class="mc"><div class="mc-lbl">{label}</div>'
            f'<div class="{c}" style="color:{vc}">{value}</div>{delta}</div>')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div class="brand-bar">'
            '<span class="brand-title">Stellar-Quant</span>'
            '<span class="brand-sep">·</span>'
            '<span class="brand-author">이준협</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-label">Ticker (종목코드)</div>',
                    unsafe_allow_html=True)
        ticker = st.text_input(
            "Ticker",
            value="TSLA",
            max_chars=20,
            placeholder="AAPL, TSLA, 005930.KS …",
            label_visibility="collapsed",
            key="sq_ticker_symbol",
        ).strip().upper()

        st.markdown('<div class="sb-label">Simulation (시뮬레이션)</div>',
                    unsafe_allow_html=True)
        n_paths = st.select_slider(
            "Monte Carlo paths",
            options=[100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
            value=1_000_000,
            format_func=lambda x: f"{x:,}",
            label_visibility="collapsed",
            key="sq_n_paths",
        )
        years = st.slider(
            "Horizon",
            0.25,
            3.0,
            1.0,
            0.25,
            format="%.2f yr",
            label_visibility="collapsed",
            key="sq_horizon_years",
        )

        with st.expander("고급 설정", expanded=False):
            st.markdown('<div class="sb-label">Fan chart</div>', unsafe_allow_html=True)
            fan_paths = st.slider(
                "Fan paths",
                1000,
                20000,
                5000,
                1000,
                help="Fan chart에 그리는 개별 경로 수입니다.",
                label_visibility="collapsed",
                key="sq_fan_paths",
            )
            n_steps = st.slider(
                "Time steps",
                52,
                504,
                252,
                26,
                help="경로를 나누는 시간 스텝 수 (거래일 스케일).",
                label_visibility="collapsed",
                key="sq_fan_steps",
            )
            n_threads = st.slider(
                "C++ threads",
                0,
                32,
                0,
                help="0이면 사용 가능한 코어 수에 맞춥니다.",
                label_visibility="collapsed",
                key="sq_cpp_threads",
            )
            st.markdown('<div class="sb-label">Jump diffusion (Merton)</div>',
                        unsafe_allow_html=True)
            j_lambda = st.slider(
                "Jump lambda",
                0.0,
                15.0,
                0.0,
                0.01,
                format="%.3f",
                label_visibility="collapsed",
                key="sq_jump_lambda",
            )
            j_mu = st.slider(
                "Jump mu",
                -0.35,
                0.35,
                0.0,
                0.005,
                format="%.4f",
                label_visibility="collapsed",
                key="sq_jump_mu",
            )
            j_sigma = st.slider(
                "Jump sigma",
                0.0,
                1.0,
                0.05,
                0.005,
                format="%.4f",
                label_visibility="collapsed",
                key="sq_jump_sigma",
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        run = st.button(
            "Run Simulation  →",
            use_container_width=True,
            type="primary",
            key="sq_run_simulation",
        )

        st.markdown(
            '<div class="sb-foot">'
            'C++17 Multi-threaded GBM Engine<br>'
            'pybind11 · zero-copy · WebGL</div>',
            unsafe_allow_html=True)

        ej = st.session_state.get("est_jump")
        if ej is not None:
            st.caption(
                f"참고 추정(역사적): λ̂={ej.lambda_annual:.4f}, "
                f"μ̂_J={ej.mu_jump:.4f}, σ̂_J={ej.sigma_jump:.4f}"
            )

    # ── Landing ───────────────────────────────────────────
    if not run and "metrics" not in st.session_state:
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'justify-content:center;height:58vh;text-align:center;">'
            f'<div style="font-size:3rem;margin-bottom:12px;opacity:0.9;">✦</div>'
            f'<h2 style="color:{TEXT};font-weight:700;margin:0;font-size:1.5rem;letter-spacing:-0.03em;">'
            f'Stellar-Quant</h2>'
            f'<p style="color:{MUTED};margin:8px 0 0 0;font-size:0.9375rem;font-weight:500;letter-spacing:-0.02em;">'
            f'이준협</p>'
            f'<p style="color:{MUTED};margin-top:16px;font-size:0.875rem;max-width:420px;line-height:1.55;letter-spacing:-0.01em;">'
            f'Monte Carlo 시뮬레이션<br>'
            f'좌측에서 설정 후 <b style="color:{ACCENT};">Run Simulation</b>을 실행하세요.</p></div>',
            unsafe_allow_html=True)
        return

    # ── Simulate ──────────────────────────────────────────
    if run:
        tick = ticker.strip()
        if not tick:
            st.error("종목 코드를 입력해 주세요.")
            return
        with st.spinner("시세 데이터 수집 중…"):
            try:
                close = fetch_prices(tick, period="2y")
                params = estimate_gbm_params(close, ticker=tick)
                st.session_state.est_jump = estimate_jump_params(close)
            except ValueError as err:
                st.error(
                    "**가격 데이터를 가져오지 못했습니다.**\n\n"
                    f"{err}"
                )
                return
        spinner_engine = (
            "엔진 가동 중: 1,000만 경로 연산 중…"
            if n_paths >= 10_000_000
            else f"엔진 가동 중: {n_paths:,} 경로 연산 중…"
        )
        with st.spinner(spinner_engine):
            sim = import_simulator()
            t0 = time.perf_counter()
            terminal = np.asarray(
                sim.simulate_gbm_paths(
                    n_paths=n_paths,
                    s0=params.s0,
                    mu=params.mu,
                    sigma=params.sigma,
                    t=years,
                    seed=42,
                    n_threads=int(n_threads),
                    jump_lambda=j_lambda,
                    jump_mu=j_mu,
                    jump_sigma=j_sigma,
                ),
                dtype=np.float64,
            )
            path_matrix = np.asarray(
                sim.simulate_gbm_path_matrix(
                    n_paths=fan_paths,
                    n_steps=int(n_steps),
                    s0=params.s0,
                    mu=params.mu,
                    sigma=params.sigma,
                    t=years,
                    seed=43,
                    n_threads=int(n_threads),
                    jump_lambda=j_lambda,
                    jump_mu=j_mu,
                    jump_sigma=j_sigma,
                ),
                dtype=np.float64,
            )
            elapsed = time.perf_counter() - t0
        metrics = compute_risk_metrics(terminal, params.s0)
        st.session_state.update(
            dict(
                ticker=ticker,
                params=params,
                terminal=terminal,
                path_matrix=path_matrix,
                metrics=metrics,
                elapsed=elapsed,
                n_paths=n_paths,
                years=years,
                n_steps=int(n_steps),
                n_threads=int(n_threads),
                jump_lambda=j_lambda,
                jump_mu=j_mu,
                jump_sigma=j_sigma,
            )
        )
        st.toast("시뮬레이션 완료!", icon="✦")

    # ── Read state ────────────────────────────────────────
    ss = st.session_state
    ticker   = ss["ticker"]
    params   = ss["params"]
    terminal = ss["terminal"]
    path_matrix = ss["path_matrix"]
    metrics  = ss["metrics"]
    elapsed  = ss["elapsed"]
    n_paths  = ss["n_paths"]
    years    = ss["years"]
    n_steps_u = int(ss.get("n_steps", 252))
    n_threads_u = int(ss.get("n_threads", 0))
    jl = float(ss.get("jump_lambda", 0.0))
    jm = float(ss.get("jump_mu", 0.0))
    js = float(ss.get("jump_sigma", 0.0))
    cur, s0  = params.currency, params.s0
    fp = lambda v: _fp(v, cur)
    thru = n_paths / elapsed if elapsed > 0 else 0

    # ── Header ────────────────────────────────────────────
    otxt, oclr = _outlook(metrics["up_probability_pct"])
    st.markdown(
        f'<div class="hbar">'
        f'<div class="hbar-brand">'
        f'<h1>Stellar-Quant</h1>'
        f'<span class="hbar-author">이준협</span>'
        f'</div>'
        f'<span class="badge">{ticker}</span>'
        f'<span style="color:{oclr};font-weight:600;font-size:0.8125rem;letter-spacing:-0.01em;">{otxt}</span>'
        f'<span class="perf">{elapsed:.3f}s · {thru:,.0f} paths/s · '
        f'{n_paths:,} simulations</span></div>',
        unsafe_allow_html=True)

    # ── Key metrics (최종 예상가 · VaR · CVaR) ─────────────
    st.markdown(
        '<div class="stitle" style="margin-top:4px;">핵심 지표 · Key metrics</div>',
        unsafe_allow_html=True,
    )
    k1, k2, k3 = st.columns(3, gap="medium")
    p50d = _dpct(metrics["p50"], s0)
    c50 = GREEN if p50d >= 0 else RED
    with k1:
        st.markdown(
            _mc(
                "최종 예상가 Median (Sₜ 중앙)",
                fp(metrics["p50"]),
                _dhtml(p50d),
                lg=True,
                vc=c50,
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            _mc(
                "VaR (95%) 최대 손실",
                fp(metrics["var_95_abs"]),
                f'<div class="mc-delta d-neg">▼ {metrics["var_95_pct"]:.1f}%</div>',
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            _mc(
                "CVaR (95%) ES 꼬리 손실",
                fp(metrics["cvar_95_abs"]),
                f'<div class="mc-delta d-neg">▼ {metrics["cvar_95_pct"]:.1f}%</div>',
            ),
            unsafe_allow_html=True,
        )

    # ── Charts ────────────────────────────────────────────
    c1, c2 = st.columns(2, gap="medium")
    cfg = {"displayModeBar": False}
    with c1:
        st.plotly_chart(build_hist(terminal, s0, metrics, cur, ticker),
                        use_container_width=True, config=cfg)
    with c2:
        st.plotly_chart(build_fan(path_matrix, s0, years, cur, ticker),
                        use_container_width=True, config=cfg)

    # ── Metric Cards — 보조 지표 ──────────────────────────
    st.markdown('<div class="stitle">투자 전망 · Outlook</div>',
                unsafe_allow_html=True)
    up = metrics["up_probability_pct"]
    md = _dpct(float(terminal.mean()), s0)
    cols_a = st.columns(3, gap="small")
    with cols_a[0]:
        st.markdown(_mc("현재가 Current Price", fp(s0)), unsafe_allow_html=True)
    with cols_a[1]:
        st.markdown(_mc("평균 예측가 Mean", fp(float(terminal.mean())),
                    _dhtml(md)), unsafe_allow_html=True)
    with cols_a[2]:
        ico = "📈 🟢" if up >= 50 else "📉 🔴"
        c = GREEN if up >= 50 else RED
        st.markdown(_mc("상승 확률 Profit Prob.", f"{ico} {up:.1f}%",
                    large=True, vc=c), unsafe_allow_html=True)

    # ── Risk Table ────────────────────────────────────────
    st.markdown('<div class="stitle">리스크 시나리오 · Risk scenarios</div>',
                unsafe_allow_html=True)
    scn = [
        ("Best (최선)", "95th", fp(metrics["p95"]), _dpct(metrics["p95"], s0), GREEN),
        ("Expected (기대)", "50th", fp(metrics["p50"]), _dpct(metrics["p50"], s0),
         GREEN if metrics["p50"] >= s0 else RED),
        ("Worst (최악)", "5th", fp(metrics["p05"]), _dpct(metrics["p05"], s0), RED),
    ]
    rows = ""
    for lb, pc, pr, dl, cl in scn:
        a = "▲" if dl >= 0 else "▼"
        rows += (f'<tr><td style="font-weight:600;font-family:var(--sans)">{lb}</td>'
                 f'<td style="color:{MUTED}">{pc}</td>'
                 f'<td style="font-weight:600">{pr}</td>'
                 f'<td style="color:{cl};font-weight:600">{a} {dl:+.1f}%</td></tr>')

    tl, tr = st.columns([1.2, 1], gap="medium")
    with tl:
        st.markdown(
            f'<table class="rtbl"><thead><tr>'
            f'<th>Scenario (시나리오)</th><th>Percentile (백분위)</th>'
            f'<th>Price (가격)</th><th>Change (변동)</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>',
            unsafe_allow_html=True)
    with tr:
        st.markdown(
            f'<table class="rtbl"><thead><tr>'
            f'<th>Parameter (파라미터)</th><th>Value (값)</th></tr></thead><tbody>'
            f'<tr><td style="font-family:var(--sans)">μ Annual Drift (기대수익률)</td>'
            f'<td>{params.mu:.6f}</td></tr>'
            f'<tr><td style="font-family:var(--sans)">σ Volatility (변동성)</td>'
            f'<td>{params.sigma:.6f}</td></tr>'
            f'<tr><td style="font-family:var(--sans)">Jump λ / μ_J / σ_J</td>'
            f'<td>{jl:.4f} / {jm:.4f} / {js:.4f}</td></tr>'
            f'<tr><td style="font-family:var(--sans)">90% Confidence (신뢰구간)</td>'
            f'<td>{fp(metrics["p05"])} — {fp(metrics["p95"])}</td></tr>'
            f'<tr><td style="font-family:var(--sans)">Paths (시뮬레이션 횟수)</td>'
            f'<td>{n_paths:,}</td></tr>'
            f'<tr><td style="font-family:var(--sans)">Fan steps / 스레드</td>'
            f'<td>{n_steps_u} / {n_threads_u if n_threads_u else "auto"}</td></tr>'
            f'<tr><td style="font-family:var(--sans)">C++ Engine (엔진 시간)</td>'
            f'<td>{elapsed:.3f}s</td></tr>'
            f'</tbody></table>',
            unsafe_allow_html=True)

    # ── Math Formula Section ──────────────────────────────
    st.markdown('<div class="stitle">수학 모델 · Model reference</div>',
                unsafe_allow_html=True)
    st.markdown(_build_math_section(params, cur, jl, jm, js), unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────
    st.markdown(
        '<div class="disc">'
        '※ 과거 2년 Yahoo Finance 데이터 기반 GBM 몬테카를로 시뮬레이션 결과입니다. '
        '실제 투자 판단의 유일한 근거로 사용할 수 없습니다. '
        'Past performance does not guarantee future results.</div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
