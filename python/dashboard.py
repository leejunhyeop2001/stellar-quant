"""Stellar-Quant — Interactive Streamlit Dashboard (Cosmic Edition)."""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_utils import (
    GbmParams,
    currency_symbol,
    estimate_gbm_params,
    fetch_prices,
    fmt_price,
)
from loader import import_simulator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stellar-Quant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette — deep space
# ---------------------------------------------------------------------------
BG       = "#0a0e17"
BG2      = "#0e1322"
CARD     = "rgba(14,19,34,0.82)"
BORDER   = "rgba(120,140,180,0.12)"
TEXT     = "#c8d6e5"
MUTED    = "#7889a8"
ACCENT   = "#6b8cc7"
CYAN     = "#7eb8da"
TEAL     = "#6bc9a0"
PINK     = "#e8856c"
ORANGE   = "#e8a44c"
RED      = "#e06060"
GREEN    = "#5ec49a"
MEDIAN_C = "#e8856c"
Q_CLR    = "#6bc9a0"

# ---------------------------------------------------------------------------
# CSS — cosmic / nebula
# ---------------------------------------------------------------------------
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
  --sans: 'Inter','Malgun Gothic',sans-serif;
  --mono: 'JetBrains Mono','Consolas',monospace;
}}

/* ── Cosmic background ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
  background: radial-gradient(ellipse at 20% 0%, rgba(80,120,200,0.05) 0%, transparent 50%),
              radial-gradient(ellipse at 80% 100%, rgba(80,160,200,0.03) 0%, transparent 50%),
              {BG} !important;
  color: {TEXT} !important;
  font-family: var(--sans) !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0c1120 0%, #0e1528 50%, #101828 100%) !important;
  border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; font-family: var(--sans) !important; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
  border: none !important; background: none !important;
}}
section[data-testid="stSidebar"] hr {{ display: none !important; }}
section[data-testid="stSidebar"] .stTextInput > div > div {{
  background: rgba(80,120,200,0.06) !important;
  border: 1px solid rgba(80,120,200,0.12) !important;
  border-radius: 10px !important;
  transition: all 0.25s ease !important;
}}
section[data-testid="stSidebar"] .stTextInput > div > div:focus-within {{
  background: rgba(80,120,200,0.10) !important;
  border-color: rgba(80,120,200,0.3) !important;
  box-shadow: 0 0 16px rgba(80,120,200,0.08) !important;
}}
section[data-testid="stSidebar"] label {{
  color: {MUTED} !important; font-size: 0.7rem !important;
  font-weight: 600 !important; letter-spacing: 0.06em !important;
}}
.sb-label {{
  font-size: 0.62rem; font-weight: 700; color: rgba(107,140,199,0.6);
  text-transform: uppercase; letter-spacing: 0.12em;
  padding: 18px 0 6px 0; margin: 0;
}}
section[data-testid="stSidebar"] button[kind="primary"] {{
  background: linear-gradient(135deg, {ACCENT}, #4a6db0) !important;
  border: none !important; color: #fff !important;
  font-weight: 700 !important; border-radius: 10px !important;
  padding: 11px !important; font-size: 0.88rem !important;
  box-shadow: 0 0 16px rgba(80,120,200,0.12) !important;
  transition: box-shadow 0.3s ease !important;
}}
section[data-testid="stSidebar"] button[kind="primary"]:hover {{
  box-shadow: 0 0 24px rgba(80,120,200,0.25) !important;
}}
.sb-foot {{
  margin-top: 12px; font-size: 0.6rem; color: rgba(107,127,163,0.35);
  line-height: 1.6;
}}

/* ── Cards ─────────────────────────────────────────────── */
.mc {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 14px; padding: 18px 20px;
  backdrop-filter: blur(6px);
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}}
.mc:hover {{
  border-color: rgba(100,140,200,0.2);
  box-shadow: 0 0 20px rgba(100,140,200,0.05);
}}
.mc-lbl {{
  font-size: 0.68rem; font-weight: 600; color: {MUTED};
  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px;
}}
.mc-val {{
  font-size: 1.5rem; font-weight: 800; line-height: 1.2;
  font-family: var(--mono); color: {TEXT};
}}
.mc-val-lg {{
  font-size: 1.85rem; font-weight: 800; line-height: 1.2;
  font-family: var(--mono);
}}
.mc-delta {{ font-size: 0.8rem; font-weight: 600; margin-top: 3px; font-family: var(--mono); }}
.d-pos {{ color: {GREEN}; }}
.d-neg {{ color: {RED}; }}

/* ── Section title ─────────────────────────────────────── */
.stitle {{
  font-size: 0.75rem; font-weight: 700; color: {MUTED};
  text-transform: uppercase; letter-spacing: 0.1em;
  margin: 16px 0 10px 0; padding-bottom: 7px;
  border-bottom: 1px solid {BORDER};
}}

/* ── Tables ────────────────────────────────────────────── */
.rtbl {{
  width:100%; border-collapse:separate; border-spacing:0;
  border-radius:12px; overflow:hidden;
  border:1px solid {BORDER}; font-size:0.82rem;
}}
.rtbl th {{
  background: rgba(6,8,15,0.8); color: {MUTED}; font-weight:600;
  text-align:left; padding:9px 14px;
  font-size:0.68rem; letter-spacing:0.06em; text-transform:uppercase;
}}
.rtbl td {{
  background: {CARD}; color: {TEXT}; padding:9px 14px;
  border-top:1px solid {BORDER}; font-family:var(--mono); font-size:0.8rem;
}}
.rtbl tr {{ transition: background 0.2s ease; }}
.rtbl tr:hover td {{ background: rgba(80,120,200,0.05); }}

/* ── Math formula panel ────────────────────────────────── */
.math-panel {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 14px; padding: 24px 28px;
  backdrop-filter: blur(6px);
  margin-top: 8px;
}}
.math-panel h3 {{
  font-size: 0.88rem; font-weight: 800; color: {ACCENT};
  margin: 0 0 14px 0; letter-spacing: 0.02em;
}}
.math-eq {{
  font-family: var(--mono);
  font-size: 1.0rem; color: {CYAN};
  background: rgba(100,160,200,0.04);
  border-left: 3px solid rgba(100,160,200,0.2);
  padding: 10px 16px; border-radius: 0 8px 8px 0;
  margin: 10px 0;
  letter-spacing: 0.02em;
  line-height: 1.7;
}}
.math-desc {{
  font-size: 0.78rem; color: {MUTED}; line-height: 1.7;
  padding: 2px 0 0 4px;
}}
.math-desc b {{ color: {TEXT}; font-weight: 700; }}
.math-var {{
  display: inline-block;
  font-family: var(--mono); color: {CYAN};
  font-weight: 600; min-width: 26px;
}}

/* ── Header bar ────────────────────────────────────────── */
.hbar {{
  display:flex; align-items:center; gap:14px;
  margin-bottom:8px; padding-bottom:8px;
  border-bottom:1px solid {BORDER};
}}
.hbar h1 {{ font-size:1.3rem; font-weight:800; margin:0; color:{TEXT}; }}
.hbar .badge {{
  background: linear-gradient(135deg, {ACCENT}, #4a6db0);
  color:#fff; font-weight:700; font-size:0.76rem;
  padding:3px 10px; border-radius:6px; font-family:var(--mono);
}}
.hbar .perf {{
  color:{MUTED}; font-size:0.7rem; margin-left:auto; font-family:var(--mono);
}}

.disc {{
  font-size:0.68rem; color:{MUTED}; margin-top:10px;
  padding:8px 12px; background:rgba(6,8,15,0.5);
  border-radius:8px; border-left:3px solid {BORDER};
}}

#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{
  padding-top:1rem !important; padding-bottom:0.5rem !important;
  max-width:100% !important;
}}
[data-testid="stPlotlyChart"] {{
  background: {CARD};
  border:1px solid {BORDER};
  border-radius:14px; padding:4px;
  backdrop-filter: blur(6px);
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


def compute_risk_metrics(samples: np.ndarray, s0: float) -> dict[str, float]:
    qs = np.quantile(samples, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    p01, p05, p25, p50, p75, p95, p99 = (float(q) for q in qs)
    up = float(np.mean(samples > s0) * 100.0)
    va = float(max(0.0, s0 - p05))
    vp = (va / s0) * 100.0 if s0 else 0.0
    return {"p01": p01, "p05": p05, "p25": p25, "p50": p50,
            "p75": p75, "p95": p95, "p99": p99,
            "up_probability_pct": up, "var_95_abs": va, "var_95_pct": vp}


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
# Chart common
# ---------------------------------------------------------------------------
_AX = dict(gridcolor="rgba(160,175,200,0.08)", zerolinecolor="rgba(160,175,200,0.08)",
           gridwidth=1, tickfont=dict(size=10.5, family="JetBrains Mono"),
           title_font=dict(size=11, color=MUTED))

_LAY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,22,0.55)",
    font=dict(family="Inter, Malgun Gothic, sans-serif", color=TEXT, size=12),
    hoverlabel=dict(bgcolor="rgba(12,17,30,0.92)", bordercolor="rgba(140,155,180,0.25)",
                    font_size=12, font_family="Inter, Malgun Gothic"),
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
        marker=dict(color="rgba(100,160,220,0.40)", line_width=0),
        opacity=0.75, name="분포 (Distribution)",
        hovertemplate=f"{sym}%{{x:{hf}}}<br>밀도: %{{y:.5f}}<extra></extra>",
    ))
    lp = np.log(terminal[terminal > 0])
    mu, sig = float(lp.mean()), float(lp.std(ddof=1))
    xp = np.linspace(p01, p99, 300)
    yp = lognormal_pdf(xp, mu, sig)
    pm = float(yp.max()) if len(yp) else 1.0

    fig.add_trace(go.Scatter(x=xp, y=yp, mode="lines",
        line=dict(color="#e8856c", width=2.5, shape="spline", smoothing=1.3),
        name="PDF (확률밀도)", hoverinfo="skip"))

    for v, c, d, w, n in [
        (s0, "rgba(255,255,255,0.7)", "dash", 1.8, "현재가 (Current)"),
        (m["p05"], "#e8a44c", "solid", 2.0, "VaR 5% (최대손실)"),
        (m["p50"], "#e8856c", "solid", 1.8, "Median (중앙값)"),
        (m["p95"], "#6bc9a0", "dash", 1.3, "95th (상한선)"),
    ]:
        fig.add_trace(go.Scatter(x=[v,v], y=[0, pm*1.02], mode="lines",
            line=dict(color=c, width=w, dash=d),
            name=f"{n} {_fp(v,cur)}", hoverinfo="skip"))

    fig.add_annotation(x=m["p05"], y=pm*0.88,
        text=f"<b>VaR 5%</b><br>{_fp(m['p05'],cur)}",
        showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#e8a44c",
        ax=50, ay=-30, font=dict(color="#e8a44c", size=10.5),
        bgcolor="rgba(6,8,15,0.85)", bordercolor="#e8a44c", borderwidth=1, borderpad=4)

    fig.update_layout(**_LAY,
        title=dict(text=f"<b>Terminal Price Distribution (최종가 분포)</b> — {ticker}",
                   font=dict(size=12.5, color=TEXT), x=0.5, y=0.97),
        margin=dict(l=55, r=15, t=70, b=50),
        legend=dict(bgcolor="rgba(6,8,15,0.7)", bordercolor=BORDER, borderwidth=1,
                    font=dict(size=9, color=TEXT), x=0.98, y=0.98, xanchor="right", yanchor="top"),
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
        line=dict(color="rgba(140,160,190,0.05)", width=0.5),
        showlegend=False, hoverinfo="skip"))

    tr = t[::-1]
    fig.add_trace(go.Scatter(x=np.concatenate([t,tr]),
        y=np.concatenate([d95, d05[::-1]]),
        fill="toself", fillcolor="rgba(100,160,220,0.07)",
        line_width=0, name="5%-95% (신뢰구간)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.concatenate([t,tr]),
        y=np.concatenate([d75, d25[::-1]]),
        fill="toself", fillcolor="rgba(100,160,220,0.16)",
        line_width=0, name="25%-75% (신뢰구간)", hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=t, y=d50, mode="lines",
        line=dict(color="#e8856c", width=2.8), name="Median (중간값)",
        hovertemplate=f"%{{x:.2f}}Y — {sym}%{{y:{hf}}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=d05, mode="lines",
        line=dict(color="#6bc9a0", width=1.2, dash="dot"), name="5% (하한)",
        hovertemplate=f"%{{x:.2f}}Y — {sym}%{{y:{hf}}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=d95, mode="lines",
        line=dict(color="#6bc9a0", width=1.2, dash="dot"), name="95% (상한)",
        hovertemplate=f"%{{x:.2f}}Y — {sym}%{{y:{hf}}}<extra></extra>"))

    fig.add_hline(y=s0, line_color="rgba(255,255,255,0.45)", line_dash="dash", line_width=1.5)
    fig.add_annotation(x=0.01, y=s0, xref="paper",
        text=f" 현재가 {_fp(s0,cur)} ", showarrow=False, yanchor="bottom",
        font=dict(color="#fff", size=10), bgcolor="rgba(6,8,15,0.7)", borderpad=3)

    tp, tl = _mticks(yrs)
    yl = min(float(q05.min()), s0) * 0.88
    yh = float(q95.max()) * 1.12

    fig.update_layout(**_LAY,
        title=dict(text=f"<b>Stock Price Simulation (주가 경로 시뮬레이션)</b> — {ticker}",
                   font=dict(size=12.5, color=TEXT), x=0.5, y=0.97),
        margin=dict(l=60, r=70, t=70, b=50),
        legend=dict(bgcolor="rgba(6,8,15,0.7)", bordercolor=BORDER, borderwidth=1,
                    font=dict(size=9, color=TEXT), x=0.02, y=0.98, xanchor="left", yanchor="top"),
        xaxis=dict(**_AX, title="예측 기간 Period", tickvals=tp, ticktext=tl),
        yaxis=dict(**_AX, title=f"주가 Price ({cur})", range=[yl, yh],
                   tickformat=hf, tickprefix=sym),
        height=CH)

    for v, c in [(float(q50[-1]), PINK), (float(q05[-1]), TEAL), (float(q95[-1]), TEAL)]:
        fig.add_annotation(x=yrs, y=v, text=f" <b>{_fp(v,cur)}</b> ",
            showarrow=False, xanchor="left", font=dict(color=c, size=10.5),
            bgcolor="rgba(6,8,15,0.55)", borderpad=2)
    return fig


# ---------------------------------------------------------------------------
# Math formula section builder
# ---------------------------------------------------------------------------
def _build_math_section(params: GbmParams, cur: str) -> str:
    fp = lambda v: _fp(v, cur)
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
  <div class="math-eq">S(t+Δt) = S(t) · exp( (μ − ½σ²)Δt + σ√Δt · Zₜ )</div>
  <div class="math-desc">
    Fan Chart의 각 시간 단계마다 적용되는 수식입니다.<br>
    GBM 해석해를 직접 이산화하므로 <b>근사 오차가 없습니다</b>.
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
  <h3>✦ 입력 파라미터 (Input Parameters Used)</h3>
  <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px 24px;">
    <div class="math-desc"><span class="math-var">S₀</span> <b>현재 주가</b><br>Current Price = {fp(params.s0)}</div>
    <div class="math-desc"><span class="math-var">μ</span> <b>기대수익률 (연율)</b><br>Annual Drift = {params.mu:.6f}</div>
    <div class="math-desc"><span class="math-var">σ</span> <b>변동성 (연율)</b><br>Annual Volatility = {params.sigma:.6f}</div>
    <div class="math-desc"><span class="math-var">T</span> <b>예측 기간</b><br>Time Horizon (years)</div>
    <div class="math-desc"><span class="math-var">Z</span> <b>표준정규 난수</b><br>Z ~ N(0,1), 스레드별 독립 생성</div>
    <div class="math-desc"><span class="math-var">dt</span> <b>시간 간격</b><br>Δt = T / 거래일 수</div>
    <div class="math-desc"><span class="math-var">W</span> <b>위너 과정</b><br>Wiener Process (브라운 운동)</div>
    <div class="math-desc"><span class="math-var">Q</span> <b>분위수 함수</b><br>Quantile (백분위수)</div>
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
            f'<div style="padding:4px 0 14px 0;">'
            f'<span style="font-size:1.15rem;font-weight:800;color:{TEXT};">'
            f'✦ Stellar-Quant</span></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Ticker (종목코드)</div>',
                    unsafe_allow_html=True)
        ticker = st.text_input(
            "Ticker", value="TSLA", max_chars=20,
            placeholder="AAPL, TSLA, 005930.KS …",
            label_visibility="collapsed",
        ).strip().upper()

        st.markdown('<div class="sb-label">Simulation (시뮬레이션 설정)</div>',
                    unsafe_allow_html=True)
        n_paths = st.select_slider(
            "Paths (경로 수)",
            options=[100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
            value=1_000_000, format_func=lambda x: f"{x:,}",
        )
        years = st.slider("Horizon (예측 기간)", 0.25, 3.0, 1.0, 0.25,
                           format="%.2f yr")
        fan_paths = st.slider("Fan Chart Paths (경로 수)", 1000, 20000, 5000, 1000)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        run = st.button("Run Simulation  →",
                        use_container_width=True, type="primary")

        st.markdown(
            '<div class="sb-foot">'
            'C++17 Multi-threaded GBM Engine<br>'
            'pybind11 · zero-copy · WebGL</div>',
            unsafe_allow_html=True)

    # ── Landing ───────────────────────────────────────────
    if not run and "metrics" not in st.session_state:
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'justify-content:center;height:58vh;text-align:center;">'
            f'<div style="font-size:3.5rem;margin-bottom:8px;">✦</div>'
            f'<h2 style="color:{TEXT};font-weight:800;margin:0;font-size:1.6rem;">'
            f'Stellar-Quant</h2>'
            f'<p style="color:{MUTED};margin-top:8px;font-size:0.95rem;max-width:420px;">'
            f'Monte Carlo GBM 시뮬레이터<br>'
            f'좌측에서 종목코드를 입력하고 '
            f'<b style="color:{ACCENT};">Run Simulation</b>을 클릭하세요.</p></div>',
            unsafe_allow_html=True)
        return

    # ── Simulate ──────────────────────────────────────────
    if run:
        with st.spinner("데이터 수집 중 …"):
            close = fetch_prices(ticker, period="2y")
            params = estimate_gbm_params(close, ticker=ticker)
        with st.spinner(f"C++ 시뮬레이션 ({n_paths:,} paths) …"):
            sim = import_simulator()
            t0 = time.perf_counter()
            terminal = np.asarray(sim.simulate_gbm_paths(
                n_paths=n_paths, s0=params.s0, mu=params.mu,
                sigma=params.sigma, t=years, seed=42, n_threads=0), dtype=np.float64)
            path_matrix = np.asarray(sim.simulate_gbm_path_matrix(
                n_paths=fan_paths, n_steps=252, s0=params.s0, mu=params.mu,
                sigma=params.sigma, t=years, seed=43, n_threads=0), dtype=np.float64)
            elapsed = time.perf_counter() - t0
        metrics = compute_risk_metrics(terminal, params.s0)
        st.session_state.update(dict(
            ticker=ticker, params=params, terminal=terminal,
            path_matrix=path_matrix, metrics=metrics,
            elapsed=elapsed, n_paths=n_paths, years=years))

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
    cur, s0  = params.currency, params.s0
    fp = lambda v: _fp(v, cur)
    thru = n_paths / elapsed if elapsed > 0 else 0

    # ── Header ────────────────────────────────────────────
    otxt, oclr = _outlook(metrics["up_probability_pct"])
    st.markdown(
        f'<div class="hbar"><h1>✦ Stellar-Quant</h1>'
        f'<span class="badge">{ticker}</span>'
        f'<span style="color:{oclr};font-weight:700;font-size:0.8rem;">{otxt}</span>'
        f'<span class="perf">⚡ {elapsed:.3f}s · {thru:,.0f} paths/s · '
        f'{n_paths:,} simulations</span></div>',
        unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────
    c1, c2 = st.columns(2, gap="medium")
    cfg = {"displayModeBar": False}
    with c1:
        st.plotly_chart(build_hist(terminal, s0, metrics, cur, ticker),
                        use_container_width=True, config=cfg)
    with c2:
        st.plotly_chart(build_fan(path_matrix, s0, years, cur, ticker),
                        use_container_width=True, config=cfg)

    # ── Metric Cards ──────────────────────────────────────
    st.markdown('<div class="stitle">📊 Investment Outlook (투자 전망)</div>',
                unsafe_allow_html=True)
    up = metrics["up_probability_pct"]
    p50d = _dpct(metrics["p50"], s0)
    md = _dpct(float(terminal.mean()), s0)
    cols = st.columns(5, gap="small")
    with cols[0]:
        st.markdown(_mc("현재가 Current Price", fp(s0)), unsafe_allow_html=True)
    with cols[1]:
        c = GREEN if p50d >= 0 else RED
        st.markdown(_mc("중앙 예측가 Median", fp(metrics["p50"]),
                    _dhtml(p50d), True, c), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(_mc("평균 예측가 Mean", fp(float(terminal.mean())),
                    _dhtml(md)), unsafe_allow_html=True)
    with cols[3]:
        ico = "📈 🟢" if up >= 50 else "📉 🔴"
        c = GREEN if up >= 50 else RED
        st.markdown(_mc("상승 확률 Profit Prob.", f"{ico} {up:.1f}%",
                    large=True, vc=c), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(_mc("VaR(95%) 최대 손실", fp(metrics["var_95_abs"]),
                    f'<div class="mc-delta d-neg">▼ {metrics["var_95_pct"]:.1f}%</div>'),
                    unsafe_allow_html=True)

    # ── Risk Table ────────────────────────────────────────
    st.markdown('<div class="stitle">⚠️ Risk Scenarios (리스크 시나리오)</div>',
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
            f'<tr><td style="font-family:var(--sans)">90% Confidence (신뢰구간)</td>'
            f'<td>{fp(metrics["p05"])} — {fp(metrics["p95"])}</td></tr>'
            f'<tr><td style="font-family:var(--sans)">Paths (시뮬레이션 횟수)</td>'
            f'<td>{n_paths:,}</td></tr>'
            f'<tr><td style="font-family:var(--sans)">C++ Engine (엔진 시간)</td>'
            f'<td>{elapsed:.3f}s</td></tr>'
            f'</tbody></table>',
            unsafe_allow_html=True)

    # ── Math Formula Section ──────────────────────────────
    st.markdown('<div class="stitle">✦ Mathematical Model (수학 모델)</div>',
                unsafe_allow_html=True)
    st.markdown(_build_math_section(params, cur), unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────
    st.markdown(
        '<div class="disc">'
        '※ 과거 2년 Yahoo Finance 데이터 기반 GBM 몬테카를로 시뮬레이션 결과입니다. '
        '실제 투자 판단의 유일한 근거로 사용할 수 없습니다. '
        'Past performance does not guarantee future results.</div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
