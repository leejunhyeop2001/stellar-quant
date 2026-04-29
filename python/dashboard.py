"""Stellar-Quant — Interactive Streamlit Dashboard (Cosmic Edition)."""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_utils import (
    GbmParams,
    YahooFinanceFetchError,
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
# Palette — Toss-inspired premium dark
# ---------------------------------------------------------------------------
BG = "#0F1014"            # Page background
CARD = "#16171C"          # Card surface
CARD_HI = "#1C1D23"       # Slightly elevated card
BORDER = "rgba(255,255,255,0.04)"
TEXT = "#F4F5F7"
MUTED = "#8B93A1"
SUBTLE = "#5C6573"
ACCENT = "#3182F6"        # Toss blue
CYAN = "#38bdf8"
TEAL = "#22c55e"
PINK = "#a78bfa"
ORANGE = "#FF9500"
RED = "#FF4D4F"
GREEN = "#1ABC72"

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
  letter-spacing: -0.012em;
}}
[data-testid="stHeader"] {{
  background: transparent !important;
  height: 0 !important;
  min-height: 0 !important;
}}
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
  display: none !important;
}}

/* ──────────────────────────────────────────────────────────────────
 * Material 아이콘 ligature 누출 방지
 * Streamlit 컴포넌트가 폰트 미적용 시 keyboard_double_arrow_*, arrow_*
 * 등의 텍스트가 그대로 노출됨 → 모두 숨김.
 * ────────────────────────────────────────────────────────────────── */
[data-testid="stIconMaterial"],
[data-testid="stMaterialIcon"],
[data-testid="stIcon"],
[data-testid="stExpanderToggleIcon"],
button[data-testid="baseButton-header"] [data-testid="stMarkdownContainer"]:first-child span,
[data-testid="collapsedControl"] span:not(:empty),
section[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] span:not(:empty),
[data-testid="stHeader"] span[aria-hidden="true"],
[data-testid="stSidebar"] [data-baseweb="slider"] [aria-hidden="true"],
.material-icons,
.material-symbols-outlined {{
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  height: 0 !important;
  font-size: 0 !important;
  line-height: 0 !important;
  opacity: 0 !important;
}}

[data-testid="stSidebar"] {{
  background: {BG} !important;
  border-right: 1px solid {BORDER} !important;
  padding: 4px 4px 16px 4px !important;
}}
[data-testid="stSidebar"] * {{ font-family: var(--sans) !important; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
  border: none !important;
  background: none !important;
}}
section[data-testid="stSidebar"] hr {{ display: none !important; }}

section[data-testid="stSidebar"] .stTextInput > div > div {{
  background: {CARD} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 14px !important;
  padding: 4px 6px !important;
}}
section[data-testid="stSidebar"] .stTextInput input {{
  font-size: 0.9375rem !important;
  padding: 10px 12px !important;
}}
section[data-testid="stSidebar"] .stTextInput > div > div:focus-within {{
  border-color: rgba(49,130,246,0.55) !important;
  box-shadow: 0 0 0 3px rgba(49,130,246,0.12) !important;
}}

section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"],
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
  display: none !important;
}}

[data-testid="stExpander"] {{
  background: {CARD} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 16px !important;
  margin-top: 14px !important;
  overflow: hidden !important;
}}
[data-testid="stExpander"] details {{ background: transparent !important; }}
[data-testid="stExpander"] summary {{
  list-style: none !important;
  list-style-type: none !important;
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 0 !important;
  min-height: 2.5rem !important;
  padding: 12px 16px !important;
  font-size: 0.9375rem !important;
  font-weight: 600 !important;
  color: {TEXT} !important;
  cursor: pointer !important;
}}
[data-testid="stExpander"] summary::-webkit-details-marker,
[data-testid="stExpander"] summary::marker {{
  display: none !important;
  content: "" !important;
}}
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] {{
  flex: 1 1 auto !important;
  display: inline-flex !important;
  align-items: center !important;
}}
/* 우측 +/− 토글 인디케이터 (chevron 텍스트가 숨겨져도 시각 단서 유지) */
[data-testid="stExpander"] summary::after {{
  content: "+";
  margin-left: auto;
  font-weight: 400;
  color: {MUTED};
  font-size: 1.35rem;
  line-height: 1;
  padding-left: 8px;
}}
[data-testid="stExpander"] details[open] summary::after {{
  content: "−";
}}

.sb-label {{
  font-size: 0.75rem;
  font-weight: 600;
  color: {MUTED};
  letter-spacing: 0;
  padding: 18px 4px 10px 4px;
  margin: 0;
  text-transform: uppercase;
}}

.brand-bar {{
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px 10px;
  padding: 8px 4px 20px 4px;
  margin-bottom: 4px;
}}
.brand-title {{
  font-size: 1.125rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  color: {TEXT};
}}
.brand-sep {{ color: {SUBTLE}; font-weight: 400; }}
.brand-author {{
  font-size: 0.875rem;
  font-weight: 500;
  color: {MUTED};
}}

section[data-testid="stSidebar"] button[kind="primary"] {{
  background: {ACCENT} !important;
  border: none !important;
  color: #fff !important;
  font-weight: 700 !important;
  border-radius: 14px !important;
  padding: 14px !important;
  font-size: 0.9375rem !important;
  letter-spacing: -0.01em !important;
  height: 52px !important;
  transition: transform 0.08s ease, background 0.15s ease !important;
}}
section[data-testid="stSidebar"] button[kind="primary"]:hover {{
  background: #1d6ce0 !important;
  transform: translateY(-1px);
}}
section[data-testid="stSidebar"] button[kind="primary"]:active {{
  transform: translateY(0);
}}

.sb-foot {{
  margin-top: 18px;
  padding: 14px 4px 0 4px;
  font-size: 0.6875rem;
  color: {SUBTLE};
  line-height: 1.6;
  border-top: 1px solid {BORDER};
}}

/* 메트릭 카드 — 토스 스타일 */
.mc {{
  background: {CARD};
  border: none;
  border-radius: 24px;
  padding: 28px 28px 24px 28px;
  min-height: 132px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  transition: transform 0.15s ease, background 0.15s ease;
}}
.mc:hover {{
  background: {CARD_HI};
  transform: translateY(-2px);
}}
.mc-lbl {{
  font-size: 0.8125rem;
  font-weight: 500;
  color: {MUTED};
  letter-spacing: -0.005em;
  margin-bottom: 12px;
}}
.mc-val {{
  font-size: 1.875rem;
  font-weight: 800;
  line-height: 1.15;
  color: {TEXT};
  letter-spacing: -0.035em;
  font-variant-numeric: tabular-nums;
}}
.mc-val-lg {{
  font-size: 2.375rem;
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.04em;
  font-variant-numeric: tabular-nums;
}}
.mc-delta {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.8125rem;
  font-weight: 700;
  margin-top: 12px;
  padding: 5px 11px;
  border-radius: 999px;
  letter-spacing: -0.005em;
  font-variant-numeric: tabular-nums;
  width: fit-content;
}}
.d-pos {{ color: {GREEN}; background: rgba(26,188,114,0.12); }}
.d-neg {{ color: {RED};   background: rgba(255,77,79,0.12); }}

/* 섹션 타이틀 */
.stitle {{
  font-size: 1.0625rem;
  font-weight: 700;
  color: {TEXT};
  letter-spacing: -0.03em;
  margin: 40px 0 18px 2px;
  padding-bottom: 0;
  border-bottom: none;
}}
.stitle-sub {{
  font-size: 0.8125rem;
  font-weight: 500;
  color: {MUTED};
  margin-left: 8px;
  letter-spacing: -0.005em;
}}

/* 테이블 — 보더 없음, 행 간격 넓게 */
.rtbl {{
  width: 100%;
  border-collapse: collapse;
  background: {CARD};
  border-radius: 24px;
  overflow: hidden;
  font-size: 0.9375rem;
}}
.rtbl thead tr {{
  background: transparent;
}}
.rtbl th {{
  background: transparent;
  color: {MUTED};
  font-weight: 500;
  text-align: left;
  padding: 18px 24px 12px 24px;
  font-size: 0.75rem;
  letter-spacing: 0;
  text-transform: uppercase;
  border: none;
}}
.rtbl td {{
  background: transparent;
  color: {TEXT};
  padding: 16px 24px;
  border: none;
  font-size: 0.9375rem;
  font-variant-numeric: tabular-nums;
}}
.rtbl tbody tr {{
  border-top: 1px solid {BORDER};
  transition: background 0.12s ease;
}}
.rtbl tbody tr:hover {{ background: rgba(255,255,255,0.025); }}

/* 수학 패널 — 토스 카드 스타일 + 큰 수식 */
.math-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}}
@media (max-width: 900px) {{
  .math-grid {{ grid-template-columns: 1fr; }}
}}
.math-panel {{
  background: {CARD};
  border: none;
  border-radius: 24px;
  padding: 32px 32px 28px 32px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  transition: background 0.15s ease;
}}
.math-panel:hover {{ background: {CARD_HI}; }}
.math-panel h3 {{
  font-size: 0.9375rem;
  font-weight: 700;
  color: {TEXT};
  margin: 0;
  letter-spacing: -0.025em;
}}
.math-panel h3 .math-h-sub {{
  display: block;
  font-size: 0.75rem;
  font-weight: 500;
  color: {MUTED};
  margin-top: 4px;
  letter-spacing: 0;
}}
.math-eq {{
  font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', Georgia, serif;
  font-size: 1.5rem;
  font-weight: 500;
  color: {TEXT};
  background: transparent;
  border: none;
  padding: 8px 4px;
  margin: 0;
  text-align: center;
  line-height: 1.6;
  letter-spacing: 0;
}}
.math-eq small {{ font-size: 1.125rem; color: {MUTED}; }}
.math-desc {{
  font-size: 0.875rem;
  color: {MUTED};
  line-height: 1.7;
  letter-spacing: -0.005em;
  margin: 0;
}}
.math-desc b {{ color: {TEXT}; font-weight: 600; }}
.math-var {{
  display: inline-block;
  font-family: 'Cambria Math', Georgia, serif;
  color: {ACCENT};
  font-weight: 600;
  min-width: 28px;
  font-size: 1.0625rem;
}}

/* Hero 헤더 */
.hbar {{
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px 16px;
  margin: 8px 2px 4px 2px;
  padding: 0;
  border-bottom: none;
}}
.hbar-brand {{
  display: flex;
  align-items: baseline;
  gap: 10px;
  flex-wrap: wrap;
}}
.hbar h1 {{
  font-size: 1.625rem;
  font-weight: 800;
  margin: 0;
  color: {TEXT};
  letter-spacing: -0.04em;
}}
.hbar .hbar-author {{
  font-size: 0.9375rem;
  font-weight: 500;
  color: {MUTED};
}}
.hbar .badge {{
  background: rgba(49,130,246,0.12);
  color: #6BA8FF;
  font-weight: 700;
  font-size: 0.75rem;
  padding: 6px 12px;
  border-radius: 999px;
  letter-spacing: 0.02em;
  border: none;
}}
.hbar .perf {{
  color: {MUTED};
  font-size: 0.75rem;
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.005em;
}}

.disc {{
  font-size: 0.75rem;
  color: {MUTED};
  margin: 32px 0 8px 0;
  padding: 18px 22px;
  background: {CARD};
  border-radius: 16px;
  border: none;
  line-height: 1.7;
  letter-spacing: -0.005em;
}}

/* Streamlit 기본 크롬 정리 */
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden !important; height: 0 !important; }}
.block-container {{
  padding: 1.25rem 1.75rem 2rem 1.75rem !important;
  max-width: 1400px !important;
}}

/* Plotly 차트 컨테이너 — 카드 형태로 감싸되 차트 자체는 투명 */
[data-testid="stPlotlyChart"] {{
  background: {CARD};
  border: none;
  border-radius: 24px;
  padding: 12px;
  overflow: hidden;
}}
[data-testid="stPlotlyChart"] .main-svg {{
  background: transparent !important;
}}

/* 입력 컨트롤(슬라이더, 셀렉트 슬라이더) 톤 */
[data-baseweb="slider"] [role="slider"] {{
  background: {ACCENT} !important;
  border-color: {ACCENT} !important;
  box-shadow: 0 2px 8px rgba(49,130,246,0.35) !important;
}}
section[data-testid="stSidebar"] [data-testid="stSliderTickBar"] {{
  background: rgba(255,255,255,0.04) !important;
}}

/* 토스트 톤 */
[data-testid="stToast"] {{
  background: {CARD_HI} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 16px !important;
  color: {TEXT} !important;
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
    a = "↑" if pct >= 0 else "↓"
    return f'<span class="mc-delta {c}">{a} {pct:+.2f}%</span>'

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
# Chart common — Toss-style transparent canvas, no grid
# ---------------------------------------------------------------------------
_AX = dict(
    showgrid=False,
    zeroline=False,
    showline=False,
    tickfont=dict(
        size=11,
        family="Pretendard Variable, Pretendard, sans-serif",
        color="#8B93A1",
    ),
    title_font=dict(
        size=12,
        color="#8B93A1",
        family="Pretendard Variable, Pretendard, sans-serif",
    ),
    mirror=False,
)

_LAY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="Pretendard Variable, Pretendard, Malgun Gothic, Apple SD Gothic Neo, sans-serif",
        color="#F4F5F7",
        size=13,
    ),
    hoverlabel=dict(
        bgcolor="rgba(28,29,35,0.96)",
        bordercolor="rgba(255,255,255,0.06)",
        font_size=12,
        font_family="Pretendard Variable, Pretendard, sans-serif",
        font_color="#F4F5F7",
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
            text=f"<b>Terminal Price Distribution</b> <span style='color:#8B93A1;font-weight:500'>· 최종가 분포 — {ticker}</span>",
            font=dict(size=15, color="#F4F5F7", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.04,
            y=0.97,
            xanchor="left",
        ),
        margin=dict(l=48, r=20, t=70, b=48),
        legend=dict(
            bgcolor="rgba(28,29,35,0.7)",
            bordercolor="rgba(255,255,255,0.04)",
            borderwidth=1,
            font=dict(size=10, color="#8B93A1", family="Pretendard Variable, Pretendard, sans-serif"),
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
            text=f"<b>Stock Price Simulation</b> <span style='color:#8B93A1;font-weight:500'>· 주가 경로 — {ticker}</span>",
            font=dict(size=15, color="#F4F5F7", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.04,
            y=0.97,
            xanchor="left",
        ),
        margin=dict(l=52, r=72, t=70, b=48),
        legend=dict(
            bgcolor="rgba(28,29,35,0.7)",
            bordercolor="rgba(255,255,255,0.04)",
            borderwidth=1,
            font=dict(size=10, color="#8B93A1", family="Pretendard Variable, Pretendard, sans-serif"),
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
<div class="math-grid">

<div class="math-panel">
  <h3>GBM 확률 미분 방정식<span class="math-h-sub">Stochastic Differential Equation</span></h3>
  <div class="math-eq"><i>dS</i> = μ <i>S</i> <i>dt</i> &nbsp;+&nbsp; σ <i>S</i> <i>dW</i></div>
  <p class="math-desc">
    주가 <b>S</b>의 순간 변화를 모델링합니다. 좌측은 <b>추세 (drift)</b>,
    우측은 <b>무작위 변동 (diffusion)</b>이며 <b>dW</b>는 위너 과정입니다.
  </p>
</div>

<div class="math-panel">
  <h3>해석해 — Itô's Lemma<span class="math-h-sub">이토 보조정리</span></h3>
  <div class="math-eq"><i>S</i>(T) = <i>S</i><small>₀</small> · e<sup>(μ − ½σ²)T + σ√T · Z</sup></div>
  <p class="math-desc">
    위 SDE의 정확한 해입니다. <b>Z ~ N(0,1)</b> 표준정규 난수를 대입해
    미래 시점 <b>T</b>의 주가를 한 번에 계산합니다.
  </p>
</div>

<div class="math-panel">
  <h3>이산화 경로<span class="math-h-sub">Exact Discretization</span></h3>
  <div class="math-eq"><i>S</i>(t+Δt) = <i>S</i>(t) · e<sup>(μ − ½σ²)Δt + σ√Δt Z<small>t</small> + Σ J</sup></div>
  <p class="math-desc">
    Fan Chart의 각 스텝에 적용됩니다. 점프 확산이 꺼져 있으면 Σ J = 0 이며,
    확산 난수에는 <b>대칭 변수법 (antithetic Z / −Z)</b>이 적용됩니다.
  </p>
</div>

<div class="math-panel">
  <h3>VaR — Value at Risk<span class="math-h-sub">최대 예상 손실 (95%)</span></h3>
  <div class="math-eq">VaR<small>95</small> = <i>S</i><small>₀</small> − <i>Q</i><small>0.05</small>(<i>S<small>T</small></i>)</div>
  <p class="math-desc">
    시뮬레이션 하위 <b>5%</b> 가격을 기준으로, 현재가 대비
    <b>95% 확률로 이상의 가격을 유지</b>한다는 의미입니다.
  </p>
</div>

<div class="math-panel">
  <h3>CVaR — Expected Shortfall<span class="math-h-sub">조건부 꼬리 손실</span></h3>
  <div class="math-eq">CVaR<small>95</small> = 𝔼[ Loss <small>|</small> Loss ≥ VaR<small>95</small> ]</div>
  <p class="math-desc">
    Loss = max(0, <i>S</i><small>₀</small> − <i>S<small>T</small></i>) 의 하위 5% 평균.
    VaR보다 <b>꼬리 리스크</b>를 더 보수적으로 요약합니다.
  </p>
</div>

<div class="math-panel">
  <h3>Merton Jump Diffusion<span class="math-h-sub">점프 확산 항</span></h3>
  <div class="math-eq" style="font-size:1.25rem;">
    ln <i>S<small>T</small></i> − ln <i>S</i><small>₀</small> = (μ − ½σ²)T + σ√T Z + Σ J<small>i</small>
  </div>
  <p class="math-desc">
    <b>N<small>T</small> ~ Poisson(λT)</b>, <b>J<small>i</small> ~ N(μ<small>J</small>, σ<small>J</small>²)</b>.
    엔진은 스레드별 <code>std::mt19937</code>·포아송·정규분포를 사용합니다.
    <br/>현재 설정 — λ = <b>{jump_lambda:.4f}</b>, μ<small>J</small> = <b>{jump_mu:.4f}</b>,
    σ<small>J</small> = <b>{jump_sigma:.4f}</b> ({'점프 활성' if jump_on else 'λ = 0 → GBM만'}).
  </p>
</div>

</div>

<div class="math-panel" style="margin-top:16px;">
  <h3>입력 파라미터<span class="math-h-sub">Input Parameters Used</span></h3>
  <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:18px 28px;">
    <div class="math-desc"><span class="math-var">S₀</span><br><b>현재 주가</b><br>{fp(params.s0)}</div>
    <div class="math-desc"><span class="math-var">μ</span><br><b>기대수익률 (연율)</b><br>{params.mu:.6f}</div>
    <div class="math-desc"><span class="math-var">σ</span><br><b>변동성 (연율)</b><br>{params.sigma:.6f}</div>
    <div class="math-desc"><span class="math-var">T</span><br><b>예측 기간</b><br>Time Horizon (years)</div>
    <div class="math-desc"><span class="math-var">λ</span><br><b>연간 점프 강도</b><br>{jump_lambda:.6f}</div>
    <div class="math-desc"><span class="math-var">μ<small>J</small></span><br><b>로그 점프 평균</b><br>{jump_mu:.6f}</div>
    <div class="math-desc"><span class="math-var">σ<small>J</small></span><br><b>로그 점프 변동성</b><br>{jump_sigma:.6f}</div>
    <div class="math-desc"><span class="math-var">Z</span><br><b>표준정규 난수</b><br>Antithetic pairs</div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Metric card
# ---------------------------------------------------------------------------
def _mc(label, value, delta="", lg=False, large=False, vc=TEXT):
    c = "mc-val-lg" if (lg or large) else "mc-val"
    return (
        f'<div class="mc">'
        f'<div class="mc-lbl">{label}</div>'
        f'<div class="{c}" style="color:{vc}">{value}</div>'
        f'{delta}'
        f'</div>'
    )


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
            "Run Simulation",
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
            f'justify-content:center;height:62vh;text-align:center;">'
            f'<h1 style="color:{TEXT};font-weight:800;margin:0;font-size:2.75rem;letter-spacing:-0.045em;">'
            f'Stellar-Quant</h1>'
            f'<p style="color:{MUTED};margin:14px 0 0 0;font-size:1.0625rem;font-weight:500;letter-spacing:-0.02em;">'
            f'C++ × Monte Carlo 주가 시뮬레이션</p>'
            f'<p style="color:{SUBTLE};margin-top:28px;font-size:0.9375rem;max-width:480px;line-height:1.7;letter-spacing:-0.01em;">'
            f'좌측 사이드바에서 종목과 시뮬레이션 설정을 입력한 뒤<br>'
            f'<b style="color:{ACCENT};">Run Simulation</b> 버튼을 눌러 분석을 시작하세요.</p>'
            f'</div>',
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
            except YahooFinanceFetchError as err:
                st.error(
                    "**시세 데이터를 가져오지 못했습니다.**\n\n"
                    f"{err}"
                )
                return
            except ValueError as err:
                st.error(
                    "**가격 데이터를 처리할 수 없습니다.**\n\n"
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
        # ✦ 등은 ALL_EMOJIS 미등록 시 StreamlitAPIException — 검증 통과 이모지 또는 Material 단축코드만 사용
        st.toast("시뮬레이션 완료!", icon="✅")

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
        '<div class="stitle">핵심 지표<span class="stitle-sub">Key metrics</span></div>',
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
    st.markdown('<div class="stitle">투자 전망<span class="stitle-sub">Outlook</span></div>',
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
        c = GREEN if up >= 50 else RED
        st.markdown(_mc("상승 확률 Profit Prob.", f"{up:.1f}%",
                    large=True, vc=c), unsafe_allow_html=True)

    # ── Risk Table ────────────────────────────────────────
    st.markdown('<div class="stitle">리스크 시나리오<span class="stitle-sub">Risk scenarios</span></div>',
                unsafe_allow_html=True)
    scn = [
        ("Best (최선)", "95th", fp(metrics["p95"]), _dpct(metrics["p95"], s0), GREEN),
        ("Expected (기대)", "50th", fp(metrics["p50"]), _dpct(metrics["p50"], s0),
         GREEN if metrics["p50"] >= s0 else RED),
        ("Worst (최악)", "5th", fp(metrics["p05"]), _dpct(metrics["p05"], s0), RED),
    ]
    rows = ""
    for lb, pc, pr, dl, cl in scn:
        a = "↑" if dl >= 0 else "↓"
        rows += (f'<tr><td style="font-weight:600">{lb}</td>'
                 f'<td style="color:{MUTED}">{pc}</td>'
                 f'<td style="font-weight:700">{pr}</td>'
                 f'<td style="color:{cl};font-weight:700">{a} {dl:+.2f}%</td></tr>')

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
            f'<tr><td>μ Annual Drift</td><td>{params.mu:.6f}</td></tr>'
            f'<tr><td>σ Volatility</td><td>{params.sigma:.6f}</td></tr>'
            f'<tr><td>Jump λ / μ_J / σ_J</td>'
            f'<td>{jl:.4f} / {jm:.4f} / {js:.4f}</td></tr>'
            f'<tr><td>90% Confidence</td>'
            f'<td>{fp(metrics["p05"])} — {fp(metrics["p95"])}</td></tr>'
            f'<tr><td>Paths</td><td>{n_paths:,}</td></tr>'
            f'<tr><td>Fan steps / 스레드</td>'
            f'<td>{n_steps_u} / {n_threads_u if n_threads_u else "auto"}</td></tr>'
            f'<tr><td>C++ Engine</td><td>{elapsed:.3f}s</td></tr>'
            f'</tbody></table>',
            unsafe_allow_html=True)

    # ── Math Formula Section ──────────────────────────────
    st.markdown('<div class="stitle">수학 모델<span class="stitle-sub">Model reference</span></div>',
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
