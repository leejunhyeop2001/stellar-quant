"""Stellar-Quant — Interactive Streamlit Dashboard (Cosmic Edition)."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_utils import (
    GbmParams,
    JumpParams,
    YahooFinanceFetchError,
    GBM_MU_ANNUAL_CAP,
    GBM_SIGMA_ANNUAL_FLOOR,
    MU_MARKET_PRIOR,
    MU_SHRINKAGE_SAMPLE_WEIGHT,
    RISK_FREE_RATE_ANNUAL,
    clamp_gbm_for_simulation,
    compute_kelly_leverage_fraction,
    compute_risk_metrics,
    compute_sortino_from_terminal,
    currency_symbol,
    detect_currency,
    estimate_gbm_params,
    estimate_jump_params,
    fetch_prices,
    fmt_price,
    shrink_mu_toward_market_prior,
)
from loader import import_simulator


@st.cache_resource(show_spinner=False)
def _cached_simulator():
    """C++ 모듈은 한 번만 import — Streamlit rerun 시 재로딩 방지."""
    return import_simulator()

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
BG = "#000000"            # 앱 전체 배경 (깊이)
CARD = "#101012"          # 카드 표면
CARD_HI = "#16161A"       # 호버/약한 상승
TEXT = "#F4F5F7"
MUTED = "#8B93A1"
SUBTLE = "#5C6573"
ACCENT = "#0064FF"        # Toss 브랜드 블루 (슬라이더·강조)
ACCENT_GLOW = "rgba(0,100,255,0.14)"
RED = "#FF4B4B"
GREEN = ACCENT

# ---------------------------------------------------------------------------
# 기본 시뮬레이션 파라미터 (모델 오차가 MC 오차보다 커서 기본 경로 수는 보수적으로 둔다)
# ---------------------------------------------------------------------------
FIXED_N_PATHS = 3_000_000
FIXED_HORIZON_YEARS = 1.0
FIXED_FAN_PATHS = 8_000
FIXED_N_STEPS = 252
FIXED_N_THREADS = 0  # 0 = 사용 가능 코어 자동
FIXED_JUMP_LAMBDA = 1.5
FIXED_JUMP_MU = -0.15
FIXED_JUMP_SIGMA = 0.10
JUMP_MODE_CONSERVATIVE = "conservative"
JUMP_MODE_HISTORICAL = "historical"
JUMP_MODE_OFF = "off"
JUMP_MODE_LABELS = {
    JUMP_MODE_CONSERVATIVE: "보수적 스트레스",
    JUMP_MODE_HISTORICAL: "역사적 추정",
    JUMP_MODE_OFF: "순수 GBM",
}
# yfinance 실패 시 Manual Fallback (GBM 입력)
FIXED_MANUAL_S0 = 250.0
FIXED_MANUAL_MU = 0.10
FIXED_MANUAL_SIGMA = 0.40

TICKER_CUSTOM = "__CUSTOM__"
# (심볼, 표시 라벨) — 사이드바 selectbox
TICKER_PRESETS: list[tuple[str, str]] = [
    ("TSLA", "Tesla · TSLA"),
    ("AAPL", "Apple · AAPL"),
    ("NVDA", "NVIDIA · NVDA"),
    ("MSFT", "Microsoft · MSFT"),
    ("GOOGL", "Alphabet · GOOGL"),
    ("AMZN", "Amazon · AMZN"),
    ("META", "Meta · META"),
    ("AMD", "AMD · AMD"),
    ("INTC", "Intel · INTC"),
    ("005930.KS", "삼성전자 · 005930.KS"),
    ("000660.KS", "SK하이닉스 · 000660.KS"),
    ("373220.KQ", "LG에너지솔루션 · 373220.KQ"),
    (TICKER_CUSTOM, "직접 입력…"),
]
TICKER_LABELS: dict[str, str] = dict(TICKER_PRESETS)
TICKER_OPTIONS: list[str] = [sym for sym, _ in TICKER_PRESETS]


@dataclass(frozen=True, slots=True)
class SidebarConfig:
    """Per-run inputs: ticker + run flag; 나머지는 모듈 상수 FIXED_*."""

    ticker: str
    n_paths: int
    years: float
    manual_s0: float
    manual_mu: float
    manual_sigma: float
    fan_paths: int
    n_steps: int
    n_threads: int
    jump_mode: str
    jump_lambda: float
    jump_mu: float
    jump_sigma: float
    run: bool


@dataclass(frozen=True, slots=True)
class MarketData:
    """Yahoo Finance data transformed into model-ready parameters."""

    params: GbmParams
    jump: JumpParams
    mu_uncertainty: float
    source: str


@dataclass(frozen=True, slots=True)
class EngineOutput:
    """Raw arrays returned by the C++ Monte Carlo engine."""

    terminal: np.ndarray
    path_matrix: np.ndarray
    elapsed: float


@dataclass(frozen=True, slots=True)
class DashboardResult:
    """Complete render-ready simulation result."""

    ticker: str
    params: GbmParams
    terminal: np.ndarray
    path_matrix: np.ndarray
    metrics: dict[str, float]
    elapsed: float
    n_paths: int
    years: float
    n_steps: int
    n_threads: int
    jump_mode: str
    jump_lambda: float
    jump_mu: float
    jump_sigma: float
    mu_uncertainty: float
    fan_fig: go.Figure | None


def _fixed_engine_config(ticker: str, run: bool, jump_mode: str) -> SidebarConfig:
    return SidebarConfig(
        ticker=ticker.strip().upper(),
        n_paths=FIXED_N_PATHS,
        years=FIXED_HORIZON_YEARS,
        manual_s0=FIXED_MANUAL_S0,
        manual_mu=FIXED_MANUAL_MU,
        manual_sigma=FIXED_MANUAL_SIGMA,
        fan_paths=FIXED_FAN_PATHS,
        n_steps=FIXED_N_STEPS,
        n_threads=FIXED_N_THREADS,
        jump_mode=jump_mode,
        jump_lambda=FIXED_JUMP_LAMBDA,
        jump_mu=FIXED_JUMP_MU,
        jump_sigma=FIXED_JUMP_SIGMA,
        run=run,
    )


# ---------------------------------------------------------------------------
# Typography — Pretendard + mono for figures
# ---------------------------------------------------------------------------
GLOBAL_CSS = f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css');

:root {{
  --sans: 'Pretendard Variable', Pretendard, -apple-system, BlinkMacSystemFont,
          'Segoe UI', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
  --sans-ui: 'Pretendard Variable', Pretendard, -apple-system, BlinkMacSystemFont, sans-serif;
  --toss-blue: #0064FF;
  --mono: 'SF Mono', 'Consolas', 'JetBrains Mono', ui-monospace, monospace;
  --card: #101012;
  --card-hi: #16161A;
  --shadow-card: 0 1px 0 rgba(255,255,255,0.04) inset, 0 12px 40px rgba(0,0,0,0.55);
  --shadow-card-hover: 0 1px 0 rgba(255,255,255,0.06) inset, 0 18px 48px rgba(0,0,0,0.65);
}}

*,
*::before,
*::after {{
  box-sizing: border-box !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
  background: {BG} !important;
  background-clip: padding-box !important;
  color: {TEXT} !important;
  font-family: var(--sans) !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  letter-spacing: -0.02em !important;
}}
[data-testid="stHeader"] {{
  background: transparent !important;
  height: 3.25rem !important;
  min-height: 3.25rem !important;
  visibility: visible !important;
  pointer-events: none !important;
  z-index: 999998 !important;
}}
/* 상단 툴바는 숨기지 않는다 — 최신 Streamlit은 사이드바 열기/접기가 stToolbar에 있음 */
[data-testid="stDecoration"] {{
  display: none !important;
}}
[data-testid="stToolbar"] {{
  display: flex !important;
  visibility: visible !important;
  pointer-events: auto !important;
  background: transparent !important;
  gap: 4px !important;
}}
[data-testid="stHeader"] [data-testid="stToolbar"],
[data-testid="stHeader"] [data-testid="stToolbar"] * {{
  pointer-events: auto !important;
}}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] [data-testid="baseButton-secondary"] {{
  color: {TEXT} !important;
  background: rgba(16,16,18,0.92) !important;
  border-radius: 12px !important;
}}

/* ──────────────────────────────────────────────────────────────────
 * Material 아이콘 ligature 누출 방지 (사이드바·본문 위젯만)
 * 헤더/stToolbar는 제외 — 햄버거 등 네비게이션 아이콘이 사라지지 않게 함.
 * ────────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stIconMaterial"],
section[data-testid="stSidebar"] [data-testid="stMaterialIcon"],
section[data-testid="stSidebar"] [data-testid="stExpanderToggleIcon"],
section[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] span:not(:empty),
[data-testid="stSidebar"] [data-baseweb="slider"] [aria-hidden="true"],
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
.block-container [data-baseweb="slider"] [aria-hidden="true"],
section[data-testid="stSidebar"] .material-icons,
section[data-testid="stSidebar"] .material-symbols-outlined,
[data-testid="stExpander"] .material-icons,
[data-testid="stExpander"] .material-symbols-outlined {{
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  height: 0 !important;
  font-size: 0 !important;
  line-height: 0 !important;
  opacity: 0 !important;
}}

/* Sidebar collapsed 상태에서 다시 여는 컨트롤은 숨기지 않는다. */
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] *,
[data-testid="collapsedControl"] span,
[data-testid="collapsedControl"] span:not(:empty),
[data-testid="collapsedControl"] [data-testid="stIcon"] {{
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  width: auto !important;
  height: auto !important;
  font-size: 18px !important;
  line-height: 1 !important;
  color: {ACCENT} !important;
}}
[data-testid="collapsedControl"] {{
  position: fixed !important;
  top: 14px !important;
  left: 14px !important;
  z-index: 999999 !important;
  pointer-events: auto !important;
  border-radius: 999px !important;
  background: #101012 !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.45) !important;
  padding: 6px !important;
}}

[data-testid="stSidebar"] {{
  /* Streamlit 기본에는 스플리터 드래그 없음 → 컴팩트 고정 폭으로 밀도 최적화 */
  width: 288px !important;
  min-width: 288px !important;
  max-width: 288px !important;
  flex: 0 0 288px !important;
  box-sizing: border-box !important;
  background: {BG} !important;
  background-clip: padding-box !important;
  border-right: none !important;
  box-shadow: 4px 0 24px rgba(0,0,0,0.35) !important;
  padding: 10px 12px 18px 12px !important;
  font-family: var(--sans-ui) !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
}}
[data-testid="stSidebar"] * {{ font-family: inherit !important; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
  border: none !important;
  background: none !important;
}}
section[data-testid="stSidebar"] > div [data-testid="stVerticalBlock"] {{
  gap: 14px !important;
}}

/* 종목 입력 — 카드형 과대 패딩 제거, 한 줄 입력에 맞춤 */
section[data-testid="stSidebar"] .stTextInput > div > div {{
  background-color: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 6px 20px rgba(0,0,0,0.4) !important;
  padding: 3px 6px !important;
  margin-bottom: 12px !important;
  min-height: 0 !important;
  overflow: hidden !important;
  transition: box-shadow 0.18s ease, background 0.18s ease !important;
}}
section[data-testid="stSidebar"] .stTextInput > div > div:hover {{
  transform: none !important;
  background-color: #141418 !important;
  box-shadow: 0 0 0 1px rgba(0,100,255,0.15), 0 8px 24px rgba(0,0,0,0.45) !important;
}}
section[data-testid="stSidebar"] .stTextInput input {{
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  padding: 11px 12px !important;
  min-height: 44px !important;
  line-height: 1.35 !important;
  letter-spacing: -0.02em !important;
}}
section[data-testid="stSidebar"] .stTextInput > div > div:focus-within {{
  box-shadow: 0 0 0 3px {ACCENT_GLOW}, var(--shadow-card-hover) !important;
}}

/* 사이드바 라벨 / 보조 — 시각적 노이즈 ↓ */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
  display: flex !important;
  align-items: center !important;
  flex-wrap: wrap !important;
  gap: 6px !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: {TEXT} !important;
  letter-spacing: -0.02em !important;
  text-transform: none !important;
  margin: 0 0 8px 2px !important;
  padding: 0 !important;
  line-height: 1.35 !important;
  opacity: 1 !important;
}}
section[data-testid="stSidebar"] [data-testid="stWidgetHelp"] {{
  margin-left: 2px !important;
  margin-top: 1px !important;
  opacity: 0.5 !important;
}}
section[data-testid="stSidebar"] [data-baseweb="slider"],
section[data-testid="stSidebar"] [data-testid="stSlider"] {{
  margin-top: 8px !important;
  margin-bottom: 4px !important;
}}
section[data-testid="stSidebar"] .stSelectSlider [data-baseweb="slider"] {{
  margin-top: 8px !important;
}}

section[data-testid="stSidebar"] [data-testid="stCaption"] {{
  font-family: var(--sans-ui) !important;
  font-size: 0.65rem !important;
  font-weight: 400 !important;
  color: {SUBTLE} !important;
  opacity: 0.5 !important;
}}

section[data-testid="stSidebar"] .stSelectSlider {{
  padding: 14px 16px 12px 16px !important;
  margin-bottom: 4px !important;
  background-color: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 18px !important;
  box-shadow: var(--shadow-card) !important;
  overflow: hidden !important;
  transition: box-shadow 0.18s ease, background 0.18s ease !important;
}}
section[data-testid="stSidebar"] .stSelectSlider:hover {{
  transform: none !important;
  background-color: #141418 !important;
  box-shadow: var(--shadow-card-hover) !important;
}}
section[data-testid="stSidebar"] .stSlider {{
  padding: 14px 16px 12px 16px !important;
  margin-bottom: 4px !important;
  background-color: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 18px !important;
  box-shadow: var(--shadow-card) !important;
  overflow: hidden !important;
  transition: box-shadow 0.18s ease, background 0.18s ease !important;
}}
section[data-testid="stSidebar"] .stSlider:hover {{
  transform: none !important;
  background-color: #141418 !important;
  box-shadow: var(--shadow-card-hover) !important;
}}
.sb-rec-caption {{
  font-family: var(--sans-ui) !important;
  font-size: 0.625rem !important;
  font-weight: 400 !important;
  color: {SUBTLE} !important;
  line-height: 1.4 !important;
  margin: 3px 0 0 3px !important;
  padding: 0 !important;
  letter-spacing: -0.02em !important;
  opacity: 0.45 !important;
}}

section[data-testid="stSidebar"] [data-testid="stExpander"] button[kind="secondary"] {{
  border-radius: 18px !important;
  font-weight: 600 !important;
  font-family: var(--sans-ui) !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  background: #101012 !important;
  color: {TEXT} !important;
  padding: 12px 16px !important;
  margin-bottom: 8px !important;
  overflow: hidden !important;
  letter-spacing: -0.02em !important;
  font-size: 0.8125rem !important;
  transition: background 0.18s ease, box-shadow 0.18s ease !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] button[kind="secondary"]:hover {{
  transform: none !important;
  background: rgba(0,100,255,0.10) !important;
  box-shadow: 0 10px 28px rgba(0,100,255,0.12) !important;
}}
[data-testid="stExpander"] {{
  background: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  outline: none !important;
  box-shadow: 0 1px 0 rgba(255,255,255,0.035) inset, 0 14px 42px rgba(0,0,0,0.54) !important;
  border-radius: 24px !important;
  margin-top: 10px !important;
  margin-bottom: 34px !important;
  overflow: hidden !important;
  isolation: isolate !important;
  transition: box-shadow 0.22s ease, background 0.22s ease, transform 0.22s ease !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
  background: #101012 !important;
}}
[data-testid="stExpander"]:hover {{
  box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset, 0 18px 48px rgba(0,0,0,0.62), 0 0 0 1px rgba(0,100,255,0.08) !important;
}}
[data-testid="stExpander"] details {{
  background: #101012 !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  overflow: hidden !important;
}}
[data-testid="stExpander"] .streamlit-expanderHeader,
[data-testid="stExpander"] .streamlit-expanderContent {{
  background: #101012 !important;
  border: none !important;
  border-top: none !important;
  border-bottom: none !important;
  outline: none !important;
  box-shadow: none !important;
}}
[data-testid="stExpander"] hr,
[data-testid="stExpander"] [role="separator"],
[data-testid="stExpander"] summary + div::before {{
  display: none !important;
  border: none !important;
  content: none !important;
}}
[data-testid="stExpander"] [data-testid="stVerticalBlock"] {{
  background: #101012 !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  padding: 30px !important;
  padding-top: 18px !important;
  gap: 22px !important;
}}
[data-testid="stExpander"] details[open] > div {{
  animation: sq-expander-open 0.28s cubic-bezier(0.22, 1, 0.36, 1);
}}
[data-testid="stExpander"] summary {{
  list-style: none !important;
  list-style-type: none !important;
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 0 !important;
  min-height: 3.1rem !important;
  padding: 18px 22px !important;
  font-size: 1rem !important;
  font-weight: 850 !important;
  font-family: var(--sans-ui) !important;
  color: {TEXT} !important;
  background: #101012 !important;
  border: none !important;
  border-bottom: none !important;
  outline: none !important;
  box-shadow: none !important;
  letter-spacing: -0.02em !important;
  cursor: pointer !important;
  transition: background 0.22s ease, color 0.22s ease !important;
}}
[data-testid="stExpander"] summary:hover {{
  background: linear-gradient(90deg, rgba(0,100,255,0.10), rgba(16,16,18,1) 42%) !important;
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
  content: "⌄";
  margin-left: auto;
  font-weight: 800;
  color: {ACCENT};
  font-size: 1.1rem;
  line-height: 1;
  padding-left: 8px;
  transform: rotate(-90deg);
  transition: transform 0.22s ease, color 0.22s ease, opacity 0.22s ease !important;
}}
[data-testid="stExpander"] details[open] summary::after {{
  content: "⌄";
  transform: rotate(0deg);
}}

@keyframes sq-expander-open {{
  from {{
    opacity: 0;
    transform: translateY(-8px);
    filter: blur(2px);
  }}
  to {{
    opacity: 1;
    transform: translateY(0);
    filter: blur(0);
  }}
}}

.sb-section-hd {{
  font-family: var(--sans-ui) !important;
  font-size: 0.65rem;
  font-weight: 700;
  color: #0064FF;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 14px 2px 8px 2px;
  margin: 0;
  border-top: none !important;
  margin-top: 6px;
  opacity: 0.85;
}}

.brand-bar {{
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px 10px;
  padding: 6px 2px 14px 2px;
  margin-bottom: 4px;
}}
.brand-title {{
  font-family: var(--sans-ui) !important;
  font-size: 1.125rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: {TEXT};
}}
.brand-sep {{ color: {SUBTLE}; font-weight: 400; opacity: 0.5; }}
.brand-author {{
  font-family: var(--sans-ui) !important;
  font-size: 0.75rem;
  font-weight: 500;
  color: {MUTED};
  opacity: 0.5;
}}

section[data-testid="stSidebar"] button[kind="primary"] {{
  background: #0064FF !important;
  border: none !important;
  color: #fff !important;
  font-weight: 800 !important;
  font-family: var(--sans-ui) !important;
  border-radius: 16px !important;
  padding: 14px !important;
  font-size: 0.875rem !important;
  letter-spacing: -0.02em !important;
  min-height: 48px !important;
  margin-bottom: 14px !important;
  overflow: hidden !important;
  box-shadow: 0 6px 24px rgba(0,100,255,0.3) !important;
  transition: transform 0.15s ease, background 0.15s ease, box-shadow 0.15s ease !important;
}}
section[data-testid="stSidebar"] button[kind="primary"]:hover {{
  background: #0052CC !important;
  transform: scale(1.01);
  box-shadow: 0 8px 28px rgba(0,100,255,0.4) !important;
}}
section[data-testid="stSidebar"] button[kind="primary"]:active {{
  transform: scale(0.99);
}}

.sb-foot {{
  font-family: var(--sans-ui) !important;
  margin-top: 8px;
  padding: 14px 4px 8px 4px;
  font-size: 0.625rem;
  font-weight: 400;
  color: {SUBTLE};
  line-height: 1.55;
  border-top: none !important;
  opacity: 0.5;
}}

/* ── Toss 카드 — 테두리 없음, 그림자 + clip ── */
.toss-card {{
  background: #101012 !important;
  background-clip: padding-box !important;
  border-radius: 22px !important;
  border: none !important;
  box-shadow: var(--shadow-card) !important;
  overflow: hidden !important;
}}

.top-shell {{
  margin: 8px 0 42px 0;
  padding: 34px 36px;
  border-radius: 28px;
  background: #101012 !important;
  box-shadow: var(--shadow-card);
  overflow: hidden !important;
}}
.top-kicker {{
  color: {MUTED};
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  margin: 0 0 10px 0;
}}
.top-title {{
  color: {TEXT};
  font-size: clamp(2.1rem, 4vw, 3.4rem);
  font-weight: 850;
  letter-spacing: -0.055em;
  line-height: 1.02;
  margin: 0 0 12px 0;
}}
.top-sub {{
  color: {MUTED};
  font-size: 1rem;
  font-weight: 500;
  line-height: 1.65;
  margin: 0;
  max-width: 760px;
}}
.risk-hero {{
  margin: 0 0 48px 0;
  padding: 34px 36px;
  border-radius: 30px;
  background: #101012 !important;
  box-shadow: var(--shadow-card);
  overflow: hidden !important;
}}
.risk-hero-head {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  margin-bottom: 28px;
}}
.risk-label {{
  color: {MUTED};
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  margin-bottom: 8px;
}}
.risk-title {{
  color: {TEXT};
  font-size: clamp(1.85rem, 3vw, 2.7rem);
  font-weight: 850;
  letter-spacing: -0.055em;
  line-height: 1.08;
}}
.risk-pill {{
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 9px 14px;
  font-size: 0.78rem;
  font-weight: 750;
  white-space: nowrap;
}}
.risk-pill-blue {{ color: {ACCENT}; background: rgba(0,100,255,0.13); }}
.risk-pill-red {{ color: {RED}; background: rgba(255,75,75,0.13); }}
.risk-grid {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 18px;
}}
.risk-meta {{
  color: {MUTED};
  font-size: 0.8rem;
  font-weight: 500;
  line-height: 1.65;
  letter-spacing: -0.02em;
  margin: 10px 0 20px 0;
  max-width: 920px;
  opacity: 0.92;
}}
.risk-meta strong {{
  color: {TEXT};
  font-weight: 650;
}}
.risk-item {{
  padding: 22px 20px;
  border-radius: 22px;
  background: #16161A !important;
}}
.risk-item-label {{
  color: {MUTED};
  font-size: 0.72rem;
  font-weight: 650;
  margin-bottom: 10px;
}}
.risk-item-value {{
  color: {TEXT};
  font-size: 1.42rem;
  font-weight: 850;
  letter-spacing: -0.035em;
  font-variant-numeric: tabular-nums;
}}
.risk-blue {{ color: {ACCENT} !important; }}
.risk-red {{ color: {RED} !important; }}
.sq-loading {{
  margin: 24px 0 40px 0;
  padding: 22px 26px;
  border-radius: 22px;
  background: #101012 !important;
  box-shadow: var(--shadow-card);
  color: {TEXT};
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}}
.sq-loading span {{
  color: {MUTED};
  display: block;
  font-size: 0.78rem;
  font-weight: 500;
  margin-top: 8px;
}}
button[kind="primary"] {{
  background: {ACCENT} !important;
  border: none !important;
  color: #fff !important;
  font-weight: 800 !important;
  border-radius: 18px !important;
  min-height: 52px !important;
  box-shadow: 0 8px 28px rgba(0,100,255,0.32) !important;
}}
button[kind="primary"]:hover {{
  background: #0052CC !important;
  box-shadow: 0 10px 34px rgba(0,100,255,0.42) !important;
}}
@media (max-width: 900px) {{
  .risk-hero-head {{ flex-direction: column; }}
  .risk-grid {{ grid-template-columns: 1fr 1fr; }}
}}
@media (max-width: 640px) {{
  .risk-grid {{ grid-template-columns: 1fr; }}
  .top-shell, .risk-hero {{ padding: 26px 22px; }}
}}

.mc {{
  background: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 22px !important;
  padding: 26px 28px;
  min-height: 122px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-shadow: var(--shadow-card);
  overflow: hidden !important;
  transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease !important;
}}
.mc:hover {{
  background: {CARD_HI} !important;
  transform: scale(1.01);
  box-shadow: var(--shadow-card-hover) !important;
}}
.mc-lbl {{
  font-size: 0.72rem;
  font-weight: 500;
  color: {MUTED};
  letter-spacing: -0.02em;
  margin-bottom: 10px;
  opacity: 0.5;
}}
.mc-val, .mc-val-lg {{
  font-weight: 800;
  line-height: 1.12;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  color: #0064FF;
}}
.mc-val {{
  font-size: 1.875rem;
}}
.mc-val-lg {{
  font-size: 2.375rem;
}}
.mc-delta {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.72rem;
  font-weight: 600;
  margin-top: 10px;
  padding: 6px 11px;
  border-radius: 999px;
  letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
  width: fit-content;
  opacity: 0.5;
}}
.d-pos {{ color: {ACCENT}; background: rgba(0,100,255,0.14); opacity: 1; }}
.d-neg {{ color: {RED};   background: rgba(255,75,75,0.14); opacity: 1; }}

.stitle {{
  font-size: 1.08rem;
  font-weight: 700;
  color: {TEXT};
  letter-spacing: -0.02em;
  margin: 16px 0 24px 2px !important;
  padding: 0.4rem 0 0 0;
  border: none !important;
}}
.stitle-sub {{
  font-size: 0.75rem;
  font-weight: 500;
  color: {MUTED};
  margin-left: 10px;
  letter-spacing: -0.02em;
  opacity: 0.5;
}}

.rtbl-wrap {{
  background: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 22px;
  overflow: hidden !important;
  box-shadow: var(--shadow-card);
  padding: 22px 24px 20px 24px;
  transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
}}
.rtbl-wrap:hover {{
  transform: scale(1.005);
  box-shadow: var(--shadow-card-hover) !important;
  background-color: #16161A !important;
}}
.rtbl th {{
  background: transparent;
  color: {MUTED};
  font-weight: 500;
  text-align: left;
  padding: 12px 4px 18px 4px;
  font-size: 0.7rem;
  letter-spacing: -0.02em;
  text-transform: uppercase;
  border: none;
  opacity: 0.5;
}}
.rtbl td {{
  background: transparent;
  color: {TEXT};
  padding: 21px 4px;
  border: none;
  font-size: 0.9375rem;
  font-variant-numeric: tabular-nums;
}}
.rtbl tbody tr {{
  border-top: none !important;
  transition: background 0.12s ease;
}}
.rtbl tbody tr:hover {{ background: rgba(255,255,255,0.03); }}

.rtbl {{
  width: 100%;
  border-collapse: collapse;
  background: transparent;
  border-radius: 0;
  overflow: hidden;
  font-size: 0.9375rem;
}}
.rtbl thead tr {{
  background: transparent;
}}

/* 수학 패널 — 토스 카드 스타일 + 큰 수식 */
.math-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
}}
@media (max-width: 900px) {{
  .math-grid {{ grid-template-columns: 1fr; }}
}}
.math-panel h3 {{
  font-size: 0.9375rem;
  font-weight: 700;
  color: {TEXT};
  margin: 0;
  letter-spacing: -0.02em;
}}
.math-panel {{
  background: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 22px;
  padding: 26px 28px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  box-shadow: var(--shadow-card);
  overflow: hidden !important;
  transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease !important;
}}
.math-panel:hover {{
  background: {CARD_HI} !important;
  transform: scale(1.01);
  box-shadow: var(--shadow-card-hover) !important;
}}
.math-panel h3 .math-h-sub {{
  display: block;
  font-size: 0.7rem;
  font-weight: 500;
  color: {MUTED};
  margin-top: 6px;
  letter-spacing: -0.02em;
  opacity: 0.5;
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
.math-eq small {{ font-size: 1.125rem; color: {MUTED}; opacity: 0.5; }}
.math-desc {{
  font-size: 0.8125rem;
  color: {MUTED};
  line-height: 1.7;
  letter-spacing: -0.02em;
  margin: 0;
  opacity: 0.5;
}}
.math-desc b {{ color: {TEXT}; font-weight: 600; opacity: 1; }}
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
  margin: 6px 2px 8px 2px;
  padding: 0;
  border: none !important;
}}
.hbar h1 {{
  font-size: 1.625rem;
  font-weight: 800;
  margin: 0;
  color: {TEXT};
  letter-spacing: -0.02em;
}}
.hbar .hbar-author {{
  font-size: 0.8125rem;
  font-weight: 500;
  color: {MUTED};
  opacity: 0.5;
  letter-spacing: -0.02em;
}}
.hbar .badge {{
  background: rgba(0,100,255,0.15);
  color: #4D94FF;
  font-weight: 700;
  font-size: 0.7rem;
  padding: 10px 16px;
  border-radius: 999px;
  letter-spacing: -0.02em;
  border: none !important;
  box-shadow: 0 0 0 1px rgba(0,100,255,0.12) inset;
}}
.hbar .perf {{
  color: {MUTED};
  font-size: 0.7rem;
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
  opacity: 0.5;
}}

.disc {{
  font-size: 0.72rem;
  color: {MUTED};
  margin: 28px 0 12px 0;
  padding: 22px 24px;
  background: #101012 !important;
  background-clip: padding-box !important;
  border-radius: 22px;
  border: none !important;
  box-shadow: var(--shadow-card);
  overflow: hidden !important;
  line-height: 1.65;
  letter-spacing: -0.02em;
  opacity: 0.5;
}}

#MainMenu, footer {{ visibility: hidden !important; height: 0 !important; }}
header[data-testid="stHeader"] {{
  visibility: visible !important;
  background: transparent !important;
  pointer-events: none !important;
}}
header[data-testid="stHeader"] [data-testid="collapsedControl"],
header[data-testid="stHeader"] button {{
  pointer-events: auto !important;
}}
.block-container {{
  padding: 1.25rem 1.75rem 2.25rem 1.75rem !important;
  max-width: 1400px !important;
}}

.block-container [data-testid="stVerticalBlock"] {{
  gap: 2.25rem !important;
}}
.block-container [data-testid="stHorizontalBlock"] {{
  gap: 1.75rem !important;
}}

[data-testid="stPlotlyChart"] {{
  background: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 22px !important;
  padding: 20px 20px 14px 20px !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-card) !important;
  margin-bottom: 0 !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}}
[data-testid="stPlotlyChart"]:hover {{
  transform: scale(1.005);
  box-shadow: var(--shadow-card-hover) !important;
}}
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plotly,
[data-testid="stPlotlyChart"] .user-select-none {{
  overflow: hidden !important;
  border-radius: 18px !important;
  box-sizing: border-box !important;
}}
[data-testid="stPlotlyChart"] .main-svg {{
  background: transparent !important;
}}

[data-testid="stLatex"] {{
  text-align: center !important;
  background: rgba(0,100,255,0.07) !important;
  background-clip: padding-box !important;
  border-radius: 14px !important;
  border: none !important;
  padding: 16px 14px !important;
  margin: 8px 0 !important;
  overflow: hidden !important;
}}

[data-testid="stLatex"] .katex-display {{
  margin: 0 !important;
  overflow-x: auto !important;
  overflow-y: hidden !important;
}}
[data-testid="stLatex"] .katex {{
  color: #0064FF !important;
  font-size: 1.35rem !important;
  font-weight: 800 !important;
}}

.sq-math-start ~ [data-testid="stHorizontalBlock"]
  [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {{
  background: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 22px !important;
  padding: 26px 22px !important;
  box-shadow: var(--shadow-card) !important;
  overflow: hidden !important;
  transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease !important;
}}
.sq-math-start ~ [data-testid="stHorizontalBlock"]
  [data-testid="stColumn"] > [data-testid="stVerticalBlock"]:hover {{
  background: {CARD_HI} !important;
  transform: scale(1.01);
  box-shadow: var(--shadow-card-hover) !important;
}}
.sq-math-start ~ [data-testid="stHorizontalBlock"]
  [data-testid="stLatex"] {{
  background: rgba(0,100,255,0.08) !important;
  border-radius: 16px !important;
}}

/* 슬라이더 — 얇은 트랙 #2C2C2E, 큰 핸들 */
[data-baseweb="slider"] [role="slider"] {{
  background: #FFFFFF !important;
  border: 3px solid #0064FF !important;
  box-shadow: 0 0 0 4px rgba(0,100,255,0.18), 0 4px 14px rgba(0,0,0,0.35) !important;
  width: 22px !important;
  height: 22px !important;
}}
[data-baseweb="slider"] div[class*="Track"] {{
  background: #2C2C2E !important;
  height: 2px !important;
  border-radius: 999px !important;
}}
[data-baseweb="slider"] [data-testid="stSliderThumb"],
[data-baseweb="slider"] div[class*="Track"]:first-of-type,
[data-baseweb="slider"] div[class*="InnerTrack"],
[data-baseweb="slider"] [class*="InnerTrack"] {{
  background: #0064FF !important;
}}
[data-baseweb="slider"] div[class*="Track"]:not(:first-of-type) {{
  background: #2C2C2E !important;
}}
[data-baseweb="slider"] [data-testid="stSliderThumbValue"] {{
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  color: {TEXT} !important;
  font-family: var(--mono) !important;
  opacity: 0.5 !important;
}}
[data-baseweb="slider"] [data-testid="stTickBar"] div,
section[data-testid="stSidebar"] [data-testid="stSliderTickBar"] {{
  background: rgba(255,255,255,0.04) !important;
}}

[data-testid="stToast"] {{
  background: {CARD_HI} !important;
  border: none !important;
  box-shadow: var(--shadow-card-hover) !important;
  border-radius: 22px !important;
  color: {TEXT} !important;
}}

@keyframes sq-shimmer {{
  0%   {{ background-position: -900px 0; }}
  100% {{ background-position:  900px 0; }}
}}
@keyframes sq-pulse-soft {{
  0%, 100% {{ opacity: 0.55; filter: brightness(0.92); }}
  50%      {{ opacity: 1; filter: brightness(1.06); }}
}}
.sq-skeleton {{
  background: linear-gradient(
    90deg,
    {CARD} 25%,
    {CARD_HI} 50%,
    {CARD} 75%
  );
  background-size: 1800px 100%;
  animation: sq-shimmer 1.5s infinite linear, sq-pulse-soft 2.2s ease-in-out infinite;
  border-radius: 22px;
  box-shadow: var(--shadow-card);
}}
.sq-chart-skeleton {{
  height: 460px;
}}

@keyframes sq-spin {{
  to {{ transform: rotate(360deg); }}
}}
.sq-ring {{
  width: 48px;
  height: 48px;
  border: 3px solid rgba(0,100,255,0.12);
  border-top-color: {ACCENT};
  border-radius: 50%;
  animation: sq-spin 0.85s linear infinite, sq-pulse-soft 2s ease-in-out infinite;
}}

@keyframes sq-status-pulse {{
  0%, 100% {{ opacity: 0.45; transform: scaleX(0.92); }}
  50%      {{ opacity: 1; transform: scaleX(1); }}
}}
[data-testid="stStatusContainer"] {{
  position: relative !important;
}}
[data-testid="stStatusContainer"]::before {{
  content: "";
  display: block;
  height: 3px;
  margin: 0 0 18px 0;
  border-radius: 999px;
  background: linear-gradient(90deg, transparent, rgba(0,100,255,0.4), #0064FF, rgba(0,100,255,0.4), transparent);
  animation: sq-status-pulse 1.6s ease-in-out infinite;
  opacity: 0.85;
}}
[data-testid="stStatusWidget"],
[data-testid="stStatusContainer"] {{
  background: #101012 !important;
  background-clip: padding-box !important;
  border: none !important;
  border-radius: 22px !important;
  box-shadow: var(--shadow-card) !important;
  overflow: hidden !important;
  padding: 22px 26px !important;
  color: {TEXT} !important;
  box-sizing: border-box !important;
}}
[data-testid="stStatusWidget"] p,
[data-testid="stStatusContainer"] p {{
  color: {MUTED} !important;
  font-size: 0.8125rem !important;
  margin: 8px 0 !important;
  opacity: 0.5 !important;
  letter-spacing: -0.02em !important;
}}

.block-container .mc,
.block-container .rtbl-wrap,
.block-container [data-testid="stPlotlyChart"],
.block-container .math-panel,
.block-container .sq-math-start ~ [data-testid="stHorizontalBlock"] {{
  margin-bottom: 44px !important;
}}

.block-container [data-testid="element-container"],
section[data-testid="stSidebar"] [data-testid="element-container"] {{
  overflow: hidden !important;
  box-sizing: border-box !important;
}}
/* math 카드 헤더 */
.math-card-head {{
  margin-bottom: 8px;
}}
.math-card-title {{
  font-size: 0.9375rem;
  font-weight: 700;
  color: {TEXT};
  letter-spacing: -0.02em !important;
  display: block;
}}
.math-h-sub {{
  display: block;
  font-size: 0.7rem;
  font-weight: 500;
  color: {MUTED};
  margin-top: 6px;
  letter-spacing: -0.02em !important;
  opacity: 0.5 !important;
}}
/* math 카드 하단 설명 */
.math-desc-pad {{
  margin: 4px 0 0 0 !important;
  padding: 0 !important;
}}

.hbar-brand {{
  display: flex;
  align-items: baseline;
  gap: 12px;
  flex-wrap: wrap;
}}
.sq-ring-wrap {{
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  padding: 72px 0 48px 0;
}}
.sq-ring-label {{
  font-size: 0.8125rem;
  font-weight: 500;
  color: {MUTED};
  letter-spacing: -0.02em;
  opacity: 0.5;
}}
.sq-card-skeleton {{
  height: 132px;
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

CH = 460


# ---------------------------------------------------------------------------
# Fan chart
# ---------------------------------------------------------------------------
def build_fan(pm, s0, yrs, cur, ticker):
    """Fan chart: translucent quantile bands (SVG fill) + mean/median lines (Scattergl)."""
    sym = currency_symbol(cur)
    hf = ",.0f" if cur == "KRW" else ",.2f"
    _, npt = pm.shape
    tf = np.linspace(0.0, yrs, npt)
    q_mean = np.mean(pm, axis=0)
    q025, q05, q125, q25, q50, q75, q875, q95, q975 = np.quantile(
        pm,
        [0.025, 0.05, 0.125, 0.25, 0.5, 0.75, 0.875, 0.95, 0.975],
        axis=0,
    )

    di, _ = _ds(tf, 160)
    t = tf[di]
    d025, d05, d125, d25 = q025[di], q05[di], q125[di], q25[di]
    d50, d75, d875, d95, d975 = q50[di], q75[di], q875[di], q95[di], q975[di]
    d_mean = q_mean[di]

    # Toss 다크 테마용 형광 시안 (평균 경로)
    mean_line_color = "#22d3ee"

    fig = go.Figure()
    tr = t[::-1]
    # fill=toself 는 WebGL(Scattergl) 미지원 → 밴드만 go.Scatter, α 0.05~0.1
    bands = [
        ("95% 신뢰구간", d025, d975, "rgba(0,100,255,0.06)"),
        ("75% 신뢰구간", d125, d875, "rgba(0,100,255,0.08)"),
        ("50% 신뢰구간", d25, d75, "rgba(0,100,255,0.10)"),
    ]
    for name, lo, hi, color in bands:
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, tr]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            name=name,
            hoverinfo="skip",
        ))

    median_custom = np.column_stack([d05, d95, d25, d75])
    fig.add_trace(go.Scattergl(
        x=t,
        y=d50,
        mode="lines",
        customdata=median_custom,
        line=dict(color=ACCENT, width=2.6, shape="linear"),
        name="Median (중앙값)",
        hovertemplate=(
            f"<b>%{{x:.2f}}Y 예측</b><br>"
            f"Median: {sym}%{{y:{hf}}}<br>"
            f"90% 범위: {sym}%{{customdata[0]:{hf}}} ~ {sym}%{{customdata[1]:{hf}}}<br>"
            f"50% 범위: {sym}%{{customdata[2]:{hf}}} ~ {sym}%{{customdata[3]:{hf}}}"
            "<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scattergl(
        x=t,
        y=d_mean,
        mode="lines",
        line=dict(color=mean_line_color, width=5.0, shape="linear"),
        name="기대 평균 경로 (Mean)",
        hovertemplate=(
            f"<b>%{{x:.2f}}Y 예측</b><br>"
            f"Mean: {sym}%{{y:{hf}}}<extra></extra>"
        ),
    ))

    fig.add_hline(y=s0, line_color="rgba(220,228,248,0.42)", line_dash="dash", line_width=1.35)
    fig.add_annotation(x=0.01, y=s0, xref="paper",
        text=f" 현재가 {_fp(s0,cur)} ", showarrow=False, yanchor="bottom",
        font=dict(color="#fff", size=10), bgcolor="rgba(6,8,15,0.7)", borderpad=3)

    tp, tl = _mticks(yrs)
    yl = min(float(q025.min()), s0) * 0.88
    yh = float(q975.max()) * 1.12

    label_points = []
    if yrs >= 0.5:
        label_points.append((0.5, "6M"))
    if yrs >= 1.0:
        label_points.append((1.0, "1Y"))
    elif yrs > 0:
        label_points.append((yrs, f"{int(round(yrs * 12))}M"))

    for xval, label in label_points:
        idx = int(np.argmin(np.abs(tf - xval)))
        yval = float(q_mean[idx])
        fig.add_annotation(
            x=float(tf[idx]),
            y=yval,
            text=f"<b>{label}</b> · Mean {fmt_price(yval, cur)}",
            showarrow=True,
            arrowhead=2,
            arrowwidth=1.2,
            arrowcolor=mean_line_color,
            ax=22,
            ay=-34,
            font=dict(color="#F4F5F7", size=11, family="Pretendard Variable, sans-serif"),
            bgcolor="rgba(16,16,18,0.92)",
            bordercolor="rgba(34,211,238,0.35)",
            borderwidth=1,
            borderpad=6,
        )

    target_t = 1.0 if yrs >= 1.0 else yrs
    target_label = "1Y" if yrs >= 1.0 else f"{int(round(yrs * 12))}M"
    target_idx = int(np.argmin(np.abs(tf - target_t)))
    side_cards = [
        ("최선 95th", float(q95[target_idx]), ACCENT),
        ("기대 50th", float(q50[target_idx]), TEXT),
        ("최악 5th", float(q05[target_idx]), RED),
    ]
    for ypaper, (label, value, color) in zip([0.78, 0.62, 0.46], side_cards):
        fig.add_annotation(
            x=1.02,
            y=ypaper,
            xref="paper",
            yref="paper",
            text=(
                f"<span style='color:{MUTED};font-size:10px'>{target_label} · {label}</span>"
                f"<br><b style='color:{color};font-size:15px'>{fmt_price(value, cur)}</b>"
            ),
            showarrow=False,
            xanchor="left",
            align="left",
            font=dict(color=TEXT, family="Pretendard Variable, sans-serif"),
            bgcolor="rgba(16,16,18,0.96)",
            bordercolor="rgba(255,255,255,0)",
            borderwidth=0,
            borderpad=9,
        )

    fig.update_layout(**_LAY,
        title=dict(
            text=f"<b>기간별 주가 예측 팬 차트</b> <span style='color:#8B93A1;font-weight:500'>· {ticker}</span>",
            font=dict(size=15, color="#F4F5F7", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.04,
            y=0.97,
            xanchor="left",
        ),
        margin=dict(l=56, r=210, t=76, b=104),
        legend=dict(
            orientation="h",
            bgcolor="rgba(16,16,18,0.82)",
            bordercolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=10, color="#8B93A1", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.5,
            y=-0.22,
            xanchor="center",
            yanchor="top",
            traceorder="normal",
        ),
        xaxis=dict(**_AX, title="예측 기간 Period", tickvals=tp, ticktext=tl),
        yaxis=dict(**_AX, title=f"주가 Price ({cur})", range=[yl, yh],
                   tickformat=hf, tickprefix=sym),
        height=CH)

    return fig


# ---------------------------------------------------------------------------
# Math formula section — st.latex 기반 (KaTeX 렌더링)
# ---------------------------------------------------------------------------
def _render_math_section(
    params: GbmParams,
    cur: str,
    jump_lambda: float,
    jump_mu: float,
    jump_sigma: float,
) -> None:
    """Render math model cards using st.latex (KaTeX) + HTML card frames.

    Each card = markdown header ＋ st.latex ＋ markdown description.
    The column containers are styled as cards via the .sq-math-start CSS sentinel.
    파라미터(λ, μ_J, σ_J, σ, μ)는 세션에서 실시간으로 읽어 자동 반영됩니다.
    """
    fp_v = lambda v: _fp(v, cur)
    a   = ACCENT          # Toss Blue — 키워드 강조
    jl  = jump_lambda
    jm  = jump_mu
    js  = jump_sigma
    jump_on = jl > 0.0

    # (타이틀, 부제, LaTeX 수식, 설명 HTML)
    cards: list[tuple[str, str, str, str]] = [
        (
            "GBM 확률 미분 방정식",
            "Stochastic Differential Equation",
            r"dS = \mu S \, dt + \sigma S \, dW",
            f'주가의 <b style="color:{a}">추세(drift)</b>와 '
            f'<b style="color:{a}">무작위 변동(diffusion)</b>을 모델링합니다. '
            f'<b>dW</b>는 위너 과정 증분입니다.',
        ),
        (
            "해석해 — Itô's Lemma",
            "이토 보조정리",
            r"S(T)=S_0\exp\!\Bigl[(\mu-\tfrac{1}{2}\sigma^2)T+\sigma\sqrt{T}\,Z\Bigr]",
            f'SDE의 정확한 해. <b style="color:{a}">Z ~ N(0,1)</b> 표준정규 난수를 대입해 '
            f'미래 시점 T의 주가를 직접 계산합니다.',
        ),
        (
            "이산화 경로",
            "Exact Discretization",
            r"S_{t+\Delta t}=S_t\exp\!\Bigl[(\mu-\tfrac{1}{2}\sigma^2)\Delta t+\sigma\sqrt{\Delta t}\,Z_t\Bigr]",
            f'Fan Chart 각 타임스텝에 적용됩니다. 확산 난수에는 '
            f'<b style="color:{a}">대칭 변수법(antithetic Z / −Z)</b>을 사용합니다.',
        ),
        (
            "드리프트 시장 수축",
            "Shrinkage — risk governance",
            r"\mu_{\mathrm{sim}}=\tfrac{1}{2}\hat{\mu}+\tfrac{1}{2}\mu_m,\quad \mu_m=8\%",
            f'데이터에서 추정한 연율 <b>μ̂</b>만 쓰면 과거 급등·추세가 미래 전망을 과대 반영할 수 있어, '
            f'시장 평균형 prior <b>μ<sub>m</sub>={MU_MARKET_PRIOR:.0%}</b>와 '
            f'<b>{int(MU_SHRINKAGE_SAMPLE_WEIGHT * 100)}:{100 - int(MU_SHRINKAGE_SAMPLE_WEIGHT * 100)}</b> 가중 혼합 후 시뮬에 넣습니다.',
        ),
        (
            "Volatility Drag",
            "Itô / convexity in log space",
            r"\mathbb{E}[\ln S]\ \text{의 GBM 확산 항}\ \Rightarrow\ (\mu-\tfrac{1}{2}\sigma^2)\,T",
            f'지수화된 주가에서 <b style="color:{a}">로그</b>로 넘어오면서 생기는 '
            f'<b>−½σ²</b> 보정입니다. σ가 크면 (<b>μ는 같아도</b>) '
            f'로그기대·중앙값이 내려가 <b style="color:{a}">고변동 종목의 낙관 편향</b>을 완화합니다. '
            f'엔진은 각 스텝에 동일한 <code>(μ−½σ²)Δt</code>를 사용합니다.',
        ),
        (
            "VaR — Value at Risk",
            "최대 예상 손실 (95%)",
            r"\mathrm{VaR}_{95\%}=S_0-Q_{0.05}(S_T)",
            f'<b style="color:{a}">하위 5% 경로</b> 기준 현재가 대비 최대 손실. '
            f'1,000만 경로 비모수적 추정으로 높은 정밀도를 확보합니다.',
        ),
        (
            "CVaR — Expected Shortfall",
            "조건부 꼬리 손실",
            r"\mathrm{CVaR}_{95\%}=\mathbb{E}\!\left[\mathrm{Loss}\mid\mathrm{Loss}\geq\mathrm{VaR}_{95\%}\right]",
            f'Loss = max(0, S₀ − Sₜ) 하위 5% 평균. '
            f'<b style="color:{a}">꼬리 리스크(tail risk)</b>를 VaR보다 보수적으로 요약합니다.',
        ),
        (
            "Sharpe Ratio",
            "위험 조정 수익률",
            r"SR=\dfrac{\mathbb{E}[\,R_p-R_f\,]}{\sigma_p}",
            f'<b style="color:{a}">초과 수익</b>을 변동성으로 나눈 위험 조정 지표. '
            f'높을수록 단위 위험당 수익이 효율적입니다.',
        ),
        (
            "Kelly Criterion",
            "모형상 참고 비중 (연율 μ, σ)",
            r"f^\star=\max\!\left(0,\min\!\left(1,\dfrac{\mu-r_f}{\sigma^2}\right)\right),\ r_f=3\%",
            f'시뮬에 쓰는 연율 <b>μ, σ</b>로 근사합니다. <b>f∈[0,1]</b>로 캡하여 100% 초과 베팅을 막습니다. '
            f'입력 오차에 매우 민감하므로 실제 포지션은 비용·유동성·제약을 반영해 크게 축소해야 합니다.',
        ),
        (
            "Sortino Ratio",
            "하방 변동성만 분모",
            r"\mathrm{Sortino}=\dfrac{\overline{\ln(S_T/S_0)}-\ln(1+r_f)}{\sigma_-},\quad \sigma_-=\mathrm{std}\bigl(\{r_i\mid r_i<0\}\bigr)",
            f'동일한 <b>초과 평균 로그수익</b> 분자를 쓰되, 샤프와 달리 '
            f'<b style="color:{a}">손실 경로만</b>의 표준편차를 분모로 삼아 하방 리스크 대비 효율을 강조합니다.',
        ),
        (
            "Merton Jump Diffusion",
            "점프 확산 항 — 실시간 파라미터",
            r"\ln\frac{S_T}{S_0}=(\mu-\tfrac{1}{2}\sigma^2)T+\sigma\sqrt{T}\,Z+\textstyle\sum_{i=1}^{N_T}J_i",
            f'N<sub>T</sub>~Poisson(λT), J<sub>i</sub>~N(μ<sub>J</sub>,σ<sub>J</sub>²). &nbsp;'
            f'<b style="color:{a if jump_on else SUBTLE}">{"● 점프 활성" if jump_on else "○ λ=0 → GBM만"}</b>'
            f'&nbsp;— λ=<b>{jl:.4f}</b>, μ<sub>J</sub>=<b>{jm:.4f}</b>, σ<sub>J</sub>=<b>{js:.4f}</b>',
        ),
    ]

    # CSS sentinel — 이 div 이후에 오는 st.columns를 카드로 스타일링
    st.markdown('<div class="sq-math-start"></div>', unsafe_allow_html=True)

    # 2-칼럼 그리드로 카드 렌더
    for row in range(0, len(cards), 2):
        c1, c2 = st.columns(2, gap="large")
        for j, col in enumerate([c1, c2]):
            idx = row + j
            if idx >= len(cards):
                break
            title, sub, latex_src, desc_html = cards[idx]
            with col:
                # 카드 헤더
                st.markdown(
                    f'<div class="math-card-head">'
                    f'<span class="math-card-title">{title}</span>'
                    f'<span class="math-h-sub">{sub}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # KaTeX 수식
                st.latex(latex_src)
                # 설명 (핵심 용어 Toss Blue)
                st.markdown(
                    f'<p class="math-desc math-desc-pad">{desc_html}</p>',
                    unsafe_allow_html=True,
                )

    # 파라미터 요약 테이블
    st.markdown(
        f'<div class="math-panel toss-card" style="margin-top:24px;">'
        f'<h3>입력 파라미터<span class="math-h-sub">Input Parameters Used — 실시간 연동</span></h3>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px 32px;margin-top:18px;">'
        f'<div class="math-desc"><span class="math-var">S₀</span><br><b>현재 주가</b><br>{fp_v(params.s0)}</div>'
        f'<div class="math-desc"><span class="math-var" style="color:{a}">μ</span><br><b>기대수익률</b><br>{params.mu:.6f}</div>'
        f'<div class="math-desc"><span class="math-var" style="color:{a}">σ</span><br><b>변동성 (연율)</b><br>{params.sigma:.6f}</div>'
        f'<div class="math-desc"><span class="math-var">T</span><br><b>예측 기간</b><br>years</div>'
        f'<div class="math-desc"><span class="math-var" style="color:{"#e8a85c" if jump_on else SUBTLE}">λ</span>'
        f'<br><b>점프 강도</b><br><b style="color:{"#e8a85c" if jump_on else SUBTLE}">{jl:.6f}</b></div>'
        f'<div class="math-desc"><span class="math-var">μ<sub>J</sub></span><br><b>점프 평균</b><br>{jm:.6f}</div>'
        f'<div class="math-desc"><span class="math-var">σ<sub>J</sub></span><br><b>점프 변동성</b><br>{js:.6f}</div>'
        f'<div class="math-desc"><span class="math-var">Z</span><br><b>표준정규 난수</b><br>Antithetic pairs</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric card
# ---------------------------------------------------------------------------
def _mc(label, value, delta="", lg=False, large=False, vc: str | None = None):
    c = "mc-val-lg" if (lg or large) else "mc-val"
    sty = f' style="color:{vc}"' if vc is not None else ""
    return (
        f'<div class="mc toss-card">'
        f'<div class="mc-lbl">{label}</div>'
        f'<div class="{c}"{sty}>{value}</div>'
        f'{delta}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple[str, str]:
    """사이드바: 브랜드 + 종목 preset selectbox. 반환값은 심볼 또는 TICKER_CUSTOM."""

    with st.sidebar:
        st.markdown(
            '<div class="brand-bar">'
            '<span class="brand-title">Stellar-Quant</span>'
            '<span class="brand-sep">·</span>'
            '<span class="brand-author">이준협</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sb-section-hd">종목</div>',
            unsafe_allow_html=True,
        )
        preset = st.selectbox(
            "종목 선택",
            options=TICKER_OPTIONS,
            format_func=lambda s: TICKER_LABELS[s],
            index=0,
            key="sq_ticker_preset",
            help="직접 입력… 을 고르면 본문에서 티커를 입력합니다.",
        )

        st.caption(
            f"시뮬 고정값: {FIXED_N_PATHS:,} paths · {FIXED_HORIZON_YEARS:g}y · "
            f"fan {FIXED_FAN_PATHS:,} · steps {FIXED_N_STEPS}"
        )
        st.markdown(
            '<div class="sb-section-hd">모형</div>',
            unsafe_allow_html=True,
        )
        jump_mode = st.selectbox(
            "점프 모드",
            options=[JUMP_MODE_CONSERVATIVE, JUMP_MODE_HISTORICAL, JUMP_MODE_OFF],
            format_func=lambda s: JUMP_MODE_LABELS[s],
            index=0,
            key="sq_jump_mode",
            help="보수적 스트레스는 하방 점프를 고정 적용하고, 역사적 추정은 최근 데이터에서 탐지한 점프만 사용합니다.",
        )

        st.markdown(
            '<div class="sb-foot">'
            'C++23 Multi-threaded GBM Engine<br>'
            'pybind11 · zero-copy · WebGL</div>',
            unsafe_allow_html=True,
        )

        ej = st.session_state.get("est_jump")
        if ej is not None:
            st.caption(
                f"참고 추정(역사적): λ̂={ej.lambda_annual:.4f}, "
                f"μ̂_J={ej.mu_jump:.4f}, σ̂_J={ej.sigma_jump:.4f}"
            )

    return str(preset), str(jump_mode)


def _render_top_controls(preset: str, jump_mode: str) -> SidebarConfig:
    """본문: 히어로 + (직접 입력 시) 티커 입력 + 시뮬레이션 버튼."""

    st.markdown(
        '<div class="top-shell">'
        '<p class="top-kicker">C++23 Monte Carlo Risk Engine</p>'
        '<h1 class="top-title">투자 위험을 한눈에 확인하세요</h1>'
        '<p class="top-sub">'
        '왼쪽 사이드바에서 종목을 고른 뒤 시뮬레이션을 시작하세요. '
        '최근 1년 yfinance 데이터로 현재가와 변동성을 반영합니다. '
        '상승확률은 보조 지표로 보고, 가격 구간과 꼬리손실을 함께 확인하세요.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if preset == TICKER_CUSTOM:
        ticker = st.text_input(
            "종목코드 (Yahoo Finance)",
            value=str(st.session_state.get("sq_ticker_custom", "")),
            max_chars=20,
            placeholder="예: COIN, 035420.KS",
            key="sq_ticker_custom",
            help="사이드바에서 「직접 입력…」을 선택한 경우에만 사용됩니다.",
        ).strip().upper()
    else:
        ticker = str(preset).strip().upper()
        st.markdown(
            f'<p style="color:{MUTED};font-size:0.9rem;margin:0 0 16px 0;">'
            f'선택 종목: <b style="color:{TEXT}">{ticker}</b></p>',
            unsafe_allow_html=True,
        )

    run = st.button(
        "시뮬레이션 시작",
        use_container_width=True,
        type="primary",
        key="sq_run_simulation",
    )
    return _fixed_engine_config(ticker, bool(run), jump_mode)


@st.cache_data(show_spinner=False, ttl=3600)
def _load_market_data(ticker: str, period: str = "1y") -> MarketData:
    """Fetch 1y yfinance prices and convert them into GBM and jump parameters."""

    close = fetch_prices(ticker, period=period)
    params = estimate_gbm_params(close, ticker=ticker)
    n_ret = max(1, len(close) - 1)
    sample_years = max(0.25, n_ret / 252.0)
    mu_uncertainty = float(np.clip(params.sigma / np.sqrt(sample_years), 0.05, 0.30))
    return MarketData(
        params=params,
        jump=estimate_jump_params(close),
        mu_uncertainty=mu_uncertainty,
        source="yfinance",
    )


@st.cache_data(show_spinner=False, ttl=1800)
def _run_inline_backtest(ticker: str) -> dict:
    """Rolling 1-year calibration backtest for the validation tab (cached 30 min).

    Uses 7 years of data with 6-month strides and 10 K Monte Carlo paths — fast
    enough for interactive use while producing enough splits (≈ 10) for a
    meaningful calibration check.
    """
    try:
        from backtest import run_backtest as _bt_run, _summarize as _bt_summarize

        close = fetch_prices(ticker, period="7y")
        market_close = None
        try:
            market_close = fetch_prices("SPY", period="7y")
        except Exception:
            pass

        rows, currency = _bt_run(
            close,
            ticker=ticker,
            market_close=market_close,
            train_days=504,
            horizon_days=252,
            stride_days=126,
            n_paths=10_000,
            seed=42,
        )
        summary = _bt_summarize(rows)
        rows_data = [
            {
                "model": r.model,
                "asof": r.asof,
                "realized_percentile": r.realized_percentile,
                "interval_90_hit": r.interval_90_hit,
                "direction_hit": r.direction_hit,
                "median_abs_error_pct": r.median_abs_error_pct,
                "brier_up": r.brier_up,
            }
            for r in rows
        ]
        return {"rows": rows_data, "summary": summary, "currency": currency, "error": None}
    except Exception as exc:
        return {"rows": [], "summary": [], "currency": "USD", "error": str(exc)}


def _render_backtest_inline(bt: dict, ticker: str) -> None:
    """Render backtest summary comparison table and calibration scatter chart."""
    if bt.get("error"):
        st.warning(f"백테스트 실패: {bt['error']}")
        return

    summary: list[dict] = bt.get("summary", [])
    rows: list[dict] = bt.get("rows", [])
    if not summary:
        st.info("데이터 기간이 짧아 분할이 생성되지 않았습니다. CLI로 더 긴 기간을 지정하세요.")
        return

    # ── Summary comparison table ──────────────────────────────────────────
    st.markdown(
        '<div class="stitle">모형 비교 요약<span class="stitle-sub">'
        f'rolling 1Y backtest · {ticker}</span></div>',
        unsafe_allow_html=True,
    )

    table_rows_html = ""
    for item in summary:
        model = str(item["model"])
        cov = float(item["interval_90_coverage_pct"])
        dir_acc = float(item["direction_accuracy_pct"])
        med_err = float(item["mean_median_abs_error_pct"])
        brier = float(item["mean_brier_up"])
        mean_pctl = float(item["mean_realized_percentile"])
        n_splits = int(item["splits"])

        cov_cls = "risk-blue" if abs(cov - 90.0) < 15.0 else "risk-red"
        pctl_cls = "risk-blue" if abs(mean_pctl - 50.0) < 12.0 else "risk-red"
        row_weight = "800" if model == "stellar" else "500"

        table_rows_html += (
            f"<tr>"
            f'<td style="font-weight:{row_weight}">{model}</td>'
            f"<td>{n_splits}</td>"
            f'<td><span class="{cov_cls}" style="font-weight:700">{cov:.1f}%</span></td>'
            f"<td>{dir_acc:.1f}%</td>"
            f"<td>{med_err:.1f}%</td>"
            f"<td>{brier:.3f}</td>"
            f'<td><span class="{pctl_cls}" style="font-weight:700">{mean_pctl:.1f}%</span></td>'
            f"</tr>"
        )

    st.markdown(
        f'<div class="rtbl-wrap toss-card"><table class="rtbl"><thead><tr>'
        f"<th>Model</th><th>Splits</th>"
        f"<th>90% Coverage <small style='opacity:0.6'>(목표 90%)</small></th>"
        f"<th>Direction %</th><th>Median Err %</th><th>Brier Up</th>"
        f"<th>Mean Realized Pctl <small style='opacity:0.6'>(목표 50%)</small></th>"
        f"</tr></thead><tbody>{table_rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    # ── Calibration scatter (stellar model only) ──────────────────────────
    stellar_rows = [r for r in rows if r["model"] == "stellar"]
    if len(stellar_rows) < 2:
        return

    asofs = [r["asof"] for r in stellar_rows]
    pctls = [float(r["realized_percentile"]) for r in stellar_rows]
    hits = [bool(r["interval_90_hit"]) for r in stellar_rows]
    marker_colors = [ACCENT if h else RED for h in hits]
    mean_pctl_stellar = float(np.mean(pctls))

    if mean_pctl_stellar > 55:
        bias_label = f"모형 비관 편향 · 평균 {mean_pctl_stellar:.1f}%"
    elif mean_pctl_stellar < 45:
        bias_label = f"모형 낙관 편향 · 평균 {mean_pctl_stellar:.1f}%"
    else:
        bias_label = f"편향 없음 · 평균 {mean_pctl_stellar:.1f}% (이상적 50%)"

    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=list(range(len(asofs))),
            y=pctls,
            mode="markers+lines",
            marker=dict(
                color=marker_colors,
                size=11,
                line=dict(width=1.5, color="rgba(0,0,0,0.35)"),
            ),
            line=dict(color=SUBTLE, width=1.5, dash="dot"),
            text=[
                f"as-of {a}<br>실현 퍼센타일: {p:.1f}%<br>{'구간 적중 ✓' if h else '구간 빗나감 ✗'}"
                for a, p, h in zip(asofs, pctls, hits)
            ],
            hoverinfo="text",
            name="실현 퍼센타일",
        )
    )
    # Ideal band: uniform ~ [5, 95] expected range
    fig_bt.add_hrect(y0=5, y1=95, fillcolor="rgba(0,100,255,0.04)", line_width=0)
    fig_bt.add_hline(
        y=50.0,
        line_color="rgba(255,255,255,0.22)",
        line_dash="dash",
        line_width=1.2,
        annotation_text="이상적 50%",
        annotation_position="top right",
        annotation_font=dict(color=MUTED, size=10),
    )

    tick_labels = [a[:7] if len(a) >= 7 else a for a in asofs]
    fig_bt.update_layout(
        **_LAY,
        title=dict(
            text=(
                f"<b>Stellar 실현 퍼센타일 추이</b>"
                f"<span style='color:#8B93A1;font-weight:500'> · {bias_label}</span>"
            ),
            font=dict(size=14, color="#F4F5F7", family="Pretendard Variable, Pretendard, sans-serif"),
            x=0.04,
            y=0.96,
            xanchor="left",
        ),
        xaxis=dict(
            **_AX,
            title="백테스트 시점 (as-of date)",
            tickvals=list(range(len(asofs))),
            ticktext=tick_labels,
        ),
        yaxis=dict(**_AX, title="실현 퍼센타일 (%)", range=[-3, 103]),
        height=320,
        margin=dict(l=56, r=32, t=56, b=72),
        showlegend=False,
    )
    st.plotly_chart(fig_bt, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    st.caption(
        "파란 점 = 90% 구간 적중 · 빨간 점 = 구간 빗나감 · "
        "점이 50% 주변에 고르게 분포할수록 잘 보정된 모형입니다."
    )


def _manual_market_data(config: SidebarConfig) -> MarketData:
    """Build model parameters from sidebar fallback values."""

    mu_shr = shrink_mu_toward_market_prior(config.manual_mu)
    mu_c, sig_c = clamp_gbm_for_simulation(mu_shr, config.manual_sigma)
    return MarketData(
        params=GbmParams(
            s0=config.manual_s0,
            mu=mu_c,
            sigma=sig_c,
            currency=detect_currency(config.ticker),
        ),
        jump=JumpParams(lambda_annual=0.0, mu_jump=0.0, sigma_jump=0.0),
        mu_uncertainty=0.20,
        source="manual",
    )


def _resolve_jump_config(config: SidebarConfig, jump: JumpParams) -> SidebarConfig:
    """Apply the selected jump mode after market data is available."""

    if config.jump_mode == JUMP_MODE_OFF:
        return replace(config, jump_lambda=0.0, jump_mu=0.0, jump_sigma=0.0)
    if config.jump_mode == JUMP_MODE_HISTORICAL:
        return replace(
            config,
            jump_lambda=max(0.0, jump.lambda_annual),
            jump_mu=jump.mu_jump,
            jump_sigma=max(0.0, jump.sigma_jump),
        )
    return replace(
        config,
        jump_lambda=FIXED_JUMP_LAMBDA,
        jump_mu=FIXED_JUMP_MU,
        jump_sigma=FIXED_JUMP_SIGMA,
    )


def _run_cpp_engine(config: SidebarConfig, params: GbmParams) -> EngineOutput:
    """Call the pybind11 C++ Monte Carlo engine and normalize arrays."""

    sim = _cached_simulator()
    t0 = time.perf_counter()
    terminal = np.asarray(
        sim.simulate_gbm_paths(
            n_paths=config.n_paths,
            s0=params.s0,
            mu=params.mu,
            sigma=params.sigma,
            t=config.years,
            seed=42,
            n_threads=config.n_threads,
            jump_lambda=config.jump_lambda,
            jump_mu=config.jump_mu,
            jump_sigma=config.jump_sigma,
        ),
        dtype=np.float64,
    )
    path_matrix = np.asarray(
        sim.simulate_gbm_path_matrix(
            n_paths=config.fan_paths,
            n_steps=config.n_steps,
            s0=params.s0,
            mu=params.mu,
            sigma=params.sigma,
            t=config.years,
            seed=43,
            n_threads=config.n_threads,
            jump_lambda=config.jump_lambda,
            jump_mu=config.jump_mu,
            jump_sigma=config.jump_sigma,
        ),
        dtype=np.float64,
    )
    return EngineOutput(
        terminal=terminal,
        path_matrix=path_matrix,
        elapsed=time.perf_counter() - t0,
    )


def _apply_mu_uncertainty(
    terminal: np.ndarray,
    path_matrix: np.ndarray,
    years: float,
    mu_uncertainty: float,
    seed: int = 20260509,
) -> tuple[np.ndarray, np.ndarray]:
    """Widen simulated prices by sampling annual drift estimation error per path."""

    if mu_uncertainty <= 0.0 or years <= 0.0:
        return terminal, path_matrix
    rng = np.random.default_rng(seed)

    terminal_noise = rng.normal(0.0, mu_uncertainty, size=terminal.shape[0])
    terminal_adj = terminal * np.exp(terminal_noise * years)

    path_noise = rng.normal(0.0, mu_uncertainty, size=path_matrix.shape[0])
    times = np.linspace(0.0, years, path_matrix.shape[1])
    path_adj = path_matrix * np.exp(path_noise[:, None] * times[None, :])
    return terminal_adj.astype(np.float64, copy=False), path_adj.astype(np.float64, copy=False)


def _build_dashboard_result(
    config: SidebarConfig,
    market_data: MarketData,
    engine_output: EngineOutput,
) -> DashboardResult:
    """Compute risk metrics and Plotly figures from engine output."""

    params = market_data.params
    terminal, path_matrix = _apply_mu_uncertainty(
        engine_output.terminal,
        engine_output.path_matrix,
        config.years,
        market_data.mu_uncertainty,
    )
    metrics = compute_risk_metrics(
        terminal,
        params.s0,
        sigma_annual=params.sigma,
        horizon_years=config.years,
    )
    return DashboardResult(
        ticker=config.ticker,
        params=params,
        terminal=terminal,
        path_matrix=path_matrix,
        metrics=metrics,
        elapsed=engine_output.elapsed,
        n_paths=config.n_paths,
        years=config.years,
        n_steps=config.n_steps,
        n_threads=config.n_threads,
        jump_mode=config.jump_mode,
        jump_lambda=config.jump_lambda,
        jump_mu=config.jump_mu,
        jump_sigma=config.jump_sigma,
        mu_uncertainty=market_data.mu_uncertainty,
        fan_fig=build_fan(path_matrix, params.s0, config.years, params.currency, config.ticker),
    )


def _store_dashboard_result(result: DashboardResult, jump: JumpParams) -> None:
    """Persist the render-ready result across Streamlit reruns."""

    st.session_state.update(
        dict(
            ticker=result.ticker,
            params=result.params,
            terminal=result.terminal,
            path_matrix=result.path_matrix,
            metrics=result.metrics,
            elapsed=result.elapsed,
            n_paths=result.n_paths,
            years=result.years,
            n_steps=result.n_steps,
            n_threads=result.n_threads,
            jump_mode=result.jump_mode,
            jump_lambda=result.jump_lambda,
            jump_mu=result.jump_mu,
            jump_sigma=result.jump_sigma,
            mu_uncertainty=result.mu_uncertainty,
            fan_fig=result.fan_fig,
            est_jump=jump,
        )
    )


def _result_from_session_state() -> DashboardResult:
    """Rehydrate the dashboard result after Streamlit reruns."""

    ss = st.session_state
    return DashboardResult(
        ticker=ss["ticker"],
        params=ss["params"],
        terminal=ss["terminal"],
        path_matrix=ss["path_matrix"],
        metrics=ss["metrics"],
        elapsed=ss["elapsed"],
        n_paths=ss["n_paths"],
        years=ss["years"],
        n_steps=int(ss.get("n_steps", 252)),
        n_threads=int(ss.get("n_threads", 0)),
        jump_mode=str(ss.get("jump_mode", JUMP_MODE_CONSERVATIVE)),
        jump_lambda=float(ss.get("jump_lambda", 0.0)),
        jump_mu=float(ss.get("jump_mu", 0.0)),
        jump_sigma=float(ss.get("jump_sigma", 0.0)),
        mu_uncertainty=float(ss.get("mu_uncertainty", 0.0)),
        fan_fig=ss.get("fan_fig"),
    )


def _render_landing() -> None:
    st.markdown(
        f'<div style="margin:28px 0 0 0;color:{MUTED};font-size:0.95rem;'
        f'line-height:1.7;letter-spacing:-0.02em;">'
        f'결과가 생성되면 이 위치에 리스크 리포트 요약 카드가 표시됩니다.'
        f'</div>',
        unsafe_allow_html=True,
    )


def _run_simulation_flow(config: SidebarConfig) -> bool:
    """Execute market-data loading, C++ simulation, metrics, and chart building."""

    tick = config.ticker.strip()

    loading = st.empty()
    loading.markdown(
        '<div class="sq-loading">데이터 분석 중...'
        '<span>최근 1년 가격 데이터와 fallback 설정을 확인하고 있습니다.</span></div>',
        unsafe_allow_html=True,
    )

    if tick:
        try:
            market_data = _load_market_data(tick, period="1y")
        except (YahooFinanceFetchError, ValueError) as err:
            st.warning(
                "**자동 시세 데이터를 사용할 수 없어 수동 입력값을 적용했습니다.**\n\n"
                f"{err}"
            )
            market_data = _manual_market_data(config)
    else:
        st.warning("종목코드가 비어 있어 수동 fallback 값으로 진행합니다.")
        market_data = _manual_market_data(config)

    config = _resolve_jump_config(config, market_data.jump)

    loading.markdown(
        '<div class="sq-loading">C++ 엔진 시뮬레이션 가동 중...'
        f'<span>{config.n_paths:,}개 경로를 멀티스레드로 계산합니다.</span></div>',
        unsafe_allow_html=True,
    )
    engine_output = _run_cpp_engine(config, market_data.params)

    loading.markdown(
        '<div class="sq-loading">결과 산출 완료'
        '<span>리스크 지표와 상세 분석 화면을 정리하고 있습니다.</span></div>',
        unsafe_allow_html=True,
    )
    result = _build_dashboard_result(config, market_data, engine_output)
    _store_dashboard_result(result, market_data.jump)
    loading.empty()

    if market_data.source == "yfinance":
        st.caption(
            f"자동 반영: S0={market_data.params.s0:,.2f}, "
            f"σ={market_data.params.sigma:.4f} · 최근 1년 일간 수익률 기반 · "
            f"점프 모드 {JUMP_MODE_LABELS.get(config.jump_mode, config.jump_mode)}"
        )
    else:
        st.caption(
            f"Fallback 적용: S0={market_data.params.s0:,.2f}, "
            f"σ={market_data.params.sigma:.4f} · 수동 입력값 기반 · "
            f"점프 모드 {JUMP_MODE_LABELS.get(config.jump_mode, config.jump_mode)}"
        )

    st.toast("시뮬레이션 완료!", icon="✅")
    return True


def _risk_summary(metrics: dict[str, float]) -> tuple[str, str, str]:
    """정규화 Risk Score(첨도 가중) 구간으로 톤·설명 결정."""

    rs = float(metrics.get("risk_score", 0.0))
    fk = float(metrics.get("fat_tail_feel_index", 0.0))
    if rs >= 3.5:
        return (
            "위험도 매우 높음 (Critical)",
            "risk-pill-red",
            f"Fat-tail·VaR 부담이 σ√T 스케일 대비 매우 큽니다. (Risk Score {rs:.2f}, 체감 꼬리 {fk:.2f})",
        )
    if rs >= 2.5:
        return (
            "위험도 높음 (High)",
            "risk-pill-red",
            f"점프·꼬리로 인한 손실 위험이 큽니다. (Score {rs:.2f}, 체감 꼬리 {fk:.2f})",
        )
    if rs >= 1.5:
        return (
            "위험도 보통 (Moderate)",
            "risk-pill-blue",
            f"단순 BS 스케일 대비 VaR 부담이 다소 큽니다. (Score {rs:.2f})",
        )
    return (
        "위험도 낮음 (Low)",
        "risk-pill-blue",
        f"통계적 허용 범위 내 — σ√T 대비 상대 VaR이 낮습니다. (Score {rs:.2f})",
    )


def _render_risk_summary(result: DashboardResult) -> None:
    metrics, params = result.metrics, result.params
    fp = lambda v: _fp(v, params.currency)
    up = metrics["up_probability_pct"]
    if up > 90.0:
        st.warning(
            "최근 추세가 과도하게 반영되었습니다. 상승 확률이 90%를 넘습니다 — "
            "추정 드리프트·단기 추세에 기대 과열이 섞였을 수 있으니 보수적으로 해석하세요."
        )
    title, tone, desc = _risk_summary(metrics)
    median_pct = _dpct(metrics["p50"], params.s0)
    up_cls = "risk-blue" if up >= 50 else "risk-red"
    median_cls = "risk-blue" if median_pct >= 0 else "risk-red"

    sig_ann_pct = params.sigma * 100.0
    sig_sqrt_t = float(metrics.get("sigma_sqrt_horizon", 0.0))
    xk = float(metrics.get("log_return_excess_kurtosis", 0.0))
    ft = float(metrics.get("fat_tail_feel_index", 0.0))
    tadj = float(metrics.get("tail_adjustment_factor", 1.0))
    rs = float(metrics.get("risk_score", 0.0))
    rsb = float(metrics.get("risk_score_base", 0.0))
    rs_cls = "risk-red" if rs >= 2.5 else "risk-blue"

    drag_ann = float(params.mu - 0.5 * params.sigma * params.sigma)

    risk_meta = (
        f'<div class="risk-meta">'
        f'<strong>연율화 총 변동성(추정 σ)</strong> {sig_ann_pct:.2f}% · '
        f'<strong>예측기간 스케일 σ√T</strong> {sig_sqrt_t:.4f}<br>'
        f'<strong>Volatility Drag</strong> Itô 보정 로그드리프트 <strong>μ−½σ²</strong> = {drag_ann:.4f} (연율). '
        f'C++ 엔진은 각 스텝에 <code>(μ−½σ²)Δt</code>를 사용하므로, '
        f'<b>σ가 클수록 확산의 볼록성(convexity) 때문에 기대 로그수익이 깎입니다.</b><br>'
        f'<strong>드리프트 수축</strong> 표본 연율 μ̂와 시장 prior <strong>{MU_MARKET_PRIOR:.0%}</strong>을 '
        f'<strong>{int(MU_SHRINKAGE_SAMPLE_WEIGHT * 100)}:{100 - int(MU_SHRINKAGE_SAMPLE_WEIGHT * 100)}</strong> '
        f'혼합한 뒤 μ≤{GBM_MU_ANNUAL_CAP:.0%}, σ≥{GBM_SIGMA_ANNUAL_FLOOR:.0%} 클램프. '
        f'연율 μ 추정 오차는 ±{result.mu_uncertainty:.0%} 표준편차로 경로별 반영.<br>'
        f'<strong>종료가 로그수익 초과첨도</strong> {xk:+.3f} (확산만이면 ≈0) · '
        f'<strong>Fat-tail 체감 지수</strong> {ft:.3f} · 꼬리 가중 <strong>×{tadj:.3f}</strong> '
        f"(Score = {rsb:.3f} × {tadj:.3f} → <strong>{rs:.3f}</strong>)"
        f"</div>"
    )

    st.markdown(
        f'<section class="risk-hero">'
        f'<div class="risk-hero-head">'
        f'<div>'
        f'<div class="risk-label">Risk Report · {result.ticker}</div>'
        f'<div class="risk-title">{title}</div>'
        f'<p class="top-sub" style="margin-top:12px;">{desc}</p>'
        f'{risk_meta}'
        f'</div>'
        f'<span class="risk-pill {tone}">{result.elapsed:.2f}s · {result.n_paths:,} paths · Score {rs:.2f}</span>'
        f'</div>'
        f'<div class="risk-grid">'
        f'<div class="risk-item"><div class="risk-item-label">예상 중앙가</div>'
        f'<div class="risk-item-value {median_cls}">{fp(metrics["p50"])}</div>'
        f'<div class="mc-delta {"d-pos" if median_pct >= 0 else "d-neg"}">{"↑" if median_pct >= 0 else "↓"} {median_pct:+.2f}%</div></div>'
        f'<div class="risk-item"><div class="risk-item-label">90% 예측구간</div>'
        f'<div class="risk-item-value risk-blue" style="font-size:1.25rem">{fp(metrics["p05"])} ~ {fp(metrics["p95"])}</div>'
        f'<div class="mc-delta" style="opacity:0.72">단일 목표가보다 우선 해석</div></div>'
        f'<div class="risk-item"><div class="risk-item-label">상승 확률 (보조)</div>'
        f'<div class="risk-item-value {up_cls}">{up:.1f}%</div></div>'
        f'<div class="risk-item"><div class="risk-item-label">Risk Score (정규화)</div>'
        f'<div class="risk-item-value {rs_cls}">{rs:.3f}</div>'
        f'<div class="mc-delta d-neg" style="opacity:0.85">base {rsb:.3f} × tail {tadj:.3f}</div></div>'
        f'<div class="risk-item"><div class="risk-item-label">VaR 95% 손실</div>'
        f'<div class="risk-item-value risk-red">{fp(metrics["var_95_abs"])}</div>'
        f'<div class="mc-delta d-neg">↓ {metrics["var_95_pct"]:.1f}%</div></div>'
        f'<div class="risk-item"><div class="risk-item-label">CVaR 95% 꼬리손실</div>'
        f'<div class="risk-item-value risk-red">{fp(metrics["cvar_95_abs"])}</div>'
        f'<div class="mc-delta d-neg">↓ {metrics["cvar_95_pct"]:.1f}%</div></div>'
        f'</div>'
        f'</section>',
        unsafe_allow_html=True,
    )


def _render_action_metrics(result: DashboardResult) -> None:
    """Kelly·Sortino·샤프(비교)를 Toss 스타일 메트릭 카드로 표시."""
    p, m = result.params, result.metrics
    kelly = compute_kelly_leverage_fraction(p.mu, p.sigma, RISK_FREE_RATE_ANNUAL)
    sortino, _ = compute_sortino_from_terminal(result.terminal, p.s0, RISK_FREE_RATE_ANNUAL)
    term = np.asarray(result.terminal, dtype=np.float64)
    rlg = np.log(np.maximum(term, np.finfo(np.float64).tiny) / p.s0)
    rf_lg = float(np.log(1.0 + RISK_FREE_RATE_ANNUAL))
    excess = float(np.mean(rlg) - rf_lg) if rlg.size else 0.0
    if rlg.size > 1:
        sharpe_denom = float(np.std(rlg, ddof=1))
        sharpe = excess / max(sharpe_denom, 1e-12)
    else:
        sharpe = 0.0
    up = float(m["up_probability_pct"])

    st.markdown(
        '<div class="stitle">모형 참고 지표<span class="stitle-sub">Kelly · Sortino · Sharpe 대비</span></div>',
        unsafe_allow_html=True,
    )
    if kelly < 0.20:
        st.caption("Kelly 값은 입력 μ, σ 오차에 민감합니다. 실제 비중으로 바로 쓰지 말고 보수적으로 축소해 해석하세요.")

    so_vc = ACCENT
    if sortino > sharpe:
        so_vc = GREEN
    elif sortino < 0.0:
        so_vc = RED
    sk_vc = GREEN if sharpe >= 0.0 else RED

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        k_vc = ACCENT if kelly >= 0.2 else MUTED
        st.markdown(
            _mc(
                f"Kelly 참고 비중 · r={RISK_FREE_RATE_ANNUAL:.0%}",
                f"{kelly * 100:.1f}%",
                f'<div class="mc-delta" style="opacity:0.65">f = (μ−r)/σ², 실제 운용 전 축소 필요</div>',
                lg=True,
                vc=k_vc,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _mc(
                "소르티노 (하방 σ)",
                f"{sortino:.3f}",
                f'<div class="mc-delta" style="opacity:0.85">전체 σ 샤프 {sharpe:.3f} — 하방만 분모</div>',
                lg=True,
                vc=so_vc,
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _mc(
                "샤프 (전체 σ, 참고)",
                f"{sharpe:.3f}",
                f'<div class="mc-delta" style="opacity:0.65">Sortino와 동일 분자·전체 변동성</div>',
                lg=True,
                vc=sk_vc,
            ),
            unsafe_allow_html=True,
        )
    with c4:
        up_vc = GREEN if up >= 50.0 else RED
        st.markdown(
            _mc(
                "상승 확률 P(종료가 &gt; S₀)",
                f"{up:.1f}%",
                '<div class="mc-delta" style="opacity:0.65">시뮬 경로 비율 (참고)</div>',
                lg=True,
                vc=up_vc,
            ),
            unsafe_allow_html=True,
        )


def _render_key_metrics(result: DashboardResult) -> None:
    metrics, s0, cur = result.metrics, result.params.s0, result.params.currency
    fp = lambda v: _fp(v, cur)
    st.markdown(
        '<div class="stitle">핵심 지표<span class="stitle-sub">Key metrics</span></div>',
        unsafe_allow_html=True,
    )
    k1, k2, k3 = st.columns(3, gap="large")
    with k1:
        st.markdown(
            _mc("최종 예상가 Median (Sₜ 중앙)", fp(metrics["p50"]), _dhtml(_dpct(metrics["p50"], s0)), lg=True),
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


def _render_charts(result: DashboardResult) -> None:
    fan_fig = result.fan_fig
    if fan_fig is None:
        _skel = st.empty()
        _skel.markdown(
            f'<div style="display:grid;grid-template-columns:1fr;gap:18px;">'
            f'<div class="sq-skeleton sq-chart-skeleton"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        fan_fig = build_fan(result.path_matrix, result.params.s0, result.years, result.params.currency, result.ticker)
        st.session_state["fan_fig"] = fan_fig
        _skel.empty()

    cfg = {"displayModeBar": False, "responsive": True}
    st.plotly_chart(fan_fig, use_container_width=True, config=cfg)


def _render_outlook(result: DashboardResult) -> None:
    metrics, terminal, s0, cur = result.metrics, result.terminal, result.params.s0, result.params.currency
    fp = lambda v: _fp(v, cur)
    st.markdown(
        '<div class="stitle">투자 전망<span class="stitle-sub">Outlook</span></div>',
        unsafe_allow_html=True,
    )
    up = metrics["up_probability_pct"]
    cols_a = st.columns(3, gap="large")
    with cols_a[0]:
        st.markdown(_mc("현재가 Current Price", fp(s0)), unsafe_allow_html=True)
    with cols_a[1]:
        mean_price = float(terminal.mean())
        st.markdown(_mc("평균 예측가 Mean", fp(mean_price), _dhtml(_dpct(mean_price, s0))), unsafe_allow_html=True)
    with cols_a[2]:
        st.markdown(
            _mc("상승 확률 Profit Prob.", f"{up:.1f}%", large=True, vc=GREEN if up >= 50 else RED),
            unsafe_allow_html=True,
        )


def _render_risk_tables(result: DashboardResult) -> None:
    metrics, params, s0, cur = result.metrics, result.params, result.params.s0, result.params.currency
    fp = lambda v: _fp(v, cur)
    st.markdown(
        '<div class="stitle">리스크 시나리오<span class="stitle-sub">Risk scenarios</span></div>',
        unsafe_allow_html=True,
    )
    scn = [
        ("Best (최선)", "95th", fp(metrics["p95"]), _dpct(metrics["p95"], s0), GREEN),
        ("Expected (기대)", "50th", fp(metrics["p50"]), _dpct(metrics["p50"], s0), GREEN if metrics["p50"] >= s0 else RED),
        ("Worst (최악)", "5th", fp(metrics["p05"]), _dpct(metrics["p05"], s0), RED),
    ]
    rows = ""
    for lb, pc, pr, dl, cl in scn:
        a = "↑" if dl >= 0 else "↓"
        rows += (
            f'<tr><td style="font-weight:600">{lb}</td>'
            f'<td style="color:{MUTED}">{pc}</td>'
            f'<td style="font-weight:700">{pr}</td>'
            f'<td style="color:{cl};font-weight:700">{a} {dl:+.2f}%</td></tr>'
        )

    tl, tr = st.columns([1.2, 1], gap="large")
    with tl:
        st.markdown(
            f'<div class="rtbl-wrap toss-card"><table class="rtbl"><thead><tr>'
            f'<th>Scenario (시나리오)</th><th>Percentile (백분위)</th>'
            f'<th>Price (가격)</th><th>Change (변동)</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>',
            unsafe_allow_html=True,
        )
    with tr:
        st.markdown(
            f'<div class="rtbl-wrap toss-card"><table class="rtbl"><thead><tr>'
            f'<th>Parameter (파라미터)</th><th>Value (값)</th></tr></thead><tbody>'
            f'<tr><td>μ Annual Drift</td><td>{params.mu:.6f}</td></tr>'
            f'<tr><td>σ Volatility</td><td>{params.sigma:.6f}</td></tr>'
            f'<tr><td>Jump λ / μ_J / σ_J</td>'
            f'<td>{result.jump_lambda:.4f} / {result.jump_mu:.4f} / {result.jump_sigma:.4f}</td></tr>'
            f'<tr><td>Jump mode</td><td>{JUMP_MODE_LABELS.get(result.jump_mode, result.jump_mode)}</td></tr>'
            f'<tr><td>μ uncertainty</td><td>±{result.mu_uncertainty:.1%} annual std</td></tr>'
            f'<tr><td>90% Confidence</td>'
            f'<td>{fp(metrics["p05"])} — {fp(metrics["p95"])}</td></tr>'
            f'<tr><td>Paths</td><td>{result.n_paths:,}</td></tr>'
            f'<tr><td>Fan steps / 스레드</td>'
            f'<td>{result.n_steps} / {result.n_threads if result.n_threads else "auto"}</td></tr>'
            f'<tr><td>C++ Engine</td><td>{result.elapsed:.3f}s</td></tr>'
            f'</tbody></table></div>',
            unsafe_allow_html=True,
        )


def _render_validation_reference(result: DashboardResult) -> None:
    """Show model confidence notes, inline backtest button, and CLI entry point."""

    st.markdown(
        '<div class="stitle">과거 검증<span class="stitle-sub">Calibration backtest</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disc toss-card">'
        '이 숫자들은 단일 목표가가 아니라 모형 가정 하의 분포입니다. '
        '1천만 번 시뮬레이션해도 입력 μ·σ·점프 가정이 틀리면 결과는 정밀한 착각입니다. '
        '<b>90% Coverage가 90% 근처</b>이고 <b>Mean Realized Pctl이 50% 근처</b>일수록 잘 보정된 모형입니다. '
        '아래 버튼으로 현재 종목의 7년 롤링 백테스트를 대시보드 안에서 바로 확인하세요.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Inline backtest runner ────────────────────────────────────────────
    bt_session_key = f"sq_bt_{result.ticker}"

    if st.button(
        f"과거 검증 실행 · {result.ticker}  (7년 롤링 · 10K paths · ≈20초)",
        key="sq_run_bt",
        type="primary",
    ):
        with st.spinner("롤링 백테스트 계산 중... 잠시 기다려 주세요."):
            st.session_state[bt_session_key] = _run_inline_backtest(result.ticker)

    bt_stored = st.session_state.get(bt_session_key)
    if bt_stored is not None:
        _render_backtest_inline(bt_stored, result.ticker)

    # ── CLI reference ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="stitle" style="margin-top:28px">CLI 전체 실행'
        '<span class="stitle-sub">50K paths · 10년 데이터</span></div>',
        unsafe_allow_html=True,
    )
    st.code(
        f"python python\\backtest.py --ticker {result.ticker} --period 10y --paths 50000",
        language="powershell",
    )
    st.markdown(
        '<div class="rtbl-wrap toss-card"><table class="rtbl"><thead><tr>'
        '<th>확인 항목</th><th>해석</th></tr></thead><tbody>'
        '<tr><td>90% Coverage</td><td>실제 1년 뒤 가격이 p05~p95 구간에 들어온 비율. 90% 근처가 이상적입니다.</td></tr>'
        '<tr><td>Direction Accuracy</td><td>상승확률 50% 기준 방향 예측 적중률. 단독 투자 판단으로 쓰지 않습니다.</td></tr>'
        '<tr><td>Median Abs Error</td><td>중앙 예측가와 실제 가격의 평균 절대 오차율. 낮을수록 좋습니다.</td></tr>'
        '<tr><td>Brier Up</td><td>상승확률의 확률 예측 오차. 낮을수록 보정이 좋습니다.</td></tr>'
        '<tr><td>Mean Realized Pctl</td><td>실제 가격이 예측 분포의 몇 퍼센타일에 해당했는지 평균. 50%가 이상적입니다.</td></tr>'
        '</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _render_dashboard_result(result: DashboardResult) -> None:
    _render_risk_summary(result)
    _render_action_metrics(result)
    with st.expander("💡 상세 지표와 수학 모델 보기", expanded=False):
        chart_tab, risk_tab, validation_tab, model_tab = st.tabs(["차트", "리스크 지표", "과거 검증", "수학 모델"])
        with chart_tab:
            _render_charts(result)
        with risk_tab:
            _render_key_metrics(result)
            _render_outlook(result)
            _render_risk_tables(result)
        with validation_tab:
            _render_validation_reference(result)
        with model_tab:
            st.markdown(
                '<div class="stitle">수학 모델<span class="stitle-sub">Model reference</span></div>',
                unsafe_allow_html=True,
            )
            _render_math_section(
                result.params,
                result.params.currency,
                result.jump_lambda,
                result.jump_mu,
                result.jump_sigma,
            )
            st.markdown(
                '<div class="disc toss-card">'
                '※ 최근 1년 Yahoo Finance 데이터 기반 GBM 몬테카를로 시뮬레이션 결과입니다. '
                '데이터 조회 실패 시 사이드바의 수동 fallback 파라미터를 사용합니다. '
                '실제 투자 판단의 유일한 근거로 사용할 수 없습니다. '
                'Past performance does not guarantee future results.</div>',
                unsafe_allow_html=True,
            )


def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    preset, jump_mode = _render_sidebar()
    config = _render_top_controls(preset, jump_mode)

    if not config.run and "metrics" not in st.session_state:
        _render_landing()
        return

    if config.run and not _run_simulation_flow(config):
        return

    _render_dashboard_result(_result_from_session_state())


if __name__ == "__main__":
    main()
