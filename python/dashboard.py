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
    MU_MARKET_PRIOR,
    clamp_gbm_for_simulation,
    compute_risk_metrics,
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
    JUMP_MODE_CONSERVATIVE: "위기 상황 가정 (Crisis)",
    JUMP_MODE_HISTORICAL: "과거의 반복 (History)",
    JUMP_MODE_OFF: "평온한 시장 (Smooth)",
}
JUMP_MODE_CAPTIONS = {
    JUMP_MODE_CONSERVATIVE: "최악의 시나리오를 가정합니다. 하방 충격을 고정 적용해 리스크를 보수적으로 평가합니다.",
    JUMP_MODE_HISTORICAL: "과거 데이터에서 탐지한 실제 급락 패턴을 반영합니다.",
    JUMP_MODE_OFF: "순수한 GBM 확산만 적용합니다. 갑작스러운 시장 충격이 없다고 가정합니다.",
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


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_exchange_rate() -> float:
    """USD/KRW 환율 (1 USD = ? KRW). 실패 시 기본값 1350 반환."""
    try:
        rate_series = fetch_prices("KRW=X", period="5d")
        if not rate_series.empty:
            return float(rate_series.iloc[-1])
    except Exception:
        pass
    return 1350.0


def _on_krw_change() -> None:
    """KRW 입력 변경 시 USD를 실시간 환산."""
    rate = _fetch_exchange_rate()
    krw = float(st.session_state.get("sq_inv_krw", 0.0))
    st.session_state["sq_inv_usd"] = round(krw / rate, 2)


def _on_usd_change() -> None:
    """USD 입력 변경 시 KRW를 실시간 환산."""
    rate = _fetch_exchange_rate()
    usd = float(st.session_state.get("sq_inv_usd", 0.0))
    st.session_state["sq_inv_krw"] = round(usd * rate, 0)


def _build_portfolio_fan(
    path_matrix: np.ndarray,
    s0: float,
    inv_amt: float,
    years: float,
    currency: str,
) -> go.Figure:
    """투자금 기준 포트폴리오 팬 차트 — 주식 사이트 스타일."""
    cur_sym = "₩" if currency == "KRW" else "$"
    scale = inv_amt / s0
    pm = path_matrix * scale
    n_steps = pm.shape[1]

    # ~80 pts 다운샘플
    step = max(1, n_steps // 80)
    idx = list(range(0, n_steps, step))
    if idx[-1] != n_steps - 1:
        idx.append(n_steps - 1)
    pm_ds = pm[:, idx]
    t = np.linspace(0.0, years * 12.0, len(idx))

    q05 = np.quantile(pm_ds, 0.05, axis=0)
    q25 = np.quantile(pm_ds, 0.25, axis=0)
    q50 = np.quantile(pm_ds, 0.50, axis=0)
    q75 = np.quantile(pm_ds, 0.75, axis=0)
    q95 = np.quantile(pm_ds, 0.95, axis=0)

    y_lo = min(float(q05.min()), inv_amt) * 0.90
    y_hi = max(float(q95.max()), inv_amt) * 1.08

    ret50 = (q50 - inv_amt) / inv_amt * 100.0
    cd50 = np.column_stack([ret50, t])

    fig = go.Figure()

    # 90% 채움 — hover 없음, legend만
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]).tolist(),
        y=np.concatenate([q95, q05[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(0,100,255,0.07)",
        line=dict(width=0), name="이례적 변동 범위", hoverinfo="skip",
    ))
    # 50% 채움
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]).tolist(),
        y=np.concatenate([q75, q25[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(0,100,255,0.22)",
        line=dict(width=0), name="현실적 기대 범위", hoverinfo="skip",
    ))
    # P95 테두리 — hover 없음
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=q95.tolist(), mode="lines",
        line=dict(color="rgba(0,100,255,0.35)", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    # P05 테두리 — hover 없음
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=q05.tolist(), mode="lines",
        line=dict(color="rgba(255,75,75,0.35)", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    # 투자 원금 기준선
    fig.add_trace(go.Scatter(
        x=[float(t[0]), float(t[-1])], y=[inv_amt, inv_amt],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dot"),
        name="투자 원금", hoverinfo="skip",
    ))
    # 중앙값 — 유일한 hover 대상
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=q50.tolist(), mode="lines",
        line=dict(color="#0064FF", width=3),
        name="가장 확률 높은 미래",
        customdata=cd50,
        hovertemplate=(
            f"<b>%{{customdata[1]:.0f}}개월 뒤</b>  "
            f"예상 자산 <b>{cur_sym}%{{y:,.0f}}</b>"
            f"  (%{{customdata[0]:+.1f}}%)"
            "<extra></extra>"
        ),
    ))

    # 말풍선 annotation — 1년 뒤 중앙값
    final_median = float(q50[-1])
    final_ret = float(ret50[-1])
    ret_sign = "+" if final_ret >= 0 else ""
    ann_color = "#0064FF" if final_ret >= 0 else "#FF4B4B"

    # X 눈금: 2개월 간격
    total_months = int(round(years * 12))
    x_ticks = list(range(0, total_months + 1, 2))
    if x_ticks[-1] != total_months:
        x_ticks.append(total_months)

    fig.update_layout(
        paper_bgcolor="#101012",
        plot_bgcolor="#101012",
        height=560,
        autosize=False,
        dragmode=False,
        margin=dict(l=48, r=220, t=56, b=100),
        font=dict(
            family="Pretendard Variable, Pretendard, -apple-system, sans-serif",
            color="#F4F5F7", size=12,
        ),
        shapes=[
            dict(type="rect", xref="paper", yref="y",
                 x0=0, x1=1, y0=y_lo, y1=inv_amt,
                 fillcolor="rgba(255,75,75,0.12)", line_width=0, layer="below"),
            dict(type="rect", xref="paper", yref="y",
                 x0=0, x1=1, y0=inv_amt, y1=y_hi,
                 fillcolor="rgba(0,100,255,0.12)", line_width=0, layer="below"),
        ],
        xaxis=dict(
            range=[0, total_months],
            tickvals=x_ticks,
            ticktext=[f"{m}M" for m in x_ticks],
            tickfont=dict(size=11, color="#5C6573"),
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            fixedrange=True,
            showspikes=False,
        ),
        yaxis=dict(
            range=[y_lo, y_hi],
            tickprefix=cur_sym, tickformat="~s",
            tickfont=dict(size=11, color="#5C6573"),
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            fixedrange=True,
            showspikes=False,
        ),
        legend=dict(
            bgcolor="rgba(10,10,12,0.80)",
            bordercolor="rgba(255,255,255,0.07)",
            borderwidth=1,
            font=dict(size=11, color="#8B93A1"),
            orientation="v",
            xanchor="left", x=0.01,
            yanchor="top", y=0.97,
            itemsizing="constant",
            tracegroupgap=1,
        ),
        hovermode=False,
        annotations=[
            # ── 왼쪽 시작점: 현재 원금 ─────────────────────────────────
            dict(
                x=float(t[0]), y=inv_amt,
                xanchor="left", yanchor="bottom",
                text=f"<b>{cur_sym}{inv_amt:,.0f}</b>",
                font=dict(color="rgba(255,255,255,0.55)", size=11,
                          family="Pretendard Variable, sans-serif"),
                showarrow=False,
                yshift=8,
            ),
            # ── 오른쪽 끝: 최상 시나리오 (P95) ────────────────────────
            dict(
                x=float(t[-1]), y=float(q95[-1]),
                xanchor="left", yanchor="middle",
                text=f"{cur_sym}{float(q95[-1]):,.0f}",
                font=dict(color="rgba(80,160,255,0.80)", size=10,
                          family="Pretendard Variable, sans-serif"),
                showarrow=False,
                xshift=8,
            ),
            # ── 오른쪽 끝: 최악 시나리오 (P05) ────────────────────────
            dict(
                x=float(t[-1]), y=float(q05[-1]),
                xanchor="left", yanchor="middle",
                text=f"{cur_sym}{float(q05[-1]):,.0f}",
                font=dict(color="rgba(255,100,100,0.80)", size=10,
                          family="Pretendard Variable, sans-serif"),
                showarrow=False,
                xshift=8,
            ),
            # ── 오른쪽 끝: 중앙값 말풍선 ──────────────────────────────
            dict(
                x=float(t[-1]), y=final_median,
                xanchor="left", yanchor="middle",
                text=f"<b>{cur_sym}{final_median:,.0f}</b><br>{ret_sign}{final_ret:.1f}%",
                font=dict(color="white", size=13,
                          family="Pretendard Variable, sans-serif"),
                bgcolor=ann_color,
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
                borderpad=9,
                showarrow=True,
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=ann_color,
                ax=34, ay=0,
            ),
        ],
    )
    return fig


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
            "시장 상황 가정",
            options=[JUMP_MODE_CONSERVATIVE, JUMP_MODE_HISTORICAL, JUMP_MODE_OFF],
            format_func=lambda s: JUMP_MODE_LABELS[s],
            index=0,
            key="sq_jump_mode",
        )
        st.caption(JUMP_MODE_CAPTIONS.get(jump_mode, ""))

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

    # ── KRW / USD 투자금 입력 (시뮬 전 설정) ─────────────────────────────
    rate = _fetch_exchange_rate()
    if "sq_inv_krw" not in st.session_state:
        st.session_state["sq_inv_krw"] = 1_000_000.0
    if "sq_inv_usd" not in st.session_state:
        st.session_state["sq_inv_usd"] = 1_000.0
    tab_krw, tab_usd = st.tabs(["KRW", "USD"])
    with tab_krw:
        st.number_input(
            "투자 예정 금액 (원)",
            min_value=0.0,
            step=100_000.0,
            format=",.0f",
            key="sq_inv_krw",
            on_change=_on_krw_change,
        )
        krw_val = float(st.session_state["sq_inv_krw"])
        st.caption(f"≈ ${krw_val / rate:,.0f} USD  ·  환율 {rate:,.0f} KRW/USD 기준")
    with tab_usd:
        st.number_input(
            "투자 예정 금액 (USD)",
            min_value=0.0,
            step=100.0,
            format=",.2f",
            key="sq_inv_usd",
            on_change=_on_usd_change,
        )
        usd_val = float(st.session_state["sq_inv_usd"])
        st.caption(f"≈ ₩{usd_val * rate:,.0f} KRW  ·  환율 {rate:,.0f} KRW/USD 기준")

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

    # EWMA 가중 변동성 (λ=0.94, RiskMetrics 방식) — 최근 시장 충격에 더 민감하게 반응
    if len(close) >= 31:
        log_ret = np.log(close.values[1:] / close.values[:-1]).astype(np.float64)
        n = len(log_ret)
        decay = 0.94
        weights = np.array([decay ** i for i in range(n - 1, -1, -1)], dtype=np.float64)
        weights /= weights.sum()
        ewma_sigma = float(np.sqrt(float(np.sum(weights * log_ret ** 2)) * 252))
        # 역사적 vol(40%)와 EWMA vol(60%) 혼합, 역사적 sigma의 70~150% 범위로 제한
        blended = float(np.clip(
            0.6 * ewma_sigma + 0.4 * params.sigma,
            params.sigma * 0.70, params.sigma * 1.50,
        ))
        params = replace(params, sigma=blended)

    return MarketData(
        params=params,
        jump=estimate_jump_params(close),
        mu_uncertainty=mu_uncertainty,
        source="yfinance",
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
            "투자 성향 및 예측 시나리오: 매우 고위험",
            "risk-pill-red",
            f"시장 충격·꼬리 손실 위험이 매우 큽니다. 원금 손실 가능성이 높습니다. (Risk Score {rs:.2f})",
        )
    if rs >= 2.5:
        return (
            "투자 성향 및 예측 시나리오: 고위험",
            "risk-pill-red",
            f"급락 이벤트 발생 시 큰 손실이 예상됩니다. 분산 투자를 권장합니다. (Score {rs:.2f})",
        )
    if rs >= 1.5:
        return (
            "투자 성향 및 예측 시나리오: 중간 위험",
            "risk-pill-blue",
            f"평균적인 주식 시장 수준의 변동성입니다. (Score {rs:.2f})",
        )
    return (
        "투자 성향 및 예측 시나리오: 낮은 위험",
        "risk-pill-blue",
        f"상대적으로 안정적인 시나리오입니다. (Score {rs:.2f})",
    )


def _render_risk_summary(result: DashboardResult) -> None:
    metrics, params = result.metrics, result.params
    fp = lambda v: _fp(v, params.currency)
    up = metrics["up_probability_pct"]

    if up > 90.0:
        st.warning(
            "상승 확률 90% 초과 — 최근 급등 추세가 과도하게 반영됐을 수 있습니다. 보수적으로 해석하세요."
        )

    title, tone, _ = _risk_summary(metrics)
    median_pct = _dpct(metrics["p50"], params.s0)
    median_cls = "risk-blue" if median_pct >= 0 else "risk-red"
    median_delta_cls = "d-pos" if median_pct >= 0 else "d-neg"
    median_arrow = "↑" if median_pct >= 0 else "↓"
    up_cls = "risk-blue" if up >= 50 else "risk-red"

    st.markdown(
        f'<section class="risk-hero">'
        f'<div class="risk-hero-head">'
        f'<div>'
        f'<div class="risk-label">Risk Report · {result.ticker}</div>'
        f'<div class="risk-title">{title}</div>'
        f'<div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">'
        f'<span class="risk-pill {tone}">상승 확률 &nbsp;:&nbsp; <b class="{up_cls}" style="font-size:1.15em;font-weight:800;letter-spacing:-0.01em">{up:.1f} %</b></span>'
        f'<span style="color:{MUTED};font-size:0.75rem;opacity:0.5">'
        f'{result.elapsed:.2f}s · {result.n_paths:,} paths</span>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div class="risk-grid" style="grid-template-columns:repeat(4,minmax(0,1fr))">'
        f'<div class="risk-item"><div class="risk-item-label">90% 예측구간</div>'
        f'<div class="risk-item-value risk-blue" style="font-size:1.15rem">'
        f'{fp(metrics["p05"])} ~ {fp(metrics["p95"])}</div>'
        f'<div class="mc-delta" style="opacity:0.65">단일 목표가보다 우선 해석</div></div>'
        f'<div class="risk-item"><div class="risk-item-label">예상 중앙가 (50th)</div>'
        f'<div class="risk-item-value {median_cls}">{fp(metrics["p50"])}</div>'
        f'<div class="mc-delta {median_delta_cls}">{median_arrow} {median_pct:+.2f}%</div></div>'
        f'<div class="risk-item"><div class="risk-item-label">95% 확률 하한선</div>'
        f'<div class="risk-item-value risk-red">{fp(metrics["var_95_abs"])}</div>'
        f'<div class="mc-delta d-neg">↓ {metrics["var_95_pct"]:.1f}% · 95% 확률로 이 가격 이상 유지</div></div>'
        f'<div class="risk-item"><div class="risk-item-label">시장 폭락 시 예상 손실</div>'
        f'<div class="risk-item-value risk-red">{fp(metrics["cvar_95_abs"])}</div>'
        f'<div class="mc-delta d-neg">↓ {metrics["cvar_95_pct"]:.1f}% · 최악 5% 상황 평균 손실</div></div>'
        f'</div>'
        f'</section>',
        unsafe_allow_html=True,
    )


def _render_investment_section(result: DashboardResult) -> None:
    """포트폴리오 팬 차트 + CSS 카드형 투자 지표 + 공포 지수."""
    s0 = result.params.s0
    cur = result.params.currency
    terminal = result.terminal
    metrics = result.metrics

    # 투자금은 _render_top_controls에서 설정한 session_state 값을 읽음
    inv_krw = float(st.session_state.get("sq_inv_krw", 1_000_000.0))
    inv_usd = float(st.session_state.get("sq_inv_usd", 1_000.0))
    inv_amt = inv_krw if cur == "KRW" else inv_usd
    if inv_amt <= 0.0 or s0 <= 0.0:
        return

    shares = inv_amt / s0
    mean_price = float(np.mean(terminal))
    expected_value = shares * mean_price
    ret_pct = (mean_price - s0) / s0 * 100.0
    ret_cls = "risk-blue" if ret_pct >= 0 else "risk-red"
    ret_delta_cls = "d-pos" if ret_pct >= 0 else "d-neg"
    ret_arrow = "↑" if ret_pct >= 0 else "↓"
    cvar_pct = metrics["cvar_95_pct"]
    cvar_loss = inv_amt * cvar_pct / 100.0

    st.markdown(
        '<div class="stitle" style="margin-top:28px">투자금 시뮬레이터'
        '<span class="stitle-sub">투자 원금 기준 자산가치 예측</span></div>',
        unsafe_allow_html=True,
    )

    # ── 포트폴리오 팬 차트 ────────────────────────────────────────────────
    fan_fig = _build_portfolio_fan(result.path_matrix, s0, inv_amt, result.years, cur)
    st.plotly_chart(
        fan_fig,
        use_container_width=True,
        config={"staticPlot": True},
    )

    # ── 포트폴리오 메트릭 카드 ────────────────────────────────────────────
    st.markdown(
        f'<div class="risk-grid" style="grid-template-columns:repeat(3,minmax(0,1fr));margin-top:8px">'
        f'<div class="risk-item">'
        f'<div class="risk-item-label">예상 수익률 (1Y 평균)</div>'
        f'<div class="risk-item-value {ret_cls}">{ret_pct:+.1f}%</div>'
        f'<div class="mc-delta {ret_delta_cls}">{ret_arrow} 평균 경로 기준</div>'
        f'</div>'
        f'<div class="risk-item">'
        f'<div class="risk-item-label">예상 최종 자산</div>'
        f'<div class="risk-item-value {ret_cls}">{_fp(expected_value, cur)}</div>'
        f'<div class="mc-delta {ret_delta_cls}">{ret_arrow} {ret_pct:+.1f}% · 투자금 {_fp(inv_amt, cur)} 기준</div>'
        f'</div>'
        f'<div class="risk-item">'
        f'<div class="risk-item-label">시장 폭락 시 예상 손실</div>'
        f'<div class="risk-item-value risk-red">{_fp(cvar_loss, cur)}</div>'
        f'<div class="mc-delta d-neg">↓ {cvar_pct:.1f}% · 최악 5% 상황 평균 손실</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 공포 지수 ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="stitle" style="margin-top:24px">리스크 공포 지수'
        '<span class="stitle-sub">피부로 느끼는 리스크</span></div>',
        unsafe_allow_html=True,
    )
    halved_prob = float(np.mean(terminal < 0.5 * s0)) * 100.0
    loss_prob = float(np.mean(terminal < s0)) * 100.0
    double_prob = float(np.mean(terminal > 2.0 * s0)) * 100.0
    q10_price = float(np.quantile(terminal, 0.10))
    q10_loss_pct = (q10_price - s0) / s0 * 100.0

    halved_cls = "risk-red" if halved_prob > 5.0 else "risk-blue"
    loss_cls = "risk-red" if loss_prob > 40.0 else "risk-blue"
    q10_cls = "risk-red" if q10_loss_pct < -15.0 else "risk-blue"

    st.markdown(
        f'<div class="risk-grid" style="grid-template-columns:repeat(4,minmax(0,1fr));margin-top:8px">'
        f'<div class="risk-item">'
        f'<div class="risk-item-label">반토막(-50%) 확률</div>'
        f'<div class="risk-item-value {halved_cls}">{halved_prob:.1f}%</div>'
        f'<div class="mc-delta d-neg">1년 뒤 {_fp(s0 * 0.5, cur)} 이하</div>'
        f'</div>'
        f'<div class="risk-item">'
        f'<div class="risk-item-label">원금 손실 확률</div>'
        f'<div class="risk-item-value {loss_cls}">{loss_prob:.1f}%</div>'
        f'<div class="mc-delta d-neg">현재가 {_fp(s0, cur)} 미만</div>'
        f'</div>'
        f'<div class="risk-item">'
        f'<div class="risk-item-label">2배 달성 확률</div>'
        f'<div class="risk-item-value risk-blue">{double_prob:.1f}%</div>'
        f'<div class="mc-delta d-pos">목표가 {_fp(s0 * 2.0, cur)}</div>'
        f'</div>'
        f'<div class="risk-item">'
        f'<div class="risk-item-label">하위 10% 시나리오</div>'
        f'<div class="risk-item-value {q10_cls}">{q10_loss_pct:+.1f}%</div>'
        f'<div class="mc-delta d-neg">↓ 주가 {_fp(q10_price, cur)} 예상</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"현재 주가 {_fp(s0, cur)} 기준. 투자금 {_fp(inv_amt, cur)}으로 반토막 시 "
        f"예상 손실 ≈ {_fp(inv_amt * 0.5, cur)}"
    )


def _render_dashboard_result(result: DashboardResult) -> None:
    _render_risk_summary(result)
    _render_investment_section(result)


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
