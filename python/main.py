"""Stellar-Quant — GBM Monte Carlo simulation + risk analysis + visualization."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

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
from loader import PROJECT_ROOT, import_simulator

# ---------------------------------------------------------------------------
# Korean font setup
# ---------------------------------------------------------------------------
_KR_FONTS = ["Malgun Gothic", "NanumGothic", "AppleGothic", "sans-serif"]

def _setup_korean_font() -> None:
    if platform.system() == "Windows":
        mpl.rc("font", family="Malgun Gothic")
    else:
        for f in _KR_FONTS:
            if f in {fe.name for fe in mpl.font_manager.fontManager.ttflist}:
                mpl.rc("font", family=f)
                break
    mpl.rcParams["axes.unicode_minus"] = False

_setup_korean_font()

# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------
MAX_TERMINAL_PATHS = 10_000_000
MAX_FAN_MATRIX_MB  = 400


def estimate_memory_mb(n_terminal: int, n_fan: int, n_steps: int) -> float:
    return (n_terminal * 8 + n_fan * (n_steps + 1) * 8) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Log-normal PDF
# ---------------------------------------------------------------------------
def lognormal_pdf(x: np.ndarray, mu_ln: float, sigma_ln: float) -> np.ndarray:
    if sigma_ln <= 0.0:
        return np.zeros_like(x)
    safe = np.where(x > 0, x, 1.0)
    z = (np.log(safe) - mu_ln) / sigma_ln
    pdf = np.exp(-0.5 * z * z) / (safe * sigma_ln * np.sqrt(2.0 * np.pi))
    return np.where(x > 0, pdf, 0.0)


# ---------------------------------------------------------------------------
# Console UI (한영 병기 + 통화 단위)
# ---------------------------------------------------------------------------
W = 72
SEPARATOR = "=" * W
THIN_SEP  = "-" * W


def _fp(v: float, cur: str) -> str:
    return fmt_price(v, cur)


def print_header(ticker: str, n_paths: int, mem_mb: float, currency: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"   STELLAR-QUANT  |  Monte Carlo GBM Simulator (주가 시뮬레이터)")
    print(SEPARATOR)
    print(f"   Ticker (종목코드)  : {ticker}")
    print(f"   Currency (통화)   : {currency}")
    print(f"   Paths  (시뮬 횟수) : {n_paths:>14,}")
    print(f"   Memory (메모리)    : {mem_mb:>13.1f} MB")
    print(SEPARATOR)


def print_params(params: GbmParams) -> None:
    fp = _fp(params.s0, params.currency)
    print(f"\n  {'Parameter (파라미터)':<38} {'Value':>18}")
    print(f"  {THIN_SEP[:60]}")
    print(f"  {'S₀ Current Price (현재 주가)':<38} {fp:>18}")
    print(f"  {'μ  Annual Drift (연간 기대수익률)':<38} {params.mu:>18.6f}")
    print(f"  {'σ  Annual Volatility (연간 변동성)':<38} {params.sigma:>18.6f}")


def print_risk_report(metrics: dict[str, float], s0: float, terminal: np.ndarray, cur: str) -> None:
    up = metrics["up_probability_pct"]
    arrow = "▲ 상승 UP" if up >= 50.0 else "▼ 하락 DOWN"
    fp = lambda v: _fp(v, cur)

    print(f"\n{SEPARATOR}")
    print(f"   RISK ANALYSIS REPORT (리스크 분석 리포트)")
    print(SEPARATOR)

    print(f"\n  {'Metric (지표)':<44} {'Value':>18}")
    print(f"  {THIN_SEP[:66]}")
    print(f"  {'Mean S_T (평균 예측가)':<44} {fp(terminal.mean()):>18}")
    print(f"  {'Std S_T (표준편차)':<44} {fp(terminal.std(ddof=1)):>18}")
    print(f"  {'Median S_T (중앙 예측가)':<44} {fp(metrics['p50']):>18}")
    print(f"  {THIN_SEP[:66]}")
    print(f"  {'1st Percentile (1% 하한)':<44} {fp(metrics['p01']):>18}")
    print(f"  {'5th Percentile (5% 하한, VaR 기준)':<44} {fp(metrics['p05']):>18}")
    print(f"  {'25th Percentile (25% 하한)':<44} {fp(metrics['p25']):>18}")
    print(f"  {'75th Percentile (75% 상한)':<44} {fp(metrics['p75']):>18}")
    print(f"  {'95th Percentile (95% 상한)':<44} {fp(metrics['p95']):>18}")
    print(f"  {'99th Percentile (99% 상한)':<44} {fp(metrics['p99']):>18}")
    print(f"  {THIN_SEP[:66]}")
    print(f"  {'Probability of Profit (상승 확률)':<44} [{arrow}] {up:.2f}%")
    print(f"  {'VaR(95%) Absolute Loss (최대 예상 손실)':<44} {fp(metrics['var_95_abs']):>18}")
    print(f"  {'VaR(95%) Loss Ratio (최대 예상 손실률)':<44} {metrics['var_95_pct']:>17.2f}%")
    print(f"  {'CVaR(95%) ES Absolute (조건부 꼬리 손실)':<44} {fp(metrics['cvar_95_abs']):>18}")
    print(f"  {'CVaR(95%) ES Ratio (조건부 꼬리 손실률)':<44} {metrics['cvar_95_pct']:>17.2f}%")


def print_performance(elapsed: float, n_paths: int) -> None:
    throughput = n_paths / elapsed if elapsed > 0 else 0
    print(f"\n  {'--- Performance Report (성능 리포트) ---':^66}")
    print(f"  {THIN_SEP[:66]}")
    print(f"  {'C++ Engine Time (엔진 구동 시간)':<44} {elapsed:>17.4f}s")
    print(f"  {'Throughput (초당 경로 처리량)':<44} {throughput:>14,.0f} paths/s")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Visualization (dark theme, bilingual, currency unit)
# ---------------------------------------------------------------------------
STYLE_COLORS = {
    "bg":       "#1C1C2E",
    "panel":    "#252540",
    "text":     "#E0E0E0",
    "grid":     "#3A3A5C",
    "median":   "#FF4C4C",
    "band_out": "#5B9BD5",
    "band_in":  "#7BAFD4",
    "pdf":      "#FF6B6B",
    "hist":     "#5B9BD5",
    "s0":       "#FFD700",
    "var":      "#FF9500",
    "backdrop": "#8888AA",
    "q_line":   "#4CAF50",
}


def _apply_dark_style(fig, axes) -> None:
    fig.patch.set_facecolor(STYLE_COLORS["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(STYLE_COLORS["panel"])
        ax.tick_params(colors=STYLE_COLORS["text"], labelsize=10)
        ax.xaxis.label.set_color(STYLE_COLORS["text"])
        ax.yaxis.label.set_color(STYLE_COLORS["text"])
        ax.title.set_color(STYLE_COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(STYLE_COLORS["grid"])
        ax.grid(True, color=STYLE_COLORS["grid"], alpha=0.3, linewidth=0.5)


def _make_price_formatter(cur: str) -> mticker.FuncFormatter:
    """Return a tick formatter that renders prices with currency symbol + commas."""
    sym = currency_symbol(cur)
    if cur == "KRW":
        return mticker.FuncFormatter(lambda v, _: f"{sym}{v:,.0f}")
    return mticker.FuncFormatter(lambda v, _: f"{sym}{v:,.0f}" if v >= 1000 else f"{sym}{v:,.2f}")


def _month_ticks(years: float):
    """Generate tick positions (in years) and labels like 0, 3M, 6M, 9M, 1Y, ..."""
    total_months = int(round(years * 12))
    positions, labels = [0.0], ["0"]
    m = 3
    while m <= total_months:
        positions.append(m / 12.0)
        if m % 12 == 0:
            labels.append(f"{m // 12}Y")
        else:
            labels.append(f"{m}M")
        m += 3
    if positions[-1] < years - 0.01:
        positions.append(years)
        labels.append(f"{total_months}M")
    return positions, labels


def _build_guide_text(ticker, s0, years, metrics, terminal, currency):
    """Build the interpretation text for the bottom panel."""
    fp = lambda v: fmt_price(v, currency)
    p05, p50, p95 = metrics["p05"], metrics["p50"], metrics["p95"]
    up = metrics["up_probability_pct"]
    var_abs, var_pct = metrics["var_95_abs"], metrics["var_95_pct"]
    mean_st = float(terminal.mean())
    period = f"{years:.0f}Y" if years == int(years) else f"{years}Y"

    if up >= 70:
        tag = "Bullish (강세)"
    elif up >= 50:
        tag = "Moderately Bullish (약간 강세)"
    elif up >= 30:
        tag = "Moderately Bearish (약간 약세)"
    else:
        tag = "Bearish (약세)"

    left = (
        f"[그래프 해석] Graph Interpretation\n"
        f"─────────────────────────────────────\n"
        f"■ 왼쪽 — 최종가 분포 (Histogram)\n"
        f"  파란 막대 = 시뮬레이션 빈도 (높을수록 가능성 ↑)\n"
        f"  빨간 곡선 = 이론적 확률밀도 (로그정규분포)\n"
        f"  검은 점선 = 현재가 {fp(s0)}\n"
        f"  주황 실선 = VaR 5% = {fp(p05)}\n"
        f"  빨간 실선 = Median = {fp(p50)}\n\n"
        f"■ 오른쪽 — 주가 경로 (Fan Chart)\n"
        f"  파란 영역 = 신뢰구간 (넓을수록 불확실성 ↑)\n"
        f"  빨간 실선 = Median 경로 (가장 가능성 높은 흐름)\n"
        f"  초록 점선 = 5%/95% 경계  |  검은 점선 = 현재가\n\n"
        f"[용어] Key Terms\n"
        f"─────────────────────────────────────\n"
        f"  VaR = 최대 예상 손실 (95% 확률로 이 이상 유지)\n"
        f"  Median = 시뮬레이션 정중앙 예측가 (50th)\n"
        f"  신뢰구간 = 해당 확률로 이 범위 안에 위치"
    )

    right = (
        f"[{ticker} 투자 전망] Investment Outlook\n"
        f"─────────────────────────────────────\n"
        f"  현재가          {fp(s0)}\n"
        f"  중앙 예측가     {fp(p50):>14}  ({(p50-s0)/s0*100:+.1f}%)\n"
        f"  평균 예측가     {fp(mean_st):>14}  ({(mean_st-s0)/s0*100:+.1f}%)\n"
        f"  상승 확률       {up:.1f}%\n"
        f"  전망            {tag}\n\n"
        f"[리스크 시나리오] Risk Scenarios\n"
        f"─────────────────────────────────────\n"
        f"  최선 (95th)     {fp(p95):>14}  ({(p95-s0)/s0*100:+.1f}%)\n"
        f"  기대 (50th)     {fp(p50):>14}  ({(p50-s0)/s0*100:+.1f}%)\n"
        f"  최악 (5th)      {fp(p05):>14}  ({(p05-s0)/s0*100:+.1f}%)\n"
        f"  VaR(95%) 손실   {fp(var_abs):>14}  ({var_pct:.1f}%)\n\n"
        f"  ※ 과거 2년 GBM 시뮬레이션 결과이며,\n"
        f"    실제 투자의 유일한 근거로 사용할 수 없습니다."
    )
    return left, right


def plot_results(
    path_matrix: np.ndarray,
    terminal: np.ndarray,
    s0: float,
    years: float,
    metrics: dict[str, float],
    ticker: str,
    currency: str,
    save_path: Path | None = None,
) -> None:
    from matplotlib.gridspec import GridSpec

    C = STYLE_COLORS
    fp = lambda v: fmt_price(v, currency)
    price_fmt = _make_price_formatter(currency)

    fig = plt.figure(figsize=(16, 13))
    fig.canvas.manager.set_window_title(f"[Stellar-Quant] Analysis (분석): {ticker}")
    fig.patch.set_facecolor(C["bg"])

    gs = GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.0, 0.55],
        hspace=0.32, wspace=0.30,
        left=0.06, right=0.94, top=0.95, bottom=0.04,
    )
    hist_ax = fig.add_subplot(gs[0, 0])
    fan_ax  = fig.add_subplot(gs[0, 1])
    guide_l = fig.add_subplot(gs[1, 0])
    guide_r = fig.add_subplot(gs[1, 1])

    _apply_dark_style(fig, [hist_ax, fan_ax])

    # ── Histogram + log-normal PDF ──────────────────────────────────────
    p01, p99 = metrics["p01"], metrics["p99"]
    clipped = terminal[(terminal >= p01) & (terminal <= p99)]

    hist_ax.hist(clipped, bins=180, density=True,
                 color=C["hist"], alpha=0.45, edgecolor="none",
                 label="Distribution (분포)")

    log_p = np.log(terminal[terminal > 0])
    mu_ln, sigma_ln = float(log_p.mean()), float(log_p.std(ddof=1))
    x = np.linspace(p01, p99, 800)
    hist_ax.plot(x, lognormal_pdf(x, mu_ln, sigma_ln),
                 color=C["pdf"], linewidth=2.2,
                 label="Log-normal PDF (로그정규 확률밀도)")

    hist_ax.axvline(s0, color="#000000", linestyle="--", linewidth=2.2,
                    label=f"현재가 (Current) = {fp(s0)}", zorder=8)
    hist_ax.axvline(metrics["p50"], color=C["median"], linestyle="-", linewidth=1.8,
                    label=f"Median (중앙값) = {fp(metrics['p50'])}", zorder=7)

    var_price = metrics["p05"]
    hist_ax.axvline(var_price, color=C["var"], linestyle="-", linewidth=1.8,
                    label=f"VaR 5% = {fp(var_price)}", zorder=7)
    hist_ax.annotate(
        f"VaR 5%: {fp(var_price)}",
        xy=(var_price, hist_ax.get_ylim()[1] * 0.65),
        xytext=(var_price + (p99 - p01) * 0.06, hist_ax.get_ylim()[1] * 0.75),
        fontsize=9, color=C["var"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["var"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc=C["panel"], ec=C["var"], alpha=0.9),
        zorder=12,
    )

    hist_ax.axvline(metrics["p95"], color=C["q_line"], linestyle="--", linewidth=1.2,
                    alpha=0.8, label=f"95th (상한) = {fp(metrics['p95'])}", zorder=6)

    hist_ax.set_xlim(p01 * 0.95, p99 * 1.05)
    hist_ax.xaxis.set_major_formatter(price_fmt)
    hist_ax.tick_params(axis="x", rotation=25)
    hist_ax.set_title(f"Terminal Price Distribution (최종가 분포): {ticker}",
                      fontsize=13, fontweight="bold", pad=12)
    hist_ax.set_xlabel(f"S_T (최종 예측가, {currency})", fontsize=11)
    hist_ax.set_ylabel("Density (확률밀도)", fontsize=11)
    hist_ax.legend(loc="upper right", fontsize=7.5,
                   facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])

    # ── Fan Chart ───────────────────────────────────────────────────────
    n_paths, n_points = path_matrix.shape
    times = np.linspace(0.0, years, n_points)
    q05, q25, q50, q75, q95 = np.quantile(
        path_matrix, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0,
    )

    backdrop = min(1000, n_paths)
    idx = np.random.default_rng(123).choice(n_paths, size=backdrop, replace=False)
    fan_ax.plot(times, path_matrix[idx].T, color=C["backdrop"], alpha=0.015, linewidth=0.4)

    fan_ax.fill_between(times, q05, q95, color=C["band_out"], alpha=0.12,
                        label="5%–95% Confidence (신뢰구간)")
    fan_ax.fill_between(times, q25, q75, color=C["band_in"],  alpha=0.20,
                        label="25%–75% Confidence (신뢰구간)")
    fan_ax.plot(times, q50, color=C["median"], linewidth=2.5,
                label="Median (중간값)", zorder=5)
    fan_ax.plot(times, q05, color=C["q_line"], linewidth=1.0, linestyle="--",
                label="5% (하한)", alpha=0.8)
    fan_ax.plot(times, q95, color=C["q_line"], linewidth=1.0, linestyle="--",
                label="95% (상한)", alpha=0.8)

    fan_ax.axhline(s0, color="#000000", linestyle="--", linewidth=2.2, zorder=8)
    fan_ax.annotate(
        f" 현재가 (Current Price)\n {fp(s0)}",
        xy=(years * 0.02, s0),
        xytext=(years * 0.12, s0 + (q95[-1] - q05[-1]) * 0.12),
        fontsize=9, color="#FFFFFF", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#FFFFFF", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", fc="#000000", alpha=0.7),
        zorder=9,
    )

    median_end = float(q50[-1])
    fan_ax.annotate(f" {fp(median_end)}", xy=(years, median_end),
                    xytext=(years * 1.005, median_end),
                    fontsize=9, color=C["median"], fontweight="bold",
                    verticalalignment="center", zorder=10)
    q05_end, q95_end = float(q05[-1]), float(q95[-1])
    fan_ax.annotate(f" {fp(q05_end)}", xy=(years, q05_end),
                    xytext=(years * 1.005, q05_end),
                    fontsize=8, color=C["q_line"], va="center", zorder=10)
    fan_ax.annotate(f" {fp(q95_end)}", xy=(years, q95_end),
                    xytext=(years * 1.005, q95_end),
                    fontsize=8, color=C["q_line"], va="center", zorder=10)

    y_lo = min(float(q05.min()), s0) * 0.90
    y_hi = float(q95.max()) * 1.10
    fan_ax.set_ylim(y_lo, y_hi)
    fan_ax.yaxis.set_major_formatter(price_fmt)

    xtick_pos, xtick_labels = _month_ticks(years)
    fan_ax.set_xticks(xtick_pos)
    fan_ax.set_xticklabels(xtick_labels)
    fan_ax.set_xlim(-years * 0.02, years * 1.10)

    fan_ax.set_title(f"Stock Price Simulation (주가 시뮬레이션): {ticker}",
                     fontsize=13, fontweight="bold", pad=12)
    fan_ax.set_xlabel("Period (예측 기간)", fontsize=11)
    fan_ax.set_ylabel(f"Price (주가, {currency})", fontsize=11)
    fan_ax.legend(loc="upper left", fontsize=7.5,
                  facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])

    # ── Guide panels (하단 해석 패널) ─────────────────────────────────
    left_text, right_text = _build_guide_text(
        ticker, s0, years, metrics, terminal, currency,
    )

    guide_font = "Malgun Gothic" if platform.system() == "Windows" else None
    for ax, txt in [(guide_l, left_text), (guide_r, right_text)]:
        ax.set_facecolor(C["panel"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C["grid"])
        kw = dict(
            transform=ax.transAxes, fontsize=9, color=C["text"],
            verticalalignment="top", horizontalalignment="left",
            linespacing=1.4,
        )
        if guide_font:
            kw["fontfamily"] = guide_font
        ax.text(0.05, 0.94, txt, **kw)

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"  Plot saved (저장 완료) → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Interpretation (터미널 투자 전망 요약)
# ---------------------------------------------------------------------------
def print_interpretation(
    ticker: str, params: GbmParams, years: float,
    metrics: dict[str, float], terminal: np.ndarray,
) -> None:
    fp = lambda v: _fp(v, params.currency)
    s0 = params.s0
    p05, p50, p95 = metrics["p05"], metrics["p50"], metrics["p95"]
    up = metrics["up_probability_pct"]
    var_abs = metrics["var_95_abs"]
    var_pct = metrics["var_95_pct"]
    mean_st = float(terminal.mean())

    if up >= 70:
        outlook = "강세 (Bullish)"
    elif up >= 50:
        outlook = "약간 강세 (Moderately Bullish)"
    elif up >= 30:
        outlook = "약간 약세 (Moderately Bearish)"
    else:
        outlook = "약세 (Bearish)"

    period = f"{years:.0f}년" if years == int(years) else f"{years}년"

    print(f"\n{SEPARATOR}")
    print(f"   INVESTMENT OUTLOOK (투자 전망) — {ticker} ({period})")
    print(SEPARATOR)
    print(f"\n  {'항목':<22} {'가격':>16} {'변동':>10}")
    print(f"  {THIN_SEP[:52]}")
    print(f"  {'현재가 (Current)':<22} {fp(s0):>16}")
    print(f"  {'중앙 예측가 (Median)':<22} {fp(p50):>16} {(p50-s0)/s0*100:>+9.1f}%")
    print(f"  {'평균 예측가 (Mean)':<22} {fp(mean_st):>16} {(mean_st-s0)/s0*100:>+9.1f}%")
    print(f"  {THIN_SEP[:52]}")
    print(f"  {'최선 (95th)':<22} {fp(p95):>16} {(p95-s0)/s0*100:>+9.1f}%")
    print(f"  {'기대 (50th)':<22} {fp(p50):>16} {(p50-s0)/s0*100:>+9.1f}%")
    print(f"  {'최악 (5th)':<22} {fp(p05):>16} {(p05-s0)/s0*100:>+9.1f}%")
    print(f"  {THIN_SEP[:52]}")
    print(f"  {'상승 확률':<22} {up:>15.1f}%")
    print(f"  {'VaR(95%) 손실':<22} {fp(var_abs):>16}  ({var_pct:.1f}%)")
    print(f"  {'CVaR(95%) ES':<22} {fp(metrics['cvar_95_abs']):>16}  ({metrics['cvar_95_pct']:.1f}%)")
    print(f"  {'전망 (Outlook)':<22} {outlook:>16}")
    print(f"\n  ※ 그래프 하단에 상세 해석 가이드가 포함되어 있습니다.")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Summary report (한영 병기 + 통화 단위 + Note 섹션)
# ---------------------------------------------------------------------------
def write_summary(
    ticker: str,
    params: GbmParams,
    years: float,
    n_paths: int,
    n_steps: int,
    terminal: np.ndarray,
    metrics: dict[str, float],
    elapsed_sec: float,
    *,
    jump_lambda: float,
    jump_mu: float,
    jump_sigma: float,
    jump_diffusion_enabled: bool,
) -> Path:
    bench_path = PROJECT_ROOT / "python" / "benchmark_results.json"
    benchmark = json.loads(bench_path.read_text("utf-8")) if bench_path.exists() else None
    throughput = n_paths / elapsed_sec if elapsed_sec > 0 else 0

    fp = lambda v: fmt_price(v, params.currency)
    cur = params.currency

    lines = [
        "# Simulation Summary (시뮬레이션 요약)\n",
        "## Configuration (실행 설정)",
        "| Parameter (항목) | Value (값) |",
        "|:--|--:|",
        f"| Ticker (종목코드) | `{ticker}` |",
        f"| Currency (통화) | `{cur}` |",
        f"| Paths (시뮬레이션 횟수) | `{n_paths:,}` |",
        f"| Steps (시점 수) | `{n_steps}` |",
        f"| Horizon (예측 기간) | `{years}` year(s) |",
        f"| S₀ Current Price (현재 주가) | `{fp(params.s0)}` |",
        f"| μ Annual Drift (연간 기대수익률) | `{params.mu:.6f}` |",
        f"| σ Annual Volatility (연간 변동성) | `{params.sigma:.6f}` |",
        "| Merton Jump Diffusion | "
        f"`{'enabled' if jump_diffusion_enabled else 'disabled (λ = 0)'}` |",
        f"| Jump λ (annual intensity) | `{jump_lambda:.6f}` |",
        f"| Jump μ_J (mean log jump) | `{jump_mu:.6f}` |",
        f"| Jump σ_J (log jump std) | `{jump_sigma:.6f}` |",
        f"| Execution Time (실행 시간) | `{elapsed_sec:.3f}` s |",
        f"| Throughput (초당 처리량) | `{throughput:,.0f}` paths/s |",
        "",
        "## Risk Metrics (리스크 지표)",
        "| Metric (지표) | Value (값) |",
        "|:--|--:|",
        f"| Mean S_T (평균 예측가) | `{fp(terminal.mean())}` |",
        f"| Std S_T (표준편차) | `{fp(terminal.std(ddof=1))}` |",
        f"| Median S_T (중앙 예측가) | `{fp(metrics['p50'])}` |",
        f"| Probability of Profit (상승 확률) | `{metrics['up_probability_pct']:.2f}%` |",
        f"| 90% Confidence Interval (신뢰구간 5%–95%) | `[{fp(metrics['p05'])} — {fp(metrics['p95'])}]` |",
        f"| VaR(95%) Absolute Loss (최대 예상 손실) | `{fp(metrics['var_95_abs'])}` |",
        f"| VaR(95%) Loss Ratio (최대 예상 손실률) | `{metrics['var_95_pct']:.2f}%` |",
        f"| CVaR(95%) ES Absolute (조건부 꼬리 손실) | `{fp(metrics['cvar_95_abs'])}` |",
        f"| CVaR(95%) ES Ratio (조건부 꼬리 손실률) | `{metrics['cvar_95_pct']:.2f}%` |",
        "",
    ]

    if benchmark:
        lines.append("## Performance Benchmark (성능 벤치마크 — C++ vs Python)")
        lines.append("| Simulations (횟수) | C++ (s) | NumPy (s) | Speed-up (배수) |")
        lines.append("|---:|---:|---:|---:|")
        for c, cpp, npy, sp in zip(
            benchmark["counts"],
            benchmark["cpp_seconds"],
            benchmark["numpy_seconds"],
            benchmark["speedup_cpp_vs_numpy"],
        ):
            lines.append(f"| {c:,} | {cpp:.6f} | {npy:.6f} | {sp:.2f}× |")
        lines.append("")

    lines.extend([
        "---",
        "",
        f"> **[Note]** {n_paths:,} paths simulated via C++ multi-threaded engine "
        f"({('Merton jump-diffusion + antithetic variates' if jump_diffusion_enabled else 'GBM baseline + antithetic variates')}) "
        f"in {elapsed_sec:.3f}s ({throughput:,.0f} paths/s). "
        f"All prices in **{cur}**. "
        f"GBM and jump parameters estimated from 2-year Yahoo Finance historical data.",
    ])

    out = PROJECT_ROOT / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Stellar-Quant GBM Monte Carlo Simulator")
    p.add_argument("--ticker",    default="AAPL")
    p.add_argument("--paths",     type=int, default=10_000_000, help="Terminal simulation count")
    p.add_argument("--steps",     type=int, default=252,        help="Time steps for fan chart")
    p.add_argument("--years",     type=float, default=1.0)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--threads",   type=int, default=0,          help="0 = auto-detect")
    p.add_argument("--fan-paths", type=int, default=10_000,     help="Path count for fan chart")
    p.add_argument("--build-dir", type=str, default=None)
    p.add_argument("--no-plot",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Memory guard ---
    fan_paths = max(100, min(args.fan_paths, args.paths))
    fan_mem_mb = (fan_paths * (args.steps + 1) * 8) / (1024 * 1024)
    if fan_mem_mb > MAX_FAN_MATRIX_MB:
        fan_paths = int(MAX_FAN_MATRIX_MB * 1024 * 1024 / ((args.steps + 1) * 8))
        print(f"[WARN] fan-paths clamped to {fan_paths:,} to stay under {MAX_FAN_MATRIX_MB} MB")
    if args.paths > MAX_TERMINAL_PATHS:
        args.paths = MAX_TERMINAL_PATHS
        print(f"[WARN] paths clamped to {args.paths:,}")

    mem = estimate_memory_mb(args.paths, fan_paths, args.steps)

    # --- Data ---
    try:
        close = fetch_prices(args.ticker, period="2y")
    except YahooFinanceFetchError as e:
        print(str(e), file=sys.stderr)
        print("잠시 후 다시 시도해 주세요.", file=sys.stderr)
        raise SystemExit(2) from None
    params = estimate_gbm_params(close, ticker=args.ticker)
    jump_p = estimate_jump_params(close)

    print_header(args.ticker, args.paths, mem, params.currency)
    print_params(params)

    # --- Simulation ---
    simulator = import_simulator(args.build_dir)
    t0 = time.perf_counter()

    jl, jm, js = jump_p.lambda_annual, jump_p.mu_jump, jump_p.sigma_jump

    terminal = np.asarray(simulator.simulate_gbm_paths(
        n_paths=args.paths,
        s0=params.s0,
        mu=params.mu,
        sigma=params.sigma,
        t=args.years,
        seed=args.seed,
        n_threads=args.threads,
        jump_lambda=jl,
        jump_mu=jm,
        jump_sigma=js,
    ), dtype=np.float64)

    path_matrix = np.asarray(simulator.simulate_gbm_path_matrix(
        n_paths=fan_paths,
        n_steps=args.steps,
        s0=params.s0,
        mu=params.mu,
        sigma=params.sigma,
        t=args.years,
        seed=args.seed + 1,
        n_threads=args.threads,
        jump_lambda=jl,
        jump_mu=jm,
        jump_sigma=js,
    ), dtype=np.float64)

    elapsed = time.perf_counter() - t0

    # --- Analysis ---
    metrics = compute_risk_metrics(terminal, params.s0)
    print_risk_report(metrics, params.s0, terminal, params.currency)
    print_performance(elapsed, args.paths)
    print_interpretation(args.ticker, params, args.years, metrics, terminal)

    # --- Report ---
    summary = write_summary(
        ticker=args.ticker,
        params=params,
        years=args.years,
        n_paths=args.paths,
        n_steps=args.steps,
        terminal=terminal,
        metrics=metrics,
        elapsed_sec=elapsed,
        jump_lambda=jl,
        jump_mu=jm,
        jump_sigma=js,
        jump_diffusion_enabled=(jl > 0.0),
    )
    print(f"  Summary (요약 저장) → {summary}\n")

    # --- Plot ---
    if not args.no_plot:
        save_path = PROJECT_ROOT / "python" / "simulation_plot.png"
        plot_results(path_matrix, terminal, params.s0, args.years,
                     metrics, args.ticker, params.currency, save_path)


if __name__ == "__main__":
    main()
