"""Yahoo Finance data fetching and GBM parameter estimation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True, slots=True)
class GbmParams:
    s0: float
    mu: float      # annualized drift
    sigma: float   # annualized volatility
    currency: str  # "USD", "KRW", etc.


@dataclass(frozen=True, slots=True)
class JumpParams:
    """Merton-style log jump sizes J ~ N(mu_jump, sigma_jump^2); λ = jumps per year."""

    lambda_annual: float  # Poisson intensity (per trading year)
    mu_jump: float        # mean log jump increment
    sigma_jump: float     # std dev of log jump increment


def detect_currency(ticker: str) -> str:
    """Determine currency symbol from ticker suffix."""
    t = ticker.upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "KRW"
    if t.endswith(".T"):
        return "JPY"
    if t.endswith(".L"):
        return "GBP"
    if t.endswith(".DE") or t.endswith(".PA"):
        return "EUR"
    if t.endswith(".HK"):
        return "HKD"
    if t.endswith(".SS") or t.endswith(".SZ"):
        return "CNY"
    return "USD"


CURRENCY_SYMBOLS = {
    "USD": "$",
    "KRW": "₩",
    "JPY": "¥",
    "GBP": "£",
    "EUR": "€",
    "HKD": "HK$",
    "CNY": "¥",
}


def currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOLS.get(code, code + " ")


def fmt_price(value: float, currency: str) -> str:
    """Format a price with currency symbol and appropriate decimals."""
    sym = currency_symbol(currency)
    if currency == "KRW":
        return f"{sym}{value:,.0f}"
    return f"{sym}{value:,.2f}"


def fetch_prices(ticker: str, period: str = "2y") -> pd.Series:
    """Download adjusted close prices from Yahoo Finance."""
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty or "Close" not in data.columns:
        raise ValueError(f"No close-price data for ticker={ticker!r}")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    if close.empty:
        raise ValueError(f"Close series is empty for ticker={ticker!r}")
    return close


def estimate_gbm_params(
    close_prices: pd.Series,
    ticker: str = "",
    trading_days: int = 252,
) -> GbmParams:
    """Estimate annualized mu and sigma from daily log returns."""
    log_ret = np.log(close_prices / close_prices.shift(1)).dropna()
    if log_ret.empty:
        raise ValueError("Not enough data to compute log returns")

    mu_daily    = float(log_ret.mean())
    sigma_daily = float(log_ret.std(ddof=1))

    return GbmParams(
        s0=float(close_prices.iloc[-1]),
        mu=mu_daily * trading_days,
        sigma=sigma_daily * np.sqrt(trading_days),
        currency=detect_currency(ticker),
    )


def estimate_jump_params(
    close_prices: pd.Series,
    trading_days: int = 252,
    z_threshold: float = 3.5,
) -> JumpParams:
    """Detect large daily moves as jumps; estimate λ, μ_j, σ_j for Merton jump-diffusion."""
    log_ret = np.log(close_prices / close_prices.shift(1)).dropna()
    if log_ret.empty:
        raise ValueError("Not enough data to estimate jump parameters")

    arr = log_ret.to_numpy(dtype=np.float64, copy=False)
    sigma_d = float(np.std(arr, ddof=1))
    mu_d = float(np.mean(arr))
    if sigma_d <= 0.0:
        return JumpParams(lambda_annual=0.0, mu_jump=0.0, sigma_jump=0.0)

    thresh = z_threshold * sigma_d
    mask = np.abs(arr - mu_d) > thresh
    n_days = int(arr.shape[0])
    years = n_days / float(trading_days)
    if years <= 0.0:
        return JumpParams(lambda_annual=0.0, mu_jump=0.0, sigma_jump=0.0)

    n_jump = int(np.count_nonzero(mask))
    lam = n_jump / years
    jump_returns = arr[mask]

    if n_jump == 0:
        return JumpParams(lambda_annual=0.0, mu_jump=0.0, sigma_jump=0.0)
    if n_jump == 1:
        return JumpParams(
            lambda_annual=float(lam),
            mu_jump=float(jump_returns[0]),
            sigma_jump=0.0,
        )

    mu_j = float(np.mean(jump_returns))
    sig_j = float(np.std(jump_returns, ddof=1))
    return JumpParams(lambda_annual=float(lam), mu_jump=mu_j, sigma_jump=sig_j)


def compute_risk_metrics(samples: np.ndarray, s0: float) -> dict[str, float]:
    """VaR(95%), CVaR / Expected Shortfall (95%), percentiles, profit probability."""
    p01 = float(np.quantile(samples, 0.01))
    p05 = float(np.quantile(samples, 0.05))
    p25 = float(np.quantile(samples, 0.25))
    p50 = float(np.quantile(samples, 0.50))
    p75 = float(np.quantile(samples, 0.75))
    p95 = float(np.quantile(samples, 0.95))
    p99 = float(np.quantile(samples, 0.99))
    up_prob = float(np.mean(samples > s0) * 100.0)
    var95_abs = float(max(0.0, s0 - p05))
    var95_pct = (var95_abs / s0) * 100.0 if s0 else 0.0

    sorted_s = np.sort(samples)
    k = max(1, int(np.floor(0.05 * sorted_s.size)))
    tail = sorted_s[:k]
    losses = s0 - tail
    cvar95_abs = float(np.mean(losses))
    cvar95_pct = (cvar95_abs / s0) * 100.0 if s0 else 0.0

    return {
        "p01": p01,
        "p05": p05,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p95": p95,
        "p99": p99,
        "up_probability_pct": up_prob,
        "var_95_abs": var95_abs,
        "var_95_pct": var95_pct,
        "cvar_95_abs": cvar95_abs,
        "cvar_95_pct": cvar95_pct,
    }
