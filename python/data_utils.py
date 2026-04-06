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
