"""Yahoo Finance data fetching and GBM parameter estimation."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


class YahooFinanceFetchError(ValueError):
    """Yahoo Finance 시세를 가져오지 못했을 때 (네트워크·차단·API 제한 등)."""

    pass


_FETCH_RETRY_HINT_KO = (
    "일시적으로 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요. "
    "(네트워크 혼잡 또는 Yahoo 측 제한일 수 있습니다.)"
)


def _close_from_frame(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    frame = df
    if isinstance(frame.columns, pd.MultiIndex):
        try:
            frame = frame.copy()
            frame.columns = [
                c[-1] if isinstance(c, tuple) and len(c) > 0 else c for c in frame.columns
            ]
        except Exception:
            frame = pd.DataFrame(df)

    for key in ("Close", "Adj Close"):
        if key in frame.columns:
            s = frame[key]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            out = pd.Series(s, dtype=float).dropna()
            if len(out) >= 10:
                return out
    return None


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
    """Download adjusted close prices from Yahoo Finance.

    세션을 넘기지 않고 yfinance 기본 동작을 사용합니다. 최신 yfinance는
    ``requests.Session`` 같은 커스텀 세션 대신 내장(curl_cffi) 세션을 기대하는 경우가 있어,
    여기서는 ``session=`` 을 지정하지 않습니다.

    Yahoo가 빈 응답을 주거나 제한하는 경우가 있어 짧게 재시도합니다.
    """
    sym = ticker.strip().upper()
    if not sym:
        raise ValueError("Ticker symbol is empty.")

    last_exc: Exception | None = None

    def try_history(p: str) -> pd.Series | None:
        nonlocal last_exc
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period=p, auto_adjust=True)
            return _close_from_frame(hist)
        except Exception as exc:
            last_exc = exc
            return None

    def try_download(p: str) -> pd.Series | None:
        nonlocal last_exc
        try:
            data = yf.download(
                sym,
                period=p,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            return _close_from_frame(data)
        except Exception as exc:
            last_exc = exc
            return None

    for attempt in range(3):
        for p in (period, "1y", "6mo", "3mo"):
            s = try_history(p)
            if s is None:
                s = try_download(p)
            if s is not None:
                return s
        time.sleep(0.5 * (attempt + 1))

    tech = ""
    if last_exc is not None:
        msg = str(last_exc).strip()
        if msg:
            tech = f"\n\n(참고: {msg})"
    raise YahooFinanceFetchError(
        f"{_FETCH_RETRY_HINT_KO}\n\n티커: {sym}.{tech}"
    ) from last_exc


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
