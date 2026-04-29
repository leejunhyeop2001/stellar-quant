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
    """Estimate annualized GBM parameters from daily log returns.

    The drift is deliberately shrunk because a one-year sample mean is noisy and
    can dominate risk simulations. Volatility blends plain historical sigma with
    an EWMA/GARCH-style estimate so recent volatility clusters carry more weight.
    """
    log_ret = np.log(close_prices / close_prices.shift(1)).dropna()
    if log_ret.empty:
        raise ValueError("Not enough data to compute log returns")

    arr = log_ret.to_numpy(dtype=np.float64, copy=False)
    mu_daily = float(np.mean(arr))
    sigma_daily = float(np.std(arr, ddof=1))
    if sigma_daily <= 0.0:
        raise ValueError("Volatility is zero; cannot estimate GBM parameters")

    # EWMA variance is a lightweight GARCH(1,1)-style volatility clustering proxy.
    lam = 0.94
    ewma_var = sigma_daily * sigma_daily
    for r in arr:
        ewma_var = lam * ewma_var + (1.0 - lam) * float(r * r)
    ewma_sigma_daily = float(np.sqrt(max(ewma_var, 0.0)))
    sigma_daily = max(sigma_daily, 0.7 * sigma_daily + 0.3 * ewma_sigma_daily)

    # Keep historical drift from overwhelming diffusion in short, noisy samples.
    mu_annual_raw = mu_daily * trading_days
    mu_annual = float(np.clip(mu_annual_raw, -0.35, 0.35))

    return GbmParams(
        s0=float(close_prices.iloc[-1]),
        mu=mu_annual,
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


# 종료 로그수익의 Fat-tail을 리스크 점수에 반영할 때, 초과첨도 1당 가중 (Gaussian ≈ 0)
RISK_TAIL_KURTOSIS_LAMBDA = 0.10


def compute_risk_metrics(
    samples: np.ndarray,
    s0: float,
    *,
    sigma_annual: float | None = None,
    horizon_years: float | None = None,
    tail_kurtosis_lambda: float = RISK_TAIL_KURTOSIS_LAMBDA,
) -> dict[str, float]:
    """VaR/CVaR, 상승 확률, 정규화 리스크 점수 및 로그수익 초과첨도.

    Risk Score (정규화): (VaR95/S0) / (σ√T). 단순 GBM에서는 분모가 대략적인 로그
    수익 표준편차 스케일에 해당. 점프·꼬리는 초과첨도 기반 계수로 점수를 상향 조정.

    ``sigma_annual``·``horizon_years`` 가 없으면 점수 필드는 0으로 두고, 첨도만 계산한다.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0 or s0 <= 0.0:
        raise ValueError("Invalid samples or initial price")

    p01 = float(np.quantile(samples, 0.01))
    p05 = float(np.quantile(samples, 0.05))
    p25 = float(np.quantile(samples, 0.25))
    p50 = float(np.quantile(samples, 0.50))
    p75 = float(np.quantile(samples, 0.75))
    p95 = float(np.quantile(samples, 0.95))
    p99 = float(np.quantile(samples, 0.99))
    log_returns = np.log(np.maximum(samples, np.finfo(np.float64).tiny) / s0)
    up_prob = float(np.mean(log_returns > 0.0) * 100.0)
    var95_abs = float(max(0.0, s0 - p05))
    var95_pct = (var95_abs / s0) * 100.0 if s0 else 0.0

    sorted_s = np.sort(samples)
    k = max(1, int(np.floor(0.05 * sorted_s.size)))
    tail = sorted_s[:k]
    losses = np.maximum(0.0, s0 - tail)
    cvar95_abs = float(np.mean(losses))
    cvar95_pct = (cvar95_abs / s0) * 100.0 if s0 else 0.0

    # Fisher 초과첨도: Gaussian 근사에서 종료 로그수익은 ~0
    lr = log_returns
    if lr.size > 4:
        m_lr = float(np.mean(lr))
        s_lr = float(np.std(lr, ddof=1))
        if s_lr > 1e-15:
            z = (lr - m_lr) / s_lr
            log_return_excess_kurtosis = float(np.mean(z * z * z * z) - 3.0)
        else:
            log_return_excess_kurtosis = 0.0
    else:
        log_return_excess_kurtosis = 0.0

    fat_tail_feel_index = float(max(0.0, log_return_excess_kurtosis))
    tail_adjustment_factor = float(1.0 + tail_kurtosis_lambda * fat_tail_feel_index)

    risk_score_base = 0.0
    risk_score = 0.0
    sigma_sqrt_horizon = 0.0
    if (
        sigma_annual is not None
        and horizon_years is not None
        and sigma_annual > 1e-15
        and horizon_years > 1e-15
    ):
        sigma_sqrt_horizon = float(sigma_annual * np.sqrt(horizon_years))
        loss_frac = float(var95_abs / s0) if s0 else 0.0
        if sigma_sqrt_horizon > 1e-15:
            risk_score_base = loss_frac / sigma_sqrt_horizon
        else:
            risk_score_base = float("inf") if loss_frac > 0.0 else 0.0
        if np.isfinite(risk_score_base):
            risk_score = float(risk_score_base * tail_adjustment_factor)
        else:
            risk_score = risk_score_base

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
        "log_return_excess_kurtosis": log_return_excess_kurtosis,
        "fat_tail_feel_index": fat_tail_feel_index,
        "tail_adjustment_factor": tail_adjustment_factor,
        "risk_score_base": float(risk_score_base) if np.isfinite(risk_score_base) else 999.0,
        "risk_score": float(risk_score) if np.isfinite(risk_score) else 999.0,
        "sigma_sqrt_horizon": sigma_sqrt_horizon,
    }
