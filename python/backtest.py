# 과거 구간으로 1년 뒤 예측 분포의 보정 상태를 검증하는 CLI
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import (
    GBM_MU_ANNUAL_CAP,
    JumpParams,
    MU_MARKET_PRIOR,
    compute_risk_metrics,
    estimate_gbm_params,
    estimate_jump_params,
    fetch_prices,
    fmt_price,
)
from loader import PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class BacktestRow:
    model: str
    asof: str
    s0: float
    actual: float
    p05: float
    p50: float
    p95: float
    up_probability_pct: float
    realized_percentile: float
    interval_90_hit: bool
    direction_hit: bool
    median_abs_error_pct: float
    brier_up: float


def _simulate_terminal_numpy(
    *,
    n_paths: int,
    s0: float,
    mu: float,
    sigma: float,
    years: float,
    jump: JumpParams,
    rng: np.random.Generator,
) -> np.ndarray:
    drift = (mu - 0.5 * sigma * sigma) * years
    diffusion = sigma * np.sqrt(years) * rng.standard_normal(n_paths)
    jump_log = np.zeros(n_paths, dtype=np.float64)
    if jump.lambda_annual > 0.0:
        counts = rng.poisson(jump.lambda_annual * years, size=n_paths)
        jump_log += counts * jump.mu_jump
        if jump.sigma_jump > 0.0:
            jump_log += np.sqrt(counts) * jump.sigma_jump * rng.standard_normal(n_paths)
    return s0 * np.exp(drift + diffusion + jump_log)


def _flat_median_mu(sigma: float) -> float:
    return 0.5 * sigma * sigma


def _model_samples(
    model: str,
    *,
    s0: float,
    mu: float,
    sigma: float,
    years: float,
    jump: JumpParams,
    beta_prior_mu: float,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if model == "stellar":
        return _simulate_terminal_numpy(
            n_paths=n_paths,
            s0=s0,
            mu=mu,
            sigma=sigma,
            years=years,
            jump=jump,
            rng=rng,
        )
    if model == "market_prior":
        return _simulate_terminal_numpy(
            n_paths=n_paths,
            s0=s0,
            mu=MU_MARKET_PRIOR,
            sigma=sigma,
            years=years,
            jump=JumpParams(0.0, 0.0, 0.0),
            rng=rng,
        )
    if model == "beta_prior":
        return _simulate_terminal_numpy(
            n_paths=n_paths,
            s0=s0,
            mu=beta_prior_mu,
            sigma=sigma,
            years=years,
            jump=JumpParams(0.0, 0.0, 0.0),
            rng=rng,
        )
    if model == "flat_median":
        return _simulate_terminal_numpy(
            n_paths=n_paths,
            s0=s0,
            mu=_flat_median_mu(sigma),
            sigma=sigma,
            years=years,
            jump=JumpParams(0.0, 0.0, 0.0),
            rng=rng,
        )
    raise ValueError(f"Unknown backtest model: {model}")


def _estimate_market_beta(asset_close: pd.Series, market_close: pd.Series | None) -> float:
    if market_close is None or market_close.empty:
        return 1.0
    asset = asset_close.copy()
    market = market_close.copy()
    asset.index = pd.to_datetime(asset.index).tz_localize(None).normalize()
    market.index = pd.to_datetime(market.index).tz_localize(None).normalize()
    market = market[~market.index.duplicated(keep="last")].sort_index()
    asset = asset[~asset.index.duplicated(keep="last")].sort_index()
    market_aligned = market.reindex(asset.index, method="ffill")
    frame = pd.concat(
        [
            np.log(asset / asset.shift(1)).rename("asset"),
            np.log(market_aligned / market_aligned.shift(1)).rename("market"),
        ],
        axis=1,
    ).dropna()
    if len(frame) < 30:
        return 1.0
    market_var = float(np.var(frame["market"].to_numpy(dtype=np.float64), ddof=1))
    if market_var <= 1e-15:
        return 1.0
    cov = float(np.cov(frame["asset"], frame["market"], ddof=1)[0, 1])
    return float(np.clip(cov / market_var, -3.0, 3.0))


def _score_samples(
    model: str,
    asof: str,
    s0: float,
    actual: float,
    samples: np.ndarray,
    sigma: float,
    years: float,
) -> BacktestRow:
    metrics = compute_risk_metrics(samples, s0, sigma_annual=sigma, horizon_years=years)
    realized_percentile = float(np.mean(samples <= actual) * 100.0)
    actual_up = actual > s0
    predicted_up_prob = metrics["up_probability_pct"] / 100.0
    return BacktestRow(
        model=model,
        asof=asof,
        s0=s0,
        actual=actual,
        p05=metrics["p05"],
        p50=metrics["p50"],
        p95=metrics["p95"],
        up_probability_pct=metrics["up_probability_pct"],
        realized_percentile=realized_percentile,
        interval_90_hit=bool(metrics["p05"] <= actual <= metrics["p95"]),
        direction_hit=bool((metrics["up_probability_pct"] >= 50.0) == actual_up),
        median_abs_error_pct=abs(actual - metrics["p50"]) / s0 * 100.0,
        brier_up=float((predicted_up_prob - float(actual_up)) ** 2),
    )


def run_backtest(
    close: pd.Series,
    *,
    ticker: str,
    market_close: pd.Series | None,
    train_days: int,
    horizon_days: int,
    stride_days: int,
    n_paths: int,
    seed: int,
) -> tuple[list[BacktestRow], str]:
    if len(close) < train_days + horizon_days + 2:
        raise ValueError(
            f"Not enough data for backtest. Need at least {train_days + horizon_days + 2} rows."
        )

    rng = np.random.default_rng(seed)
    rows: list[BacktestRow] = []
    years = horizon_days / 252.0

    split_positions = range(train_days, len(close) - horizon_days, stride_days)
    for end in split_positions:
        train = close.iloc[end - train_days:end + 1]
        actual = float(close.iloc[end + horizon_days])
        params = estimate_gbm_params(train, ticker=ticker)
        jump = estimate_jump_params(train)
        beta = _estimate_market_beta(train, market_close)
        beta_prior_mu = float(np.clip(beta * MU_MARKET_PRIOR, -0.35, GBM_MU_ANNUAL_CAP))
        asof_idx = close.index[end]
        asof = str(asof_idx.date() if hasattr(asof_idx, "date") else asof_idx)

        for model in ("stellar", "market_prior", "beta_prior", "flat_median"):
            samples = _model_samples(
                model,
                s0=params.s0,
                mu=params.mu,
                sigma=params.sigma,
                years=years,
                jump=jump,
                beta_prior_mu=beta_prior_mu,
                n_paths=n_paths,
                rng=rng,
            )
            rows.append(
                _score_samples(
                    model,
                    asof,
                    params.s0,
                    actual,
                    samples,
                    params.sigma,
                    years,
                )
            )

    return rows, params.currency


def _summarize(rows: list[BacktestRow]) -> list[dict[str, float | str | int]]:
    summary: list[dict[str, float | str | int]] = []
    models = sorted({r.model for r in rows})
    for model in models:
        subset = [r for r in rows if r.model == model]
        n = len(subset)
        if n == 0:
            continue
        coverage = np.mean([r.interval_90_hit for r in subset]) * 100.0
        direction = np.mean([r.direction_hit for r in subset]) * 100.0
        median_err = np.mean([r.median_abs_error_pct for r in subset])
        brier = np.mean([r.brier_up for r in subset])
        pct_mean = np.mean([r.realized_percentile for r in subset])
        summary.append(
            {
                "model": model,
                "splits": n,
                "interval_90_coverage_pct": float(coverage),
                "direction_accuracy_pct": float(direction),
                "mean_median_abs_error_pct": float(median_err),
                "mean_brier_up": float(brier),
                "mean_realized_percentile": float(pct_mean),
            }
        )
    return summary


def _write_outputs(
    *,
    ticker: str,
    rows: list[BacktestRow],
    summary: list[dict[str, float | str | int]],
    currency: str,
    out_prefix: Path,
) -> tuple[Path, Path]:
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    payload = {
        "ticker": ticker,
        "currency": currency,
        "summary": summary,
        "rows": [asdict(r) for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Backtest Summary: {ticker}",
        "",
        "## Model Summary",
        "",
        "| Model | Splits | 90% Coverage | Direction Accuracy | Median Abs Error | Brier Up | Mean Realized Percentile |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['model']} | {item['splits']} | "
            f"{item['interval_90_coverage_pct']:.1f}% | "
            f"{item['direction_accuracy_pct']:.1f}% | "
            f"{item['mean_median_abs_error_pct']:.1f}% | "
            f"{item['mean_brier_up']:.3f} | "
            f"{item['mean_realized_percentile']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Split Detail",
            "",
            "| Model | As-of | S0 | Actual | P05 | P50 | P95 | Up Prob. | Realized Pctl. | 90% Hit | Direction Hit |",
            "|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r.model} | {r.asof} | {fmt_price(r.s0, currency)} | "
            f"{fmt_price(r.actual, currency)} | {fmt_price(r.p05, currency)} | "
            f"{fmt_price(r.p50, currency)} | {fmt_price(r.p95, currency)} | "
            f"{r.up_probability_pct:.1f}% | {r.realized_percentile:.1f}% | "
            f"{'Y' if r.interval_90_hit else 'N'} | {'Y' if r.direction_hit else 'N'} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rolling 1-year calibration backtest")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--market-ticker", default="SPY")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--horizon-days", type=int, default=252)
    parser.add_argument("--stride-days", type=int, default=63)
    parser.add_argument("--paths", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "python" / "backtest_results",
        help="Output path prefix without extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    close = fetch_prices(args.ticker, period=args.period)
    market_close = None
    if args.market_ticker:
        try:
            market_close = fetch_prices(args.market_ticker, period=args.period)
        except Exception as exc:
            print(f"Market benchmark unavailable: {args.market_ticker} ({exc})")
    rows, currency = run_backtest(
        close,
        ticker=args.ticker,
        market_close=market_close,
        train_days=args.train_days,
        horizon_days=args.horizon_days,
        stride_days=args.stride_days,
        n_paths=args.paths,
        seed=args.seed,
    )
    summary = _summarize(rows)
    json_path, md_path = _write_outputs(
        ticker=args.ticker,
        rows=rows,
        summary=summary,
        currency=currency,
        out_prefix=args.out,
    )

    print(f"Backtest complete: {args.ticker}")
    for item in summary:
        print(
            f"  {item['model']}: coverage={item['interval_90_coverage_pct']:.1f}%, "
            f"direction={item['direction_accuracy_pct']:.1f}%, "
            f"median_err={item['mean_median_abs_error_pct']:.1f}%"
        )
    print(f"JSON  -> {json_path}")
    print(f"Report -> {md_path}")


if __name__ == "__main__":
    main()
