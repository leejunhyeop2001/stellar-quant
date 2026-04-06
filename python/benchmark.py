"""Benchmark: C++ (multi-thread) vs NumPy (vectorised) vs pure-Python (for-loop)."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from loader import PROJECT_ROOT, import_simulator

SEPARATOR = "=" * 78
THIN_SEP  = "-" * 78

# ---------------------------------------------------------------------------
# Python-only GBM implementations
# ---------------------------------------------------------------------------
def gbm_numpy(n: int, s0: float, mu: float, sigma: float, t: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    return s0 * np.exp((mu - 0.5 * sigma * sigma) * t + sigma * np.sqrt(t) * z)


def gbm_pure_python(n: int, s0: float, mu: float, sigma: float, t: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    drift = (mu - 0.5 * sigma * sigma) * t
    vol = sigma * math.sqrt(t)
    return [s0 * math.exp(drift + vol * rng.gauss(0.0, 1.0)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def best_of(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# Dark-themed chart (consistent with main.py style)
# ---------------------------------------------------------------------------
COLORS = {
    "bg":    "#1C1C2E",
    "panel": "#252540",
    "text":  "#E0E0E0",
    "grid":  "#3A3A5C",
    "cpp":   "#5B9BD5",
    "numpy": "#FF9500",
    "py":    "#4CAF50",
}


def save_chart(
    counts: list[int],
    cpp_s: list[float],
    np_s: list[float],
    py_s: list[float | None],
    out_path: Path,
) -> None:
    labels = [f"{c:,}" for c in counts]
    x = np.arange(len(counts))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["panel"])

    ax.bar(x - w, cpp_s, width=w, label="C++ Hybrid",  color=COLORS["cpp"],   edgecolor="none")
    ax.bar(x,     np_s,  width=w, label="NumPy",       color=COLORS["numpy"], edgecolor="none")

    py_vals = [v if v is not None else 0.0 for v in py_s]
    if any(v > 0 for v in py_vals):
        ax.bar(x + w, py_vals, width=w, label="Pure Python", color=COLORS["py"], edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    ax.set_xlabel("Simulations", fontsize=12, color=COLORS["text"])
    ax.set_ylabel("Time (seconds, lower = better)", fontsize=12, color=COLORS["text"])
    ax.set_title("GBM Benchmark: C++ vs NumPy vs Pure Python",
                 fontsize=14, fontweight="bold", color=COLORS["text"], pad=12)
    ax.legend(fontsize=10, facecolor=COLORS["panel"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    ax.grid(axis="y", alpha=0.25, color=COLORS["grid"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
PURE_PYTHON_LIMIT = 200_000


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark C++ GBM vs Python")
    p.add_argument("--build-dir", type=str, default=None)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--threads",   type=int,   default=0)
    p.add_argument("--repeat",    type=int,   default=3)
    p.add_argument("--s0",        type=float, default=100.0)
    p.add_argument("--mu",        type=float, default=0.10)
    p.add_argument("--sigma",     type=float, default=0.20)
    p.add_argument("--years",     type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    sim = import_simulator(args.build_dir)

    counts = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
    cpp_sec:  list[float]        = []
    np_sec:   list[float]        = []
    py_sec:   list[float | None] = []

    print(f"\n{SEPARATOR}")
    print(f"   STELLAR-QUANT  |  Performance Benchmark")
    print(SEPARATOR)
    print(f"\n  {'n':>12} | {'C++ (s)':>10} | {'NumPy (s)':>10} | {'Python (s)':>10} | {'vs NumPy':>10} | {'vs Python':>10}")
    print(f"  {THIN_SEP[:72]}")

    for n in counts:
        t_cpp = best_of(lambda n=n: sim.simulate_gbm_paths(
            n_paths=n, s0=args.s0, mu=args.mu,
            sigma=args.sigma, t=args.years, seed=args.seed, n_threads=args.threads,
        ), repeat=args.repeat)

        t_np = best_of(lambda n=n: gbm_numpy(
            n, args.s0, args.mu, args.sigma, args.years, args.seed,
        ), repeat=args.repeat)

        t_py: float | None = None
        if n <= PURE_PYTHON_LIMIT:
            t_py = best_of(lambda n=n: gbm_pure_python(
                n, args.s0, args.mu, args.sigma, args.years, args.seed,
            ), repeat=args.repeat)

        cpp_sec.append(t_cpp)
        np_sec.append(t_np)
        py_sec.append(t_py)

        sp_np = t_np / t_cpp if t_cpp > 0 else float("inf")
        sp_py = (t_py / t_cpp) if (t_py and t_cpp > 0) else None

        py_str = f"{t_py:.6f}" if t_py is not None else "       —"
        sp_py_str = f"{sp_py:>8.1f}×" if sp_py is not None else "       —"
        print(f"  {n:>12,} | {t_cpp:>10.6f} | {t_np:>10.6f} | {py_str:>10} | {sp_np:>9.2f}× | {sp_py_str:>10}")

    print(f"\n{SEPARATOR}")

    chart = PROJECT_ROOT / "python" / "benchmark_bar_chart.png"
    save_chart(counts, cpp_sec, np_sec, py_sec, chart)
    print(f"  Chart  → {chart}")

    speedup = [np_s / cpp_s if cpp_s > 0 else 0 for cpp_s, np_s in zip(cpp_sec, np_sec)]
    results = {
        "counts": counts,
        "cpp_seconds": cpp_sec,
        "numpy_seconds": np_sec,
        "python_seconds": py_sec,
        "speedup_cpp_vs_numpy": speedup,
    }
    json_path = PROJECT_ROOT / "python" / "benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_path = PROJECT_ROOT / "python" / "benchmark_summary.md"
    md = [
        "# Benchmark Summary\n",
        "| Simulations | C++ (s) | NumPy (s) | Python (s) | Speed-up (vs NumPy) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n, tc, tn, tp, s in zip(counts, cpp_sec, np_sec, py_sec, speedup):
        tp_str = f"{tp:.6f}" if tp is not None else "—"
        md.append(f"| {n:,} | {tc:.6f} | {tn:.6f} | {tp_str} | {s:.2f}× |")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  Report → {md_path}")
    print(f"{SEPARATOR}\n")


if __name__ == "__main__":
    main()
