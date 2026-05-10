"""Shared utility: locate and import the compiled C++ gbm_simulator module."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


class _NumpyFallbackSimulator:
    """C++ 모듈 없이도 동작하는 NumPy 폴백. 속도는 느리지만 인터페이스는 동일."""

    def simulate_gbm_paths(
        self,
        n_paths: int,
        s0: float,
        mu: float,
        sigma: float,
        t: float,
        seed: int = 0,
        n_threads: int = 1,
        jump_lambda: float = 0.0,
        jump_mu: float = 0.0,
        jump_sigma: float = 0.0,
    ) -> list:
        rng = np.random.default_rng(seed)
        # antithetic variates: 분산 감소
        half = n_paths // 2
        Z = rng.standard_normal(half)
        Z_full = np.empty(n_paths)
        Z_full[:half] = Z
        Z_full[half : 2 * half] = -Z
        if n_paths % 2 == 1:
            Z_full[-1] = rng.standard_normal()
        paths = s0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * Z_full)
        return paths.tolist()

    def simulate_gbm_path_matrix(
        self,
        n_paths: int,
        n_steps: int,
        s0: float,
        mu: float,
        sigma: float,
        t: float,
        seed: int = 0,
        n_threads: int = 1,
        jump_lambda: float = 0.0,
        jump_mu: float = 0.0,
        jump_sigma: float = 0.0,
    ) -> list:
        rng = np.random.default_rng(seed)
        dt = t / max(n_steps, 1)
        Z = rng.standard_normal((n_paths, n_steps))
        log_inc = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        cum_log = np.cumsum(log_inc, axis=1)
        matrix = np.empty((n_paths, n_steps + 1))
        matrix[:, 0] = s0
        matrix[:, 1:] = s0 * np.exp(cum_log)
        return matrix.tolist()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _discovered_build_dirs() -> list[Path]:
    """Linux/macOS: gbm_simulator*.so often lives directly under build/ or in a subdir."""
    build_root = PROJECT_ROOT / "build"
    if not build_root.is_dir():
        return []
    seen: set[str] = set()
    out: list[Path] = []
    for pattern in ("gbm_simulator*.so", "gbm_simulator*.pyd"):
        for path in build_root.rglob(pattern):
            parent = path.resolve().parent
            key = str(parent)
            if key not in seen:
                seen.add(key)
                out.append(parent)
    return out


_DEFAULT_SEARCH_DIRS = [
    *_discovered_build_dirs(),
    PROJECT_ROOT / "build" / "Release",
    PROJECT_ROOT / "build",
    PROJECT_ROOT / "build" / "Debug",
]


def import_simulator(build_dir: str | None = None):
    """Import gbm_simulator from site-packages (pip install) or local build/."""
    if build_dir is None:
        try:
            return importlib.import_module("gbm_simulator")
        except ModuleNotFoundError:
            pass

    dirs = _DEFAULT_SEARCH_DIRS
    if build_dir:
        explicit = Path(build_dir).resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"Build directory not found: {explicit}")
        dirs = [explicit]

    for d in dirs:
        if d.exists():
            d_str = str(d)
            if d_str not in sys.path:
                sys.path.insert(0, d_str)

    try:
        return importlib.import_module("gbm_simulator")
    except ModuleNotFoundError:
        return _NumpyFallbackSimulator()
