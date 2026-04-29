"""Shared utility: locate and import the compiled C++ gbm_simulator module."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

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
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import 'gbm_simulator'. Build the C++ module first:\n"
            "  Linux/macOS: cmake -S . -B build && cmake --build build -j\"$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)\"\n"
            "  Windows MSVC: cmake -S . -B build && cmake --build build --config Release"
        ) from exc
