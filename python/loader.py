"""Shared utility: locate and import the compiled C++ gbm_simulator module."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_SEARCH_DIRS = [
    PROJECT_ROOT / "build" / "Release",
    PROJECT_ROOT / "build",
    PROJECT_ROOT / "build" / "Debug",
]


def import_simulator(build_dir: str | None = None):
    """Add candidate directories to sys.path and import gbm_simulator."""
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
            "  cmake -S . -B build\n"
            "  cmake --build build --config Release"
        ) from exc
