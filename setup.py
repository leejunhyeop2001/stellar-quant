"""Build and install the pybind11 C++ extension via CMake (pip install / Streamlit Cloud)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        if sys.platform.startswith("win"):
            cmake_args += [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
            ]
        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]

        try:
            import pybind11

            cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        except ImportError:
            pass

        if "CMAKE_ARGS" in os.environ:
            cmake_args += os.environ["CMAKE_ARGS"].strip().split()

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cmake_cli = ["cmake", ext.sourcedir, *cmake_args]
        subprocess.run(cmake_cli, cwd=build_temp, check=True)

        build_cmd = ["cmake", "--build", ".", "--parallel"]
        if sys.platform.startswith("win"):
            build_cmd += ["--config", cfg]
        subprocess.run(build_cmd, cwd=build_temp, check=True)


setup(
    name="stellar-quant",
    version="0.1.0",
    description="Monte Carlo GBM / Merton jump-diffusion C++ engine (pybind11)",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8")
    if (ROOT / "README.md").exists()
    else "",
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "requests>=2.26",
        "yfinance",
        "matplotlib",
        "pybind11>=2.10",
        "streamlit>=1.28",
        "plotly>=5",
    ],
    ext_modules=[CMakeExtension("gbm_simulator", sourcedir=str(ROOT))],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.9",
)
