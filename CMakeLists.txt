cmake_minimum_required(VERSION 3.22)
project(StellarQuantSimulator LANGUAGES CXX)

# ISO C++25 미발표 — 툴체인 기준 최신 안정 표준은 C++23. ( MSVC: /std:c++23 )
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(gbm_simulator MODULE src/simulator.cpp)

if(MSVC)
    target_compile_options(gbm_simulator PRIVATE /O2 /W4 /fp:fast)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("/arch:AVX2" MSVC_HAS_AVX2)
    if(MSVC_HAS_AVX2)
        target_compile_options(gbm_simulator PRIVATE /arch:AVX2)
    endif()
else()
    # Avoid -march=native on CI / Streamlit Cloud for predictable portable binaries.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=x86-64-v2" HAS_X86_64_V2)
    if(HAS_X86_64_V2)
        target_compile_options(gbm_simulator PRIVATE -O3 -march=x86-64-v2 -Wall -Wextra -Wpedantic)
    else()
        target_compile_options(gbm_simulator PRIVATE -O3 -Wall -Wextra -Wpedantic)
    endif()
endif()
