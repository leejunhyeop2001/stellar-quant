cmake_minimum_required(VERSION 3.16)
project(StellarQuantSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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
    target_compile_options(gbm_simulator PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
endif()
