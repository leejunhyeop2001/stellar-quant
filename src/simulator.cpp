#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

std::size_t resolve_thread_count(std::size_t n_items, std::size_t n_threads) {
    const auto hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    return std::max<std::size_t>(1, std::min<std::size_t>(n_threads == 0 ? hw : n_threads, n_items));
}

template <typename Fn>
void dispatch_threads(std::size_t n_items, std::size_t n_threads, Fn&& fn) {
    const std::size_t tc = resolve_thread_count(n_items, n_threads);
    std::vector<std::thread> workers;
    workers.reserve(tc);

    const std::size_t base = n_items / tc;
    const std::size_t rem  = n_items % tc;
    std::size_t begin = 0;

    for (std::size_t tid = 0; tid < tc; ++tid) {
        const std::size_t end = begin + base + (tid < rem ? 1 : 0);
        workers.emplace_back(fn, begin, end, tid);
        begin = end;
    }
    for (auto& w : workers) w.join();
}

// ---------------------------------------------------------------------------
// Terminal-only: one S_T per path (flat 1-D array, ~80 MB for 10M)
// ---------------------------------------------------------------------------
struct TerminalWorker {
    double* out;
    double s0, drift, vol;
    std::uint64_t seed;

    void operator()(std::size_t lo, std::size_t hi, std::size_t tid) const {
        std::mt19937 rng(static_cast<std::mt19937::result_type>(seed + tid * 7919ULL));
        std::normal_distribution<double> N(0.0, 1.0);
        for (std::size_t i = lo; i < hi; ++i)
            out[i] = s0 * std::exp(drift + vol * N(rng));
    }
};

// ---------------------------------------------------------------------------
// Multi-step path matrix: shape (n_paths, n_steps+1), row-major
// ---------------------------------------------------------------------------
struct PathMatrixWorker {
    double* out;
    std::size_t stride;
    double s0, drift, vol;
    std::uint64_t seed;

    void operator()(std::size_t lo, std::size_t hi, std::size_t tid) const {
        std::mt19937 rng(static_cast<std::mt19937::result_type>(seed + tid * 7919ULL));
        std::normal_distribution<double> N(0.0, 1.0);
        for (std::size_t i = lo; i < hi; ++i) {
            double s = s0;
            const std::size_t row = i * stride;
            out[row] = s;
            for (std::size_t k = 1; k < stride; ++k) {
                s *= std::exp(drift + vol * N(rng));
                out[row + k] = s;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Python wrappers (GIL released during computation)
// ---------------------------------------------------------------------------
py::array_t<double> py_simulate_terminal(
    std::size_t n_paths, double s0, double mu, double sigma, double t,
    std::uint64_t seed, std::size_t n_threads) {
    if (n_paths == 0 || s0 <= 0.0 || sigma < 0.0 || t <= 0.0)
        throw std::invalid_argument("Invalid simulation parameters");

    py::array_t<double> arr(static_cast<py::ssize_t>(n_paths));
    TerminalWorker w{arr.mutable_data(), s0,
                     (mu - 0.5 * sigma * sigma) * t,
                     sigma * std::sqrt(t), seed};
    {
        py::gil_scoped_release unlock;
        dispatch_threads(n_paths, n_threads, w);
    }
    return arr;
}

py::array_t<double> py_simulate_path_matrix(
    std::size_t n_paths, std::size_t n_steps,
    double s0, double mu, double sigma, double t,
    std::uint64_t seed, std::size_t n_threads) {
    if (n_paths == 0 || n_steps == 0 || s0 <= 0.0 || sigma < 0.0 || t <= 0.0)
        throw std::invalid_argument("Invalid simulation parameters");

    const double dt = t / static_cast<double>(n_steps);
    py::array_t<double> arr({
        static_cast<py::ssize_t>(n_paths),
        static_cast<py::ssize_t>(n_steps + 1)});
    PathMatrixWorker w{arr.mutable_data(), n_steps + 1, s0,
                       (mu - 0.5 * sigma * sigma) * dt,
                       sigma * std::sqrt(dt), seed};
    {
        py::gil_scoped_release unlock;
        dispatch_threads(n_paths, n_threads, w);
    }
    return arr;
}

}  // namespace

PYBIND11_MODULE(gbm_simulator, m) {
    m.doc() = "GBM Monte Carlo simulator — multi-threaded C++17 / pybind11";

    m.def("simulate_gbm_paths", &py_simulate_terminal,
          py::arg("n_paths"), py::arg("s0"), py::arg("mu"),
          py::arg("sigma"), py::arg("t"),
          py::arg("seed") = 42, py::arg("n_threads") = 0);

    m.def("simulate_gbm_path_matrix", &py_simulate_path_matrix,
          py::arg("n_paths"), py::arg("n_steps"), py::arg("s0"),
          py::arg("mu"), py::arg("sigma"), py::arg("t"),
          py::arg("seed") = 42, py::arg("n_threads") = 0);
}
