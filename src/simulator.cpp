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

// Sum of Nj independent log-normal jump increments J_i ~ N(mu_j, sigma_j^2).
inline double jump_log_increment(std::mt19937_64& rng,
                                 std::poisson_distribution<int>& poisson,
                                 std::normal_distribution<double>& normal,
                                 double lambda_dt, double mu_j, double sigma_j) {
    if (lambda_dt <= 0.0) return 0.0;
    const int nj = poisson(rng);
    double sum = 0.0;
    for (int j = 0; j < nj; ++j) sum += mu_j + sigma_j * normal(rng);
    return sum;
}

// ---------------------------------------------------------------------------
// Terminal-only: one S_T per path (Merton jump-diffusion).
// log(S_T/S_0) = (mu - sigma^2/2)*T + sigma*sqrt(T)*Z + sum_jumps
// ---------------------------------------------------------------------------
struct TerminalWorker {
    double* out;
    double s0;
    double drift;
    double vol_sqrt_t;
    double lambda_t;
    double jump_mu;
    double jump_sigma;
    std::uint64_t seed;

    void operator()(std::size_t lo, std::size_t hi, std::size_t tid) const {
        std::mt19937_64 rng(seed + static_cast<std::uint64_t>(tid) * 7919ULL);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::poisson_distribution<int> poisson(lambda_t > 0.0 ? lambda_t : 0.0);

        for (std::size_t i = lo; i < hi;) {
            const double z = normal(rng);
            double jlog = 0.0;
            if (lambda_t > 0.0) jlog = jump_log_increment(rng, poisson, normal, lambda_t, jump_mu, jump_sigma);

            const double base = drift + jlog;
            const double w = vol_sqrt_t * z;
            out[i] = s0 * std::exp(base + w);
            ++i;
            if (i < hi) {
                out[i] = s0 * std::exp(base - w);
                ++i;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Multi-step path matrix: shape (n_paths, n_steps+1), row-major.
// Antithetic diffusion paths are stored as separate rows, not averaged away.
// ---------------------------------------------------------------------------
struct PathMatrixWorker {
    double* out;
    std::size_t stride;
    double s0;
    double drift_dt;
    double vol_sqrt_dt;
    double lambda_dt;
    double jump_mu;
    double jump_sigma;
    std::uint64_t seed;

    void operator()(std::size_t lo, std::size_t hi, std::size_t tid) const {
        std::mt19937_64 rng(seed + static_cast<std::uint64_t>(tid) * 7919ULL);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::poisson_distribution<int> poisson(lambda_dt > 0.0 ? lambda_dt : 0.0);

        for (std::size_t i = lo; i < hi;) {
            double sp = s0;
            double sm = s0;
            const std::size_t row = i * stride;
            const bool has_pair = (i + 1 < hi);
            const std::size_t pair_row = (i + 1) * stride;

            out[row] = sp;
            if (has_pair) out[pair_row] = sm;

            for (std::size_t k = 1; k < stride; ++k) {
                const double z = normal(rng);
                double jlog = 0.0;
                if (lambda_dt > 0.0) jlog = jump_log_increment(rng, poisson, normal, lambda_dt, jump_mu, jump_sigma);

                const double base = drift_dt + jlog;
                const double w = vol_sqrt_dt * z;
                sp *= std::exp(base + w);
                sm *= std::exp(base - w);
                out[row + k] = sp;
                if (has_pair) out[pair_row + k] = sm;
            }
            i += has_pair ? 2 : 1;
        }
    }
};

py::array_t<double> py_simulate_terminal(std::size_t n_paths, double s0, double mu, double sigma,
                                         double t, std::uint64_t seed, std::size_t n_threads,
                                         double jump_lambda, double jump_mu, double jump_sigma) {
    if (n_paths == 0 || s0 <= 0.0 || sigma < 0.0 || t <= 0.0)
        throw std::invalid_argument("Invalid simulation parameters");
    if (jump_lambda < 0.0 || jump_sigma < 0.0)
        throw std::invalid_argument("Invalid jump parameters");

    py::array_t<double> arr(static_cast<py::ssize_t>(n_paths));
    const double drift = (mu - 0.5 * sigma * sigma) * t;
    const double vol_sqrt_t = sigma * std::sqrt(t);
    const double lambda_t = jump_lambda * t;

    TerminalWorker w{arr.mutable_data(), s0, drift, vol_sqrt_t, lambda_t, jump_mu, jump_sigma, seed};
    {
        py::gil_scoped_release unlock;
        dispatch_threads(n_paths, n_threads, w);
    }
    return arr;
}

py::array_t<double> py_simulate_path_matrix(std::size_t n_paths, std::size_t n_steps, double s0,
                                            double mu, double sigma, double t, std::uint64_t seed,
                                            std::size_t n_threads, double jump_lambda,
                                            double jump_mu, double jump_sigma) {
    if (n_paths == 0 || n_steps == 0 || s0 <= 0.0 || sigma < 0.0 || t <= 0.0)
        throw std::invalid_argument("Invalid simulation parameters");
    if (jump_lambda < 0.0 || jump_sigma < 0.0)
        throw std::invalid_argument("Invalid jump parameters");

    const double dt = t / static_cast<double>(n_steps);
    py::array_t<double> arr(
        {static_cast<py::ssize_t>(n_paths), static_cast<py::ssize_t>(n_steps + 1)});
    const double drift_dt = (mu - 0.5 * sigma * sigma) * dt;
    const double vol_sqrt_dt = sigma * std::sqrt(dt);
    const double lambda_dt = jump_lambda * dt;

    PathMatrixWorker w{arr.mutable_data(), n_steps + 1, s0, drift_dt,
                       vol_sqrt_dt, lambda_dt, jump_mu, jump_sigma, seed};
    {
        py::gil_scoped_release unlock;
        dispatch_threads(n_paths, n_threads, w);
    }
    return arr;
}

}  // namespace

PYBIND11_MODULE(gbm_simulator, m) {
    m.doc() = "GBM / Merton jump-diffusion Monte Carlo — multi-threaded C++23 / pybind11 "
              "(per-thread std::mt19937_64, no shared RNG; antithetic diffusion; Poisson jumps)";

    m.def("simulate_gbm_paths", &py_simulate_terminal,
          py::arg("n_paths"), py::arg("s0"), py::arg("mu"), py::arg("sigma"), py::arg("t"),
          py::arg("seed") = 42ULL, py::arg("n_threads") = std::size_t{0},
          py::arg("jump_lambda") = 0.0, py::arg("jump_mu") = 0.0, py::arg("jump_sigma") = 0.0);

    m.def("simulate_gbm_path_matrix", &py_simulate_path_matrix,
          py::arg("n_paths"), py::arg("n_steps"), py::arg("s0"), py::arg("mu"), py::arg("sigma"),
          py::arg("t"), py::arg("seed") = 42ULL, py::arg("n_threads") = std::size_t{0},
          py::arg("jump_lambda") = 0.0, py::arg("jump_mu") = 0.0, py::arg("jump_sigma") = 0.0);
}
