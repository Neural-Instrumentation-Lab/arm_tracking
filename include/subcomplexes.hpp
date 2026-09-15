#pragma once
// Direct ports of the small helper classes in garrido_brain_v01.py:
// CFsubcomplex, CFsubcomplexALT, MFsubcomplex, PFSpikeHistory, ltd_kernel.

#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <random>
#include <set>

// ---------------------------------------------------------------------
// CFsubcomplex (from C_interface_for_robot_control.cpp).
// Not used by cerebellum::compute (which uses CFsubcomplexALT instead),
// but ported for parity with the original module's public interface.
// ---------------------------------------------------------------------
class CFsubcomplex {
public:
    static constexpr int MAX_AMPLITUDE = 3;          // input_current >= 0.75
    static constexpr int MEDIUM_AMPLITUDE_UP = 3;     // 0.50 < input_current < 0.75
    static constexpr int MEDIUM_AMPLITUDE_DOWN = 2;   // 0.25 < input_current <= 0.50
    static constexpr int MIN_AMPLITUDE = 1;           // input_current <= 0.25
    static constexpr double MAX_SPIKE_FREQ = 10;

    explicit CFsubcomplex(int n_neurons = 50)
        : n_neurons(n_neurons),
          max_spk_freq(MAX_SPIKE_FREQ),
          spikes_pending(n_neurons, 0),
          last_spk_time(n_neurons, -std::numeric_limits<double>::infinity()),
          t(0.0),
          rng(std::random_device{}()),
          unif(0.0, 1.0) {}

    static int burst_size(double I) {
        if (I >= 0.75) return MAX_AMPLITUDE;
        if (I > 0.50) return MEDIUM_AMPLITUDE_UP;
        if (I > 0.25) return MEDIUM_AMPLITUDE_DOWN;
        return MIN_AMPLITUDE;
    }

    std::vector<uint8_t> step(double dt, double inp_I) {
        std::vector<uint8_t> spiked(n_neurons, 0);
        for (int i = 0; i < n_neurons; ++i) spiked[i] = spikes_pending[i] > 0;
        for (int i = 0; i < n_neurons; ++i)
            if (spiked[i]) spikes_pending[i] -= 1;

        std::vector<int> idle_idx;
        for (int i = 0; i < n_neurons; ++i)
            if (spikes_pending[i] == 0) idle_idx.push_back(i);

        if (!idle_idx.empty()) {
            std::vector<int> triggered_idx;
            for (int idx : idle_idx) {
                double readiness = (t - last_spk_time[idx]) * max_spk_freq;
                readiness = std::clamp(readiness, 0.0, 1.0);
                double p_trigger = readiness * inp_I * dt * max_spk_freq;
                double draw = unif(rng);
                if (p_trigger > draw) triggered_idx.push_back(idx);
            }
            if (!triggered_idx.empty()) {
                int num_spk = burst_size(inp_I);
                for (int idx : triggered_idx) {
                    spikes_pending[idx] = num_spk;
                    last_spk_time[idx] = t + (num_spk + 1) * dt;
                }
            }
        }

        t += dt;
        return spiked;
    }

    int n_neurons;
    double max_spk_freq;
    std::vector<int64_t> spikes_pending;
    std::vector<double> last_spk_time;
    double t;

private:
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> unif;
};

// ---------------------------------------------------------------------
// CFsubcomplexALT (from ROSPoissonGenerator.cpp). This is the CF model
// cerebellum::climbingFibers actually uses.
// ---------------------------------------------------------------------
class CFsubcomplexALT {
public:
    static constexpr double SIGMA = 1.0;
    static constexpr double MIN_SPIKE_FREQ = 1.0;
    static constexpr double MAX_SPIKE_FREQ = 10.0;
    static constexpr double MIN_ERROR = 0.001;
    static constexpr double MAX_ERROR = 0.01;

    explicit CFsubcomplexALT(int n_neurons = 50)
        : n_neurons(n_neurons),
          spikes_pending(n_neurons, 0),
          rng(std::random_device{}()),
          unif(0.0, 1.0) {
        centers.resize(n_neurons);
        widths.resize(n_neurons);
        if (n_neurons == 1) {
            centers[0] = MIN_ERROR;
        } else {
            for (int i = 0; i < n_neurons; ++i)
                centers[i] = MIN_ERROR + (MAX_ERROR - MIN_ERROR) * static_cast<double>(i) / (n_neurons - 1);
        }
        const double w = (n_neurons > 1) ? SIGMA * (MAX_ERROR - MIN_ERROR) / (n_neurons - 1) : 0.0;
        std::fill(widths.begin(), widths.end(), w);
    }

    std::vector<uint8_t> step(double dt, double inp_I) {
        std::vector<uint8_t> spiked(n_neurons, 0);
        for (int i = 0; i < n_neurons; ++i) spiked[i] = spikes_pending[i] > 0;
        for (int i = 0; i < n_neurons; ++i)
            if (spiked[i]) spikes_pending[i] -= 1;

        for (int idx = 0; idx < n_neurons; ++idx) {
            if (spikes_pending[idx] != 0) continue;  // only idle neurons are eligible
            const double norm_rof = (std::tanh((inp_I - centers[idx]) / widths[idx]) + 1.0) * 0.5;
            const double sp_rof = MIN_SPIKE_FREQ + norm_rof * (MAX_SPIKE_FREQ - MIN_SPIKE_FREQ);
            const double draw = unif(rng);
            if (sp_rof * dt >= draw) {
                const double val = inp_I / centers[idx];
                int64_t ival = static_cast<int64_t>(val);  // truncation toward zero, matches numpy's int64 cast
                ival = std::clamp<int64_t>(ival, 1, 6);
                spikes_pending[idx] = ival;
            }
        }

        return spiked;
    }

    int n_neurons;
    std::vector<double> centers;
    std::vector<double> widths;
    std::vector<int64_t> spikes_pending;

private:
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> unif;
};

// ---------------------------------------------------------------------
// MFsubcomplex. Instantiated by cerebellum but never actually stepped in
// the active (one-hot) code path -- only in the commented-out RBF-based
// MF implementation. Ported for parity.
// ---------------------------------------------------------------------
class MFsubcomplex {
public:
    static constexpr double SIGMA = 0.5;

    MFsubcomplex(double min_v, double max_v, int n_neurons = 10)
        : n_neurons(n_neurons) {
        centers.resize(n_neurons);
        widths.resize(n_neurons);
        for (int i = 0; i < n_neurons; ++i)
            centers[i] = (n_neurons > 1) ? min_v + (max_v - min_v) * static_cast<double>(i) / (n_neurons - 1) : min_v;
        const double w = (n_neurons > 1) ? SIGMA * (max_v - min_v) / (n_neurons - 1) : 0.0;
        std::fill(widths.begin(), widths.end(), w);
        spikes.assign(n_neurons, 0);
    }

    const std::vector<uint8_t>& step(double inp, double /*dt*/) {
        std::fill(spikes.begin(), spikes.end(), 0);
        bool any = false;
        for (int i = 0; i < n_neurons; ++i) {
            const double current = 1.0 - std::abs((inp - centers[i]) / widths[i]);
            if (current > 0.0) {
                spikes[i] = 1;
                any = true;
            }
        }
        if (!any) {
            if (inp > centers.back())
                spikes.back() = 1;
            else
                spikes.front() = 1;
        }
        return spikes;
    }

    int n_neurons;
    std::vector<double> centers;
    std::vector<double> widths;
    std::vector<uint8_t> spikes;
};

// ---------------------------------------------------------------------
// PFSpikeHistory: bounded, prunable record of recent PF (GC) spikes.
// ---------------------------------------------------------------------
class PFSpikeHistory {
public:
    static constexpr double HISTORY_PRUNE_WINDOW = 200.0;  // timesteps

    explicit PFSpikeHistory(double prune_window = HISTORY_PRUNE_WINDOW)
        : prune_window(prune_window) {}

    void record(const std::vector<int64_t>& indices, double t) {
        if (!indices.empty()) {
            idx_batches.push_back(indices);
            times.push_back(t);
        }
    }

    void prune(double current_t) {
        const double cutoff = current_t - prune_window;
        std::vector<std::vector<int64_t>> new_idx;
        std::vector<double> new_times;
        new_idx.reserve(idx_batches.size());
        new_times.reserve(times.size());
        for (std::size_t i = 0; i < times.size(); ++i) {
            if (times[i] >= cutoff) {
                new_idx.push_back(std::move(idx_batches[i]));
                new_times.push_back(times[i]);
            }
        }
        idx_batches = std::move(new_idx);
        times = std::move(new_times);
    }

    // Return (all_indices, all_spike_times) as parallel 1D arrays.
    void flatten(std::vector<int64_t>& out_idx, std::vector<double>& out_times) const {
        out_idx.clear();
        out_times.clear();
        for (std::size_t b = 0; b < idx_batches.size(); ++b) {
            for (int64_t v : idx_batches[b]) {
                out_idx.push_back(v);
                out_times.push_back(times[b]);
            }
        }
    }

    std::size_t size() const { return idx_batches.size(); }

    double prune_window;
    std::vector<std::vector<int64_t>> idx_batches;
    std::vector<double> times;
};

// ---------------------------------------------------------------------
// ltd_kernel (Eq. 13). x = t_PFspike - t_CFspike (x <= 0). Nonzero only
// for x < -dk; peaks at x = -tau_ltd with value 1.0; decays toward 0 for
// more negative x.
// ---------------------------------------------------------------------
inline constexpr double DK = 0.07;       // kernel width parameter (s)
inline constexpr double TAU_LTD = 0.1;   // kernel time constant (s)

inline double ltd_kernel(double x, double dk = DK, double tau_ltd = TAU_LTD) {
    if (x < -dk) {
        const double z = (x + dk) / (tau_ltd - dk);
        return -z * std::exp(z + 1.0);
    }
    return 0.0;
}
