#pragma once
// Direct ports of the small helper classes in garrido_brain_v02.py:
// CFsubcomplex, CFsubcomplexALT, MFsubcomplex, PFSpikeHistory, ltd_kernel.

#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <random>
#include <deque>
#include <optional>

// Python's round(): ties to even (std::nearbyint under the default rounding mode)
inline int64_t py_round(double x) { return static_cast<int64_t>(std::nearbyint(x)); }

// spacing of spikes inside a CF burst (was 1 tick @ 2 ms)
inline constexpr double CF_BURST_ISI_S = 0.002;

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
    static constexpr double MAX_ERROR = 0.1;

    explicit CFsubcomplexALT(int n_neurons = 50, std::optional<uint64_t> seed = std::nullopt)
        : n_neurons(n_neurons),
          spikes_pending(n_neurons, 0),
          wait(n_neurons, 0),
          spiked(n_neurons, 0),
          rng(seed ? *seed : std::random_device{}()),
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

    const std::vector<uint8_t>& step(double dt, double inp_I) {
        // spikes inside a burst are CF_BURST_ISI_S apart (1 tick at the legacy 2 ms)
        const int64_t isi_ticks = std::max<int64_t>(1, py_round(CF_BURST_ISI_S / dt));
        for (int i = 0; i < n_neurons; ++i) {
            const bool pending = spikes_pending[i] > 0;
            if (pending) wait[i] -= 1;
            spiked[i] = pending && wait[i] <= 0;
            if (spiked[i]) {
                spikes_pending[i] -= 1;
                wait[i] = isi_ticks;
            }
        }

        for (int idx = 0; idx < n_neurons; ++idx) {
            if (spikes_pending[idx] != 0) continue;  // only idle neurons are eligible
            const double norm_rof = (std::tanh((inp_I - centers[idx]) / widths[idx]) + 1.0) * 0.5;
            const double sp_rof = MIN_SPIKE_FREQ + norm_rof * (MAX_SPIKE_FREQ - MIN_SPIKE_FREQ);
            const double draw = draw_override ? *draw_override : unif(rng);
            if (sp_rof * dt >= draw) {
                // numpy's float -> int64 cast truncates toward zero
                const double val = std::clamp(inp_I / centers[idx], -1.0, 7.0);
                spikes_pending[idx] = std::clamp<int64_t>(static_cast<int64_t>(val), 1, 6);
                wait[idx] = 1;     // first spike of the burst on the next tick, as before
            }
        }

        return spiked;
    }

    int n_neurons;
    std::vector<double> centers;
    std::vector<double> widths;
    std::vector<int64_t> spikes_pending;
    std::vector<int64_t> wait;   // ticks until the next burst spike
    std::vector<uint8_t> spiked;
    // testing only: when set, every random draw returns this value instead
    std::optional<double> draw_override;

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
// PFSpikeHistory: bounded, prunable record of recent PF (GC) spikes, one
// batch per tick that had at least one PF spike. Times are integer ticks,
// recorded in non-decreasing order, so pruning pops from the front.
// ---------------------------------------------------------------------
class PFSpikeHistory {
public:
    // timesteps (legacy default; the brain passes round(PF_HISTORY_S/dt))
    static constexpr int64_t HISTORY_PRUNE_WINDOW = 200;

    explicit PFSpikeHistory(int64_t prune_window = HISTORY_PRUNE_WINDOW)
        : prune_window(prune_window) {}

    void record(const std::vector<int64_t>& indices, int64_t t) {
        if (!indices.empty()) {
            idx_batches.push_back(indices);
            times.push_back(t);
        }
    }

    void prune(int64_t current_t) {
        const int64_t cutoff = current_t - prune_window;
        while (!times.empty() && times.front() < cutoff) {
            times.pop_front();
            idx_batches.pop_front();
        }
    }

    // Return (all_indices, all_spike_times) as parallel 1D arrays.
    void flatten(std::vector<int64_t>& out_idx, std::vector<int64_t>& out_times) const {
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

    int64_t prune_window;
    std::deque<std::vector<int64_t>> idx_batches;
    std::deque<int64_t> times;
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
