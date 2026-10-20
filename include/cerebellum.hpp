#pragma once
// Direct port of garrido_brain_v02.py:cerebellum (plus BrainError and the
// per-population neuron parameter tables).
//
// This is a recreation of the cerebellar control described in:
// "On Robot Compliance: A Cerebellar Control Approach" -- Abadia, Naveros,
// Garrido, Ros, Luque.
//
// Differences from the Python, none of which change results:
//   - KEEP_LEGACY_DRIVE / KEEP_LEGACY_REFRACTORY are constructor arguments
//     instead of module-level globals.
//   - kp (climbing fibers) and TORQUE_ALPHA are per-joint vectors filled with
//     the Python values, so n_dof isn't limited to 2.
//   - an optional seed makes the climbing-fiber RNG reproducible.

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

#include "neuron_params.hpp"
#include "lif_neuron.hpp"
#include "subcomplexes.hpp"

class BrainError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// --- dt handling -------------------------------------------------------------------------
// Everything that used to be "N ticks" is now defined in seconds and converted with the
// brain's dt.  LEGACY_DT is the timestep the v01 constants (W_MF_GC*4, mf_to_dcn, TORQUE_ALPHA,
// alpha/beta, ...) were tuned at.
inline constexpr double LEGACY_DT = 2e-3;
inline constexpr double PF_HISTORY_S = 0.4;       // PF spike history kept for LTD (was 200 ticks @ 2 ms)
inline constexpr double TORQUE_FILTER_S = 0.030;  // mean-filter window (was 15 taps @ 2 ms)

// Neuron parameters from Table 2.
inline const NeuronParams GC_params{
    /*Cm=*/2.0e-12, /*gL=*/1.0e-9, /*EL=*/-65e-3, /*E_AMPA=*/0.0, /*E_GABA=*/-80e-3,
    /*V_thr=*/-50.0e-3, /*T_ref=*/1.0e-3, /*tau_AMPA=*/1.0e-3, /*tau_NMDA=*/20e-3, /*tau_GABA=*/5.0e-3
};
inline const NeuronParams PC_params{
    /*Cm=*/100.0e-12, /*gL=*/6.0e-9, /*EL=*/-70e-3, /*E_AMPA=*/0.0, /*E_GABA=*/-80e-3,
    /*V_thr=*/-52.0e-3, /*T_ref=*/2.0e-3, /*tau_AMPA=*/1.2e-3, /*tau_NMDA=*/20e-3, /*tau_GABA=*/5.0e-3
};
inline const NeuronParams DCN_params{
    /*Cm=*/2.0e-12, /*gL=*/0.2e-9, /*EL=*/-70e-3, /*E_AMPA=*/0.0, /*E_GABA=*/-80.0e-3,
    /*V_thr=*/-40.0e-3, /*T_ref=*/1.0e-3, /*tau_AMPA=*/0.5e-3, /*tau_NMDA=*/14.0e-3, /*tau_GABA=*/10.0e-3
};

class cerebellum {
public:
    static constexpr double ALPHA = 0.002e-9;          // (S)
    static constexpr double BETA = -0.001e-9;          // (S)
    static constexpr double INIT_PF_PC_WT = 1.6e-9;    // (S)
    static constexpr double W_MIN = 0.0;               // pf-pc weight lims (S)
    static constexpr double W_MAX = 5e-9;

    // these numbers are all PER JOINT
    static constexpr int nMF_subgroups = 2;
    static constexpr int nMF_per_subgroup = 20;
    static constexpr int nMF = nMF_per_subgroup * nMF_subgroups;
    static constexpr std::size_t nGC = nMF_per_subgroup * nMF_per_subgroup;  // nMF_per_subgroup ** nMF_subgroups
    static constexpr std::size_t nCF = 100;
    static constexpr std::size_t nPC = 100;
    static constexpr std::size_t nDCN = 100;

    static constexpr double W_MF_GC = 0.18e-9;
    static constexpr double W_PC_DCN = 1e-9;
    static constexpr double W_CF_DCN_AMPA = 0.5e-9;
    static constexpr double W_CF_DCN_NMDA = 0.25e-9;
    static constexpr double CF_KP = 3.0;

    static constexpr double effDelay = 50e-3;  // unused downstream, kept for parity
    static constexpr double affDelay = 50e-3;

    cerebellum(std::vector<double> qMins, std::vector<double> qMaxs,
               std::size_t n_dof = 6, double dt = LEGACY_DT,
               bool keep_legacy_drive = true, bool keep_legacy_refractory = true,
               std::optional<uint64_t> seed = std::nullopt)
        : nJoints(n_dof),
          dt(dt),
          keep_legacy_drive(keep_legacy_drive),
          keep_legacy_refractory(keep_legacy_refractory),
          qmins(std::move(qMins)), qmaxs(std::move(qMaxs)),
          gcNeurons(nGC * n_dof, GC_params, keep_legacy_refractory),
          pcNeurons(nPC * n_dof, PC_params, keep_legacy_refractory),
          dcnNeurons(nDCN * n_dof, DCN_params, keep_legacy_refractory),
          pf_history(py_round(PF_HISTORY_S / dt)) {
        if (qmins.size() != nJoints || qmaxs.size() != nJoints)
            throw BrainError("cerebellum: qMins and qMaxs must each have n_dof elements.");
        if (!(dt > 0.0))
            throw BrainError("cerebellum: dt must be positive.");

        // exact refractory blocks PCs for a whole 2 ms tick after each spike, so PCs inhibit the
        // DCN less and DCN output roughly doubles; these were retuned for that
        if (keep_legacy_refractory) {
            W_MF_DCN = 0.1e-9;
            TORQUE_ALPHA.assign(nJoints, 0.24);
        } else {
            W_MF_DCN = 0.2e-9;
            TORQUE_ALPHA.assign(nJoints, 0.12);
        }

        // Constant drives are tonic conductances (S), not an impulse injected every tick.
        // With keep_legacy_drive the values equal the mean conductance the v01 code produced
        // at 2 ms; otherwise they are the physical mean of one MF spike (weight W) every LEGACY_DT.
        if (keep_legacy_drive) {
            gc_tonic_value = legacy_tonic_conductance(W_MF_GC * 4, GC_params.tau_AMPA, LEGACY_DT);
            mf_dcn_value = legacy_tonic_conductance(W_MF_DCN, DCN_params.tau_AMPA, LEGACY_DT);
        } else {
            gc_tonic_value = W_MF_GC * 4 * GC_params.tau_AMPA / LEGACY_DT;
            mf_dcn_value = W_MF_DCN * DCN_params.tau_AMPA / LEGACY_DT;
        }
        mf_to_dcn.assign(nDCN * nJoints, mf_dcn_value);

        // per-spike charge compensation: the exact integrator delivers w*tau per spike, v01
        // delivered more at dt0 = 2 ms.  These gains restore the old drive.
        if (keep_legacy_drive) {
            gain_pc_ampa = legacy_charge_gain(PC_params.tau_AMPA, LEGACY_DT);
            gain_dcn_ampa = legacy_charge_gain(DCN_params.tau_AMPA, LEGACY_DT);
            gain_dcn_nmda = legacy_charge_gain(DCN_params.tau_NMDA, LEGACY_DT);
            gain_dcn_gaba = legacy_charge_gain(DCN_params.tau_GABA, LEGACY_DT);
        } else {
            gain_pc_ampa = gain_dcn_ampa = gain_dcn_nmda = gain_dcn_gaba = 1.0;
        }

        gc_tonic.assign(nGC * nJoints, 0.0);
        for (std::size_t j = 0; j < nJoints * 2; ++j) {
            std::optional<uint64_t> s;
            if (seed) s = *seed + j;
            CF_complexes.emplace_back(static_cast<int>(nCF / 2), s);
        }
        cf_output.assign(nCF * nJoints, 0);

        n_pc_total = nPC * nJoints;
        pf_pc_wts.assign(nGC * nJoints * n_pc_total, INIT_PF_PC_WT);
        pc_out.assign(n_pc_total, 0);
        pfs.assign(nGC * nJoints, 0);
        dcnOut.assign(nDCN * nJoints, 0);

        nTaps = static_cast<std::size_t>(std::max<int64_t>(1, py_round(TORQUE_FILTER_S / dt)));
        prevDCN.assign(nTaps * nJoints, 0.0);

        kernelLookup = makeKernelLookup();

        const std::size_t n_dcn_total = nDCN * nJoints;
        dcn_ampa_in.assign(n_dcn_total, 0.0);
        dcn_nmda_in.assign(n_dcn_total, 0.0);
        dcn_gaba_in.assign(n_dcn_total, 0.0);
        pc_ampa_inp.assign(n_pc_total, 0.0);
    }

    std::size_t nJoints;
    double dt;
    int64_t t = 0;  // tick counter; time in seconds is t * dt
    bool keep_legacy_drive;
    bool keep_legacy_refractory;

    double W_MF_DCN;
    std::vector<double> TORQUE_ALPHA;

    std::vector<double> qmins, qmaxs;

    double gc_tonic_value, mf_dcn_value;
    double gain_pc_ampa, gain_dcn_ampa, gain_dcn_nmda, gain_dcn_gaba;
    std::vector<double> mf_to_dcn;

    LIFPopulation gcNeurons;
    LIFPopulation pcNeurons;
    LIFPopulation dcnNeurons;
    std::vector<double> gc_tonic;
    PFSpikeHistory pf_history;

    std::vector<CFsubcomplexALT> CF_complexes;
    std::vector<uint8_t> cf_output;

    std::size_t n_pc_total;
    std::vector<double> pf_pc_wts;  // flat, row-major: [nGC*nJoints x n_pc_total]
    std::vector<uint8_t> pc_out;
    std::vector<uint8_t> pfs;       // last granular-layer output
    std::vector<uint8_t> dcnOut;    // last DCN output (kept for saving brain results)

    std::size_t nTaps;
    std::vector<double> prevDCN;    // flat, row-major: [nTaps x nJoints]

    std::vector<double> kernelLookup;  // kernelLookup[lag] = ltd_kernel(-lag*dt), lag = 0..window

    // testing only: fix every climbing-fiber random draw to `value` (nullopt restores the RNG)
    void set_cf_draw_override(std::optional<double> value) {
        for (auto& c : CF_complexes) c.draw_override = value;
    }

    std::vector<double> makeKernelLookup() const {
        const int64_t n_lag = pf_history.prune_window + 1;
        std::vector<double> table(static_cast<std::size_t>(n_lag));
        for (int64_t k = 0; k < n_lag; ++k)
            table[static_cast<std::size_t>(k)] = ltd_kernel(static_cast<double>(-k) * dt);
        return table;
    }

    // Convert (q, q_des) into 2 digits in [0, nMF_per_subgroup - 1], one per MF subgroup,
    // via uniform binning over [lo, hi].
    std::array<int, 2> encode_mf_address(const std::array<double, 2>& state,
                                          const std::array<std::pair<double, double>, 2>& value_ranges) const {
        std::array<int, 2> digits{};
        for (int i = 0; i < 2; ++i) {
            const double lo = value_ranges[i].first;
            const double hi = value_ranges[i].second;
            const double frac = (state[i] - lo) / (hi - lo);
            // int() truncates toward zero; clamping first keeps the cast defined for big values
            const double scaled = std::clamp(frac * nMF_per_subgroup, -1.0, static_cast<double>(nMF_per_subgroup));
            digits[i] = std::clamp(static_cast<int>(scaled), 0, nMF_per_subgroup - 1);
        }
        return digits;
    }

    // Combine 2 digits into a single GC index, offset into this joint's block of GCs.
    std::size_t address_to_gc_index(const std::array<int, 2>& digits, std::size_t nJoint) const {
        return static_cast<std::size_t>(digits[0] + digits[1] * nMF_per_subgroup) + nJoint * nGC;
    }

    // ONE-HOT IMPLEMENTATION (the RBF-based MF path in the Python is commented out and not ported)
    void granularLayer(const std::vector<double>& q, const std::vector<double>& qdes, double dt) {
        std::fill(gc_tonic.begin(), gc_tonic.end(), 0.0);
        for (std::size_t j = 0; j < nJoints; ++j) {
            const auto digits = encode_mf_address({q[j], qdes[j]},
                                                  {{{qmins[j], qmaxs[j]}, {qmins[j], qmaxs[j]}}});
            gc_tonic[address_to_gc_index(digits, j)] = gc_tonic_value;   // constant AMPA conductance (S)
        }
        pfs = gcNeurons.step(dt, nullptr, nullptr, nullptr, gc_tonic.data());
    }

    static double errorCalc(double error) {
        const double max_error = 1.0;
        return 0.2 + 0.8 * (1.0 - std::exp(-error * 90.0 / max_error));
    }

    void climbingFibers(const std::vector<double>& qErr, double dt) {
        std::fill(cf_output.begin(), cf_output.end(), 0);
        const std::size_t half = nCF / 2;
        for (std::size_t j = 0; j < nJoints; ++j) {
            const double sigError = CF_KP * qErr[j];
            // positive error -> agonist microzone (DCN half that pushes +)
            const double epsAgon = (sigError >= 0.0) ? sigError : 0.0;
            const double epsAAgon = (sigError < 0.0) ? -sigError : 0.0;
            const std::size_t startIdx = j * nCF;
            const auto& agon = CF_complexes[j * 2].step(dt, epsAgon);
            std::copy(agon.begin(), agon.end(), cf_output.begin() + startIdx);
            const auto& antagon = CF_complexes[j * 2 + 1].step(dt, epsAAgon);
            std::copy(antagon.begin(), antagon.end(), cf_output.begin() + startIdx + half);
        }
    }

    // Advance PF-PC weights by one timestep's worth of LTP + LTD.
    // pf_spike_idx: flat GC/PF indices that fired THIS timestep (unique, ascending).
    // cf_spike_idx: flat CF indices that fired THIS timestep -- equal to the PC column index
    //   each maps to (one-to-one CF-PC connectivity per Table I).
    void update_pf_pc_weights(const std::vector<int64_t>& pf_spike_idx,
                               const std::vector<int64_t>& cf_spike_idx,
                               int64_t current_t) {
        // --- LTP: fixed jump of ALPHA at every PF spike, applied to ALL PC columns
        //     (every PF connects to every PC). No dt factor -- a discrete per-spike jump. ---
        for (int64_t row : pf_spike_idx) {
            double* row_ptr = &pf_pc_wts[static_cast<std::size_t>(row) * n_pc_total];
            for (std::size_t c = 0; c < n_pc_total; ++c)
                row_ptr[c] = std::clamp(row_ptr[c] + ALPHA, W_MIN, W_MAX);
        }

        // --- Record this step's PF spikes for future LTD lookups, then prune
        //     anything old enough to be kernel-negligible. ---
        pf_history.record(pf_spike_idx, current_t);
        pf_history.prune(current_t);

        // --- LTD: every CF spike this step looks back at PF history and applies a
        //     kernel-weighted decrement, restricted to that CF's own PC column.
        //     Every history entry contributes its own term (same GC can appear more than
        //     once), applied in the same order as the Python's np.add.at, then clipped. ---
        if (cf_spike_idx.empty()) return;
        pf_history.flatten(hist_idx_, hist_time_);
        touched_rows_.clear();
        for (std::size_t i = 0; i < hist_idx_.size(); ++i) {
            const int64_t lag = current_t - hist_time_[i];   // ticks since that PF spike, 0..window
            const double k = kernelLookup[static_cast<std::size_t>(lag)];
            if (k == 0.0) continue;
            const int64_t row = hist_idx_[i];
            touched_rows_.push_back(row);
            const double v = BETA * k;
            double* row_ptr = &pf_pc_wts[static_cast<std::size_t>(row) * n_pc_total];
            for (int64_t col : cf_spike_idx) row_ptr[col] += v;
        }
        for (int64_t row : touched_rows_) {
            double* row_ptr = &pf_pc_wts[static_cast<std::size_t>(row) * n_pc_total];
            for (int64_t col : cf_spike_idx) row_ptr[col] = std::clamp(row_ptr[col], W_MIN, W_MAX);
        }
    }

    void PCstep(double dt) {
        pf_idx_.clear();
        cf_idx_.clear();
        for (std::size_t i = 0; i < pfs.size(); ++i)
            if (pfs[i]) pf_idx_.push_back(static_cast<int64_t>(i));
        for (std::size_t i = 0; i < cf_output.size(); ++i)
            if (cf_output[i]) cf_idx_.push_back(static_cast<int64_t>(i));

        update_pf_pc_weights(pf_idx_, cf_idx_, t);

        std::fill(pc_ampa_inp.begin(), pc_ampa_inp.end(), 0.0);
        for (int64_t row : pf_idx_) {
            const double* row_ptr = &pf_pc_wts[static_cast<std::size_t>(row) * n_pc_total];
            for (std::size_t c = 0; c < n_pc_total; ++c) pc_ampa_inp[c] += row_ptr[c];
        }
        for (double& v : pc_ampa_inp) v = gain_pc_ampa * v;
        pc_out = pcNeurons.step(dt, pc_ampa_inp.data());
    }

    std::vector<double> dcnToTorque(const std::vector<uint8_t>& dcn, double dt) {
        const std::size_t rows = nJoints * 2;
        const std::size_t cols = dcn.size() / rows;

        std::vector<double> joint_torques(nJoints, 0.0);
        for (std::size_t j = 0; j < nJoints; ++j) {
            double side[2];
            for (std::size_t s = 0; s < 2; ++s) {
                int64_t count = 0;
                const std::size_t r = 2 * j + s;
                for (std::size_t c = 0; c < cols; ++c) count += dcn[r * cols + c] ? 1 : 0;
                // spikes per tick shrink as dt shrinks; convert to a rate and re-express per
                // legacy tick so TORQUE_ALPHA means the same thing at any dt.
                side[s] = static_cast<double>(count) * (LEGACY_DT / dt);
            }
            joint_torques[j] = side[0] + (-side[1]);
        }

        // prevDCN[:-1] = prevDCN[1:]; prevDCN[-1] = torques
        std::copy(prevDCN.begin() + nJoints, prevDCN.end(), prevDCN.begin());
        std::copy(joint_torques.begin(), joint_torques.end(), prevDCN.end() - nJoints);

        std::vector<double> corr(nJoints);
        for (std::size_t j = 0; j < nJoints; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < nTaps; ++i) s += prevDCN[i * nJoints + j];
            corr[j] = TORQUE_ALPHA[j] * (s / static_cast<double>(nTaps));
        }
        return corr;
    }

    // dt is fixed at construction (kernel table, history window, filter taps depend on it)
    std::vector<double> compute(const std::vector<double>& q, const std::vector<double>& qdes,
                                 const std::vector<double>& qErr,
                                 std::optional<double> step_dt = std::nullopt) {
        if (step_dt && std::abs(*step_dt - dt) > 1e-12)
            throw BrainError("compute(dt=" + std::to_string(*step_dt) + ") != brain dt=" +
                             std::to_string(dt) + "; build cerebellum(..., dt=dt)");
        if (q.size() != nJoints || qdes.size() != nJoints || qErr.size() != nJoints)
            throw BrainError("compute: q, qdes and qErr must each have n_dof elements.");

        // input fiber layers
        granularLayer(q, qdes, dt);
        climbingFibers(qErr, dt);
        // now we have spikes from the PFs and the CFs in cf_output and pfs
        PCstep(dt);

        const double cf_ampa = gain_dcn_ampa * W_CF_DCN_AMPA;
        const double cf_nmda = gain_dcn_nmda * W_CF_DCN_NMDA;
        for (std::size_t i = 0; i < dcn_ampa_in.size(); ++i) {
            const double cf = cf_output[i] ? 1.0 : 0.0;
            dcn_ampa_in[i] = cf_ampa * cf;
            dcn_nmda_in[i] = cf_nmda * cf;
            dcn_gaba_in[i] = (gain_dcn_gaba * (pc_out[i] ? 1.0 : 0.0)) * W_PC_DCN;
        }
        dcnOut = dcnNeurons.step(dt, dcn_ampa_in.data(), dcn_nmda_in.data(), dcn_gaba_in.data(),
                                 mf_to_dcn.data());
        auto corr = dcnToTorque(dcnOut, dt);
        t += 1;
        return corr;
    }

private:
    // scratch buffers reused every tick
    std::vector<double> dcn_ampa_in, dcn_nmda_in, dcn_gaba_in, pc_ampa_inp;
    std::vector<int64_t> pf_idx_, cf_idx_, hist_idx_, hist_time_, touched_rows_;
};
