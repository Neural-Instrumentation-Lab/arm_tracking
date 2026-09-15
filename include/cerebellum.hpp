#pragma once
// Direct port of garrido_brain_v01.py:cerebellum (plus BrainError, the
// per-population neuron parameter tables, and restrictAngle).
//
// This is a recreation of the cerebellar control described in:
// "On Robot Compliance: A Cerebellar Control Approach" -- Abadia, Naveros,
// Garrido, Ros, Luque.

#include <vector>
#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "neuron_params.hpp"
#include "lif_neuron.hpp"
#include "subcomplexes.hpp"
#include "arm_assets.hpp"

class BrainError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

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
    // --- class-level constants (cerebellum.ALPHA / .BETA / ... in Python) ---
    static constexpr double ALPHA = 0.002e-9;          // (S)
    static constexpr double BETA = -0.001e-9;          // (S)
    static constexpr double INIT_PF_PC_WT = 1.6e-9;    // (S)
    static constexpr double W_MIN = 0.0;               // pf-pc weight lims (S)
    static constexpr double W_MAX = 5e-9;

    // Signature order matches the Python constructor exactly:
    // __init__(self, qMins, qdMins, qMaxs, qdMaxs, n_dof=6)
    cerebellum(std::vector<double> qMins, std::vector<double> qdMins,
               std::vector<double> qMaxs, std::vector<double> qdMaxs,
               std::size_t n_dof = 6)
        : nJoints(n_dof),
          t(0),
          qmins(std::move(qMins)), qdmins(std::move(qdMins)),
          qmaxs(std::move(qMaxs)), qdmaxs(std::move(qdMaxs)),
          gcNeurons(nGC * n_dof, GC_params),
          gc_ampa_input(nGC * n_dof, 0.0),
          cf_output(nCF * n_dof, 0),
          n_pc_total(nPC * n_dof),
          pf_pc_wts(static_cast<std::size_t>(nGC * n_dof) * (nPC * n_dof), INIT_PF_PC_WT),
          pc_out(nPC * n_dof, 0),
          pcNeurons(nPC * n_dof, PC_params),
          dcnNeurons(nDCN * n_dof, DCN_params),
          prevDCN(15 * n_dof, 0.0),
          pfs(nGC * n_dof, 0) {
        // The Python source hardcodes TORQUE_ALPHA and the climbing-fiber
        // kp/kd gains as length-6 lists -- i.e. it implicitly assumes
        // n_dof == 6 (a 6-DOF arm). Rather than silently reading out of
        // bounds (undefined behavior in C++, unlike numpy's broadcasting
        // error), we check that assumption explicitly here.
        if (nJoints != 6) {
            throw BrainError(
                "cerebellum: n_dof must be 6 -- the original model hardcodes "
                "TORQUE_ALPHA and the climbing-fiber kp/kd gains for a 6-DOF arm.");
        }
        if (qmins.size() != nJoints || qmaxs.size() != nJoints ||
            qdmins.size() != nJoints || qdmaxs.size() != nJoints) {
            throw BrainError("cerebellum: qMins/qdMins/qMaxs/qdMaxs must each have n_dof elements.");
        }

        for (std::size_t j = 0; j < nJoints; ++j) {
            MF_complexesQ.emplace_back(qmins[j], qmaxs[j], nMF_per_subgroup);
            MF_complexesQdes.emplace_back(qmins[j], qmaxs[j], nMF_per_subgroup);
            MF_complexesQd.emplace_back(qdmins[j], qdmaxs[j], nMF_per_subgroup);
            MF_complexesQddes.emplace_back(qdmins[j], qdmaxs[j], nMF_per_subgroup);
        }
        mf_out.assign(nMF * nJoints, 0.0);

        for (std::size_t j = 0; j < nJoints * 2; ++j)
            CF_complexes.emplace_back(static_cast<int>(nCF / 2));

        mf_to_dcn.assign(nDCN * nJoints, W_MF_DCN);

        kernelLookup = makeKernelLookup();
    }

    // ---- per-instance "constants" set in __init__ ----
    std::size_t nJoints;
    int64_t t;  // step counter (matches Python's int self.t, incremented by 1 per compute() call)

    static constexpr int nMF_subgroups = 4;
    static constexpr int nMF_per_subgroup = 10;
    static constexpr int nMF = nMF_per_subgroup * nMF_subgroups;

    static constexpr double W_MF_GC = 0.18e-9;
    static constexpr double W_PC_DCN = 1e-9;
    static constexpr double W_MF_DCN = 0.1e-9;
    static constexpr double W_CF_DCN_AMPA = 0.5e-9;
    static constexpr double W_CF_DCN_NMDA = 0.25e-9;

    // TORQUE_ALPHA is a per-instance list in Python; here it's a fixed
    // 6-element array (n_dof is required to be 6 -- see constructor).
    static constexpr std::array<double, 6> TORQUE_ALPHA = {0.75, 3.0, 0.375, 1.5, 0.05, 0.05};

    static constexpr std::size_t nGC = 10000;   // nMF_per_subgroup ** nMF_subgroups
    static constexpr std::size_t nCF = 100;
    static constexpr std::size_t nPC = 100;
    static constexpr std::size_t nDCN = 100;

    std::vector<double> qmins, qdmins, qmaxs, qdmaxs;

    static constexpr double effDelay = 50e-3;  // unused downstream, kept for parity
    static constexpr double affDelay = 50e-3;

    std::vector<double> mf_to_dcn;

    LIFPopulation gcNeurons;
    std::vector<double> gc_ampa_input;
    PFSpikeHistory pf_history;

    std::vector<CFsubcomplexALT> CF_complexes;
    std::vector<uint8_t> cf_output;

    std::vector<MFsubcomplex> MF_complexesQ, MF_complexesQdes, MF_complexesQd, MF_complexesQddes;
    std::vector<double> mf_out;

    std::size_t n_pc_total;
    std::vector<double> pf_pc_wts;  // flat, row-major: [nGC*nJoints x n_pc_total]
    std::vector<uint8_t> pc_out;

    LIFPopulation pcNeurons;
    LIFPopulation dcnNeurons;

    std::vector<double> prevDCN;  // flat, row-major: [15 x nJoints]

    static constexpr double timeStep = 2e-3;

    std::array<double, 201> kernelLookup{};  // kernelLookup[k] == python's kernelLookup[-k], k in [0,200]

    std::vector<uint8_t> pfs;  // last granular-layer output (self.pfs in Python)

    std::array<double, 201> makeKernelLookup() const {
        std::array<double, 201> table{};
        for (int k = 0; k <= 200; ++k) {
            const double x = static_cast<double>(-k);
            table[k] = ltd_kernel(x * timeStep);
        }
        return table;
    }

    // x is expected to be an exact integer number of timesteps <= 0 (a
    // difference of two step counters), matching the Python dict lookup
    // self.kernelLookup[x_val] for x_val in {0, -1, ..., -200}.
    double kernelLookupAt(double x) const {
        long k = std::lround(-x);
        if (k < 0) k = 0;
        if (k > 200) k = 200;  // shouldn't happen given the 200-step prune window
        return kernelLookup[static_cast<std::size_t>(k)];
    }

    // Convert (q, qd, q_des, qd_des) into 4 digits in [0, 9], one per MF
    // subgroup, via uniform binning.
    std::array<int, 4> encode_mf_address(const std::array<double, 4>& state,
                                          const std::array<std::pair<double, double>, 4>& value_ranges) const {
        std::array<int, 4> digits{};
        for (int i = 0; i < 4; ++i) {
            const double value = state[i];
            const double lo = value_ranges[i].first;
            const double hi = value_ranges[i].second;
            const double frac = (value - lo) / (hi - lo);
            int digit = static_cast<int>(frac * nMF_per_subgroup);
            digit = std::clamp(digit, 0, nMF_per_subgroup - 1);
            digits[i] = digit;
        }
        return digits;
    }

    // Combine 4 base-10 digits into a single GC index in [0, 9999], offset
    // into this joint's block of the flat GC population.
    int64_t address_to_gc_index(const std::array<int, 4>& digits, std::size_t nJoint) const {
        const int d0 = digits[0], d1 = digits[1], d2 = digits[2], d3 = digits[3];
        return (d0 + d1 * nMF_per_subgroup
                   + d2 * nMF_per_subgroup * nMF_per_subgroup
                   + d3 * nMF_per_subgroup * nMF_per_subgroup * nMF_per_subgroup)
               + static_cast<int64_t>(nJoint) * static_cast<int64_t>(nGC);
    }

    // "ONE-HOT IMPLEMENTATION" -- the RBF-based MF path in the Python
    // source is commented out ("NOT WORKING YET") and is not ported here.
    void granularLayer(const std::vector<double>& q, const std::vector<double>& qd,
                        const std::vector<double>& qdes, const std::vector<double>& qddes,
                        double dt) {
        std::fill(gc_ampa_input.begin(), gc_ampa_input.end(), 0.0);

        for (std::size_t j = 0; j < nJoints; ++j) {
            std::array<double, 4> state{q[j], qd[j], qdes[j], qddes[j]};
            std::array<std::pair<double, double>, 4> ranges{{
                {qmins[j], qmaxs[j]},
                {qdmins[j], qdmaxs[j]},
                {qmins[j], qmaxs[j]},
                {qdmins[j], qdmaxs[j]},
            }};
            const auto digits = encode_mf_address(state, ranges);
            const int64_t addressed_index = address_to_gc_index(digits, j);
            gc_ampa_input[static_cast<std::size_t>(addressed_index)] = W_MF_GC * nMF_subgroups;
        }

        pfs = gcNeurons.step(dt, gc_ampa_input);
    }

    static double errorCalc(double error) {
        const double max_error = 1.0;
        return 0.2 + 0.8 * (1.0 - std::exp(-error * 90.0 / max_error));
    }

    void climbingFibers(const std::vector<double>& qErr, const std::vector<double>& qdErr, double dt) {
        std::fill(cf_output.begin(), cf_output.end(), 0);

        static constexpr std::array<double, 6> kp = {1.5, 2, 3, 2, 3, 3};
        static constexpr std::array<double, 6> kd = {1.5, 1, 3, 1, 3, 0.5};

        std::vector<double> agonist(nJoints), antagonist(nJoints);
        for (std::size_t j = 0; j < nJoints; ++j) {
            const double sigError = kp[j] * qErr[j] + kd[j] * qdErr[j];
            const double pError = sigError;
            const double nError = -sigError;
            agonist[j] = (sigError < 0.0) ? nError : 0.0;
            antagonist[j] = (sigError >= 0.0) ? pError : 0.0;
        }

        for (std::size_t j = 0; j < nJoints; ++j) {
            const double epsAgon = agonist[j];
            const double epsAAgon = antagonist[j];
            const std::size_t startIdx = j * nCF;
            const std::size_t half = nCF / 2;

            auto spiked_agon = CF_complexes[j * 2].step(dt, epsAgon);
            auto spiked_antagon = CF_complexes[j * 2 + 1].step(dt, epsAAgon);
            std::copy(spiked_agon.begin(), spiked_agon.end(), cf_output.begin() + startIdx);
            std::copy(spiked_antagon.begin(), spiked_antagon.end(), cf_output.begin() + startIdx + half);
        }
    }

    // Advance PF-PC weights by one timestep's worth of LTP + LTD.
    // pf_spike_idx: flat GC/PF indices that fired THIS timestep.
    // cf_spike_idx: flat CF indices that fired THIS timestep -- assumed to
    //   equal the PC column index each maps to (one-to-one CF-PC
    //   connectivity per Table I).
    void update_pf_pc_weights(const std::vector<int64_t>& pf_spike_idx,
                               const std::vector<int64_t>& cf_spike_idx,
                               double current_t) {
        // --- LTP: fixed jump of ALPHA at every PF spike, applied to ALL PC
        //     columns (every PF connects to every PC). No dt factor -- this
        //     is a discrete per-spike jump, not a rate. ---
        //
        // NOTE: Python does `pf_pc_wts[pf_spike_idx, :] += ALPHA` via numpy
        // fancy indexing. If pf_spike_idx contained a duplicate row, numpy's
        // buffered `a[idx] += x` applies the addition only ONCE per row
        // regardless of how many times it appears (unlike np.add.at, which
        // *would* accumulate once per occurrence). In real use pf_spike_idx
        // always comes from np.nonzero(...), so it's already duplicate-free
        // -- but we dedupe defensively here to match numpy's semantics
        // exactly even if this method is ever called with duplicates.
        if (!pf_spike_idx.empty()) {
            std::vector<int64_t> unique_rows(pf_spike_idx.begin(), pf_spike_idx.end());
            std::sort(unique_rows.begin(), unique_rows.end());
            unique_rows.erase(std::unique(unique_rows.begin(), unique_rows.end()), unique_rows.end());
            for (int64_t row : unique_rows) {
                double* row_ptr = &pf_pc_wts[static_cast<std::size_t>(row) * n_pc_total];
                for (std::size_t c = 0; c < n_pc_total; ++c)
                    row_ptr[c] = std::clamp(row_ptr[c] + ALPHA, W_MIN, W_MAX);
            }
        }

        // --- Record this step's PF spikes for future LTD lookups, then
        //     prune anything old enough to be kernel-negligible. ---
        pf_history.record(pf_spike_idx, current_t);
        pf_history.prune(current_t);

        // --- LTD: every CF spike this step looks back at PF history and
        //     applies a kernel-weighted decrement, restricted to that CF's
        //     own PC column. ---
        if (!cf_spike_idx.empty()) {
            std::vector<int64_t> hist_idx;
            std::vector<double> hist_time;
            pf_history.flatten(hist_idx, hist_time);
            if (!hist_idx.empty()) {
                std::set<int64_t> touched_rows;
                for (std::size_t i = 0; i < hist_idx.size(); ++i) {
                    const double x = hist_time[i] - current_t;
                    const double k = kernelLookupAt(x);
                    if (k != 0.0) {
                        const int64_t row = hist_idx[i];
                        touched_rows.insert(row);
                        const double v = BETA * k;
                        double* row_ptr = &pf_pc_wts[static_cast<std::size_t>(row) * n_pc_total];
                        for (int64_t col : cf_spike_idx) row_ptr[col] += v;
                    }
                }
                for (int64_t row : touched_rows) {
                    double* row_ptr = &pf_pc_wts[static_cast<std::size_t>(row) * n_pc_total];
                    for (int64_t col : cf_spike_idx) row_ptr[col] = std::clamp(row_ptr[col], W_MIN, W_MAX);
                }
            }
        }
    }

    void PCstep(double dt) {
        std::vector<int64_t> pf_idx, cf_idx;
        for (std::size_t i = 0; i < pfs.size(); ++i)
            if (pfs[i]) pf_idx.push_back(static_cast<int64_t>(i));
        for (std::size_t i = 0; i < cf_output.size(); ++i)
            if (cf_output[i]) cf_idx.push_back(static_cast<int64_t>(i));

        update_pf_pc_weights(pf_idx, cf_idx, static_cast<double>(t));

        std::vector<double> pc_ampa_inp(n_pc_total, 0.0);
        for (int64_t row : pf_idx) {
            const double* row_ptr = &pf_pc_wts[static_cast<std::size_t>(row) * n_pc_total];
            for (std::size_t c = 0; c < n_pc_total; ++c) pc_ampa_inp[c] += row_ptr[c];
        }
        pc_out = pcNeurons.step(dt, pc_ampa_inp);
    }

    std::vector<double> dcnToTorque(const std::vector<uint8_t>& dcnOut, double /*dt*/) {
        const std::size_t rows = nJoints * 2;
        const std::size_t cols = dcnOut.size() / rows;

        std::vector<double> torques(rows, 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            double s = 0.0;
            for (std::size_t c = 0; c < cols; ++c) s += dcnOut[r * cols + c] ? 1.0 : 0.0;
            torques[r] = s;
        }
        for (std::size_t r = 1; r < rows; r += 2) torques[r] *= -1.0;

        std::vector<double> joint_torques(nJoints, 0.0);
        for (std::size_t j = 0; j < nJoints; ++j)
            joint_torques[j] = torques[2 * j] + torques[2 * j + 1];

        // prevDCN[:-1] = prevDCN[1:]; prevDCN[-1] = torques
        for (std::size_t i = 0; i + 1 < 15; ++i)
            for (std::size_t j = 0; j < nJoints; ++j)
                prevDCN[i * nJoints + j] = prevDCN[(i + 1) * nJoints + j];
        for (std::size_t j = 0; j < nJoints; ++j) prevDCN[14 * nJoints + j] = joint_torques[j];

        std::vector<double> mean_prev(nJoints, 0.0);
        for (std::size_t j = 0; j < nJoints; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < 15; ++i) s += prevDCN[i * nJoints + j];
            mean_prev[j] = s / 15.0;
        }

        std::vector<double> corr(nJoints);
        for (std::size_t j = 0; j < nJoints; ++j) corr[j] = TORQUE_ALPHA[j] * mean_prev[j];
        return corr;
    }

    std::vector<double> compute(const std::vector<double>& q, const std::vector<double>& qd,
                                 const std::vector<double>& qdes, const std::vector<double>& qddes,
                                 const std::vector<double>& qErr, const std::vector<double>& qdErr,
                                 double dt = 2e-3) {
        // input fiber layers
        granularLayer(restrictAngle(q), qd, restrictAngle(qdes), qddes, dt);
        climbingFibers(qErr, qdErr, dt);
        // now we have spikes from the PFs and the CFs in cf_output and pfs
        PCstep(dt);

        const std::size_t n_dcn_total = dcnNeurons.size();
        std::vector<double> ampa_in(n_dcn_total), nmda_in(n_dcn_total), gaba_in(n_dcn_total);
        for (std::size_t i = 0; i < n_dcn_total; ++i) {
            const double cf = cf_output[i] ? 1.0 : 0.0;
            ampa_in[i] = W_CF_DCN_AMPA * cf + mf_to_dcn[i];
            nmda_in[i] = W_CF_DCN_NMDA * cf;
        }
        for (std::size_t i = 0; i < n_dcn_total; ++i)
            gaba_in[i] = (pc_out[i] ? 1.0 : 0.0) * W_PC_DCN;

        auto dcnOut = dcnNeurons.step(dt, ampa_in, nmda_in, gaba_in);
        // (Python checks `if dcnOut[150:199].any(): pass` here -- a no-op
        // left over from debugging. Omitted as genuinely dead code.)
        auto corr = dcnToTorque(dcnOut, dt);
        t += 1;
        return corr;
    }
};
