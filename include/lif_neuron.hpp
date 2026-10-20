#pragma once
// Direct port of lif_neuron_v02.py
//
// dt-independent conductance-based LIF population (Eqs. 4-10 of Abadia et al. 2021):
//   - each synaptic conductance is integrated exactly over a tick (its tick mean is used),
//     and V is advanced with the closed-form solution for that frozen conductance, so it is
//     stable for any dt and every spike delivers exactly w*tau of conductance-time.
//   - `ampa_tonic` is a constant AMPA conductance that bypasses the spike/decay state.
//   - refractory handling is time-exact, unless legacy_refractory reproduces v01's rule.
// See the docstring at the top of lif_neuron_v02.py for the full rationale.

#include <vector>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include "neuron_params.hpp"

// Charge per spike delivered by the v01 update order (decay, add, forward Euler) at
// timestep dt0, divided by the exact w*tau:   dt0 / (tau * (1 - exp(-dt0/tau))).
// Multiply a spike-driven weight by this to reproduce the drive you had tuned at dt0.
inline double legacy_charge_gain(double tau, double dt0 = 2e-3) {
    return dt0 / (tau * (1.0 - std::exp(-dt0 / tau)));
}

// Mean conductance produced by injecting weight `w` EVERY tick in the v01 update order at
// dt0 (steady-state sawtooth value right after the add):  w / (1 - exp(-dt0/tau)).
// Pass this as `ampa_tonic` to reproduce the drive you had tuned at dt0.
inline double legacy_tonic_conductance(double w, double tau, double dt0 = 2e-3) {
    return w / (1.0 - std::exp(-dt0 / tau));
}

// Jahr-Stevens Mg block, Eq. (10): 1/(1 + 1.2/3.57 * exp(-62 V)), V in volts.
inline double nmda_inf(double V) {
    return 1.0 / (1.0 + std::exp(-62.0 * V) * (1.2 / 3.57));
}

// ---------------------------------------------------------------------
// Scalar reference neuron (lif_neuron_v02.py:lif_neuron). Not used by
// cerebellum (which only uses LIFPopulation); ported for parity.
// ---------------------------------------------------------------------
class LIFNeuron {
public:
    explicit LIFNeuron(const NeuronParams& params = NeuronParams())
        : p(params), V(params.EL) {}

    NeuronParams p;
    double V;
    double g_AMPA = 0.0;
    double g_NMDA = 0.0;
    double g_GABA = 0.0;
    double refact_t = 0.0;
    bool spiked = false;
    std::vector<double> spike_times;
    double t = 0.0;

    bool step(double dt = 2.0e-3,
              const std::vector<double>& ampa_weights = {},
              const std::vector<double>& nmda_weights = {},
              const std::vector<double>& gaba_weights = {}) {
        spiked = false;
        for (double w : ampa_weights) g_AMPA += w;
        for (double w : nmda_weights) g_NMDA += w;
        for (double w : gaba_weights) g_GABA += w;
        const double eA = std::exp(-dt / p.tau_AMPA);
        const double eN = std::exp(-dt / p.tau_NMDA);
        const double eG = std::exp(-dt / p.tau_GABA);
        const double gA = g_AMPA * p.tau_AMPA / dt * (1 - eA);
        const double gN = g_NMDA * p.tau_NMDA / dt * (1 - eN) * nmda_inf(V);
        const double gG = g_GABA * p.tau_GABA / dt * (1 - eG);
        g_AMPA *= eA; g_NMDA *= eN; g_GABA *= eG;
        if (refact_t > 1e-12) {
            refact_t = std::max(0.0, refact_t - dt);
            t += dt;
            return false;
        }
        const double G = p.gL + gA + gN + gG;
        const double Vinf = (p.gL * p.EL + (gA + gN) * p.E_AMPA + gG * p.E_GABA) / G;
        V = Vinf + (V - Vinf) * std::exp(-dt * G / p.Cm);
        if (V >= p.V_thr) {
            spiked = true;
            V = p.EL;
            refact_t = p.T_ref;
            spike_times.push_back(t);
        }
        t += dt;
        return spiked;
    }
};

// ---------------------------------------------------------------------
// Population of N neurons sharing one NeuronParams
// (lif_neuron_v02.py:LIFPopulation). This is the class cerebellum uses.
//
// step(dt, ampa_input, nmda_input, gaba_input, ampa_tonic)
//   *_input    : per-neuron summed synaptic weights of the spikes ARRIVING this tick (S).
//                Each spike delivers exactly w*tau of conductance-time, any dt.
//   ampa_tonic : per-neuron constant AMPA conductance (S) for this tick.
// Pass nullptr for any input to skip it (Python's None). Non-null pointers must
// point at n doubles. Returns the per-neuron spike flags for this tick.
// ---------------------------------------------------------------------
class LIFPopulation {
public:
    // refractory countdown is considered finished below this (s)
    static constexpr double REF_EPS = 1e-12;

    LIFPopulation(std::size_t n, const NeuronParams& params = NeuronParams(),
                  bool legacy_refractory = false)
        : n(n), p(params),
          // true reproduces v01's refractory rule (see step); only meaningful at the dt it was tuned at
          legacy_refractory(legacy_refractory),
          V(n, params.EL),
          g_AMPA(n, 0.0),
          g_NMDA(n, 0.0),
          g_GABA(n, 0.0),
          refractory_time_left(n, 0.0),
          spike_counts(n, 0),
          spiked(n, 0) {}

    std::size_t size() const { return n; }

    std::size_t n;
    NeuronParams p;
    bool legacy_refractory;

    std::vector<double> V;
    std::vector<double> g_AMPA;
    std::vector<double> g_NMDA;
    std::vector<double> g_GABA;
    std::vector<double> refractory_time_left;
    std::vector<int64_t> spike_counts;
    std::vector<uint8_t> spiked;
    double t = 0.0;

    const std::vector<uint8_t>& step(double dt,
                                     const double* ampa_input = nullptr,
                                     const double* nmda_input = nullptr,
                                     const double* gaba_input = nullptr,
                                     const double* ampa_tonic = nullptr) {
        if (dt != prepared_dt_) prepare(dt);

        for (std::size_t i = 0; i < n; ++i) {
            // Refractory: a neuron integrates only for the part of this tick that is left after
            // its refractory time.  (v01 decremented first and then tested, which blocked only
            // ceil(T_ref/dt)-1 whole ticks -- i.e. NO refractory at all whenever T_ref <= dt.)
            double t_act;
            if (legacy_refractory) {
                // v01 rule: whole ticks only, blocked while the countdown is still > 0 after decrementing
                double r = std::max(0.0, refractory_time_left[i] - dt);
                if (r <= REF_EPS) r = 0.0;
                refractory_time_left[i] = r;
                t_act = (r > 0.0) ? 0.0 : dt;
            } else {
                t_act = std::clamp(dt - refractory_time_left[i], 0.0, dt);
                refractory_time_left[i] = std::max(0.0, refractory_time_left[i] - dt);
            }
            const bool active = t_act > REF_EPS;

            // spikes arriving at the start of this tick (refractory neurons still receive them)
            if (ampa_input) g_AMPA[i] += ampa_input[i];
            if (nmda_input) g_NMDA[i] += nmda_input[i];
            if (gaba_input) g_GABA[i] += gaba_input[i];

            // mean conductances over the tick
            double gA = g_AMPA[i] * mean_ampa_;
            if (ampa_tonic) gA = gA + ampa_tonic[i];
            const double gN = g_NMDA[i] * mean_nmda_ * nmda_inf(V[i]);
            const double gG = g_GABA[i] * mean_gaba_;
            const double G = p.gL + gA + gN + gG;
            const double V_inf = (p.gL * p.EL + (gA + gN) * p.E_AMPA + gG * p.E_GABA) / G;
            // exact solution over the non-refractory part of the tick (t_act == 0 leaves V at reset)
            V[i] = V_inf + (V[i] - V_inf) * std::exp(-t_act * G / p.Cm);

            // conductances decay for the next tick
            g_AMPA[i] *= decay_ampa_;
            g_NMDA[i] *= decay_nmda_;
            g_GABA[i] *= decay_gaba_;

            const bool s = active && (V[i] >= p.V_thr);
            spiked[i] = s;
            if (s) {
                V[i] = p.EL;
                refractory_time_left[i] = p.T_ref;
                spike_counts[i] += 1;
            }
        }

        t += dt;
        return spiked;
    }

private:
    double prepared_dt_ = -1.0;
    double decay_ampa_ = 0.0, decay_nmda_ = 0.0, decay_gaba_ = 0.0;
    double mean_ampa_ = 0.0, mean_nmda_ = 0.0, mean_gaba_ = 0.0;

    void prepare(double dt) {
        prepared_dt_ = dt;
        decay_ampa_ = std::exp(-dt / p.tau_AMPA);
        decay_nmda_ = std::exp(-dt / p.tau_NMDA);
        decay_gaba_ = std::exp(-dt / p.tau_GABA);
        // mean of exp(-s/tau) over one tick
        mean_ampa_ = p.tau_AMPA / dt * (1.0 - decay_ampa_);
        mean_nmda_ = p.tau_NMDA / dt * (1.0 - decay_nmda_);
        mean_gaba_ = p.tau_GABA / dt * (1.0 - decay_gaba_);
    }
};
