#pragma once
// Direct port of lif_neuron_v01.py

#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <optional>
#include "neuron_params.hpp"

// ---------------------------------------------------------------------
// Scalar single-neuron model (lif_neuron_v01.py:lif_neuron).
// Not used by cerebellum (which only ever uses LIFPopulation), but ported
// for parity with the original module's public interface.
//
// NOTE (preserved quirk, not "fixed"): the scalar class's NMDA activation
// gate uses a different algebraic form than LIFPopulation's:
//   scalar:      (1 / (1 + exp( 62*V))) * (1.2/3.57)
//   population:   1 / (1 + exp(-62*V)  * (1.2/3.57))
// These are NOT equivalent. This mirrors the Python source exactly -- see
// LIFNeuron::_nmda_inf vs LIFPopulation::_nmda_inf below.
// ---------------------------------------------------------------------
class LIFNeuron {
public:
    explicit LIFNeuron(const NeuronParams& params = NeuronParams())
        : p(params), V(params.EL), g_AMPA(0.0), g_NMDA(0.0), g_GABA(0.0),
          refact_t(0.0), spiked(false), t(0.0) {}

    NeuronParams p;

    // State variables
    double V;
    double g_AMPA;
    double g_NMDA;
    double g_GABA;
    double refact_t;
    bool spiked;

    // Bookkeeping
    std::vector<double> spike_times;
    double t;

    double _nmda_inf() const {
        return (1.0 / (1.0 + std::exp(62.0 * V))) * (1.2 / 3.57);
    }

    void _decay_conductances(double dt = 2e-3) {
        g_AMPA *= std::exp(-dt / p.tau_AMPA);
        g_NMDA *= std::exp(-dt / p.tau_NMDA);
        g_GABA *= std::exp(-dt / p.tau_GABA);
    }

    void _apply_synaptic_inputs(const std::vector<double>& ampa_weights,
                                 const std::vector<double>& nmda_weights,
                                 const std::vector<double>& gaba_weights) {
        for (double w : ampa_weights) g_AMPA += w;
        for (double w : nmda_weights) g_NMDA += w;
        for (double w : gaba_weights) g_GABA += w;
    }

    bool step(double dt = 2.0e-3,
              const std::vector<double>& ampa_weights = {},
              const std::vector<double>& nmda_weights = {},
              const std::vector<double>& gaba_weights = {}) {
        spiked = false;

        // Refractory period: membrane potential held at reset, no integration
        if (refact_t > 0.0) {
            refact_t = std::max(0.0, refact_t - dt);
            _decay_conductances(dt);
            _apply_synaptic_inputs(ampa_weights, nmda_weights, gaba_weights);
            t += dt;
            return false;
        }

        // 1: update conductances
        _decay_conductances(dt);
        _apply_synaptic_inputs(ampa_weights, nmda_weights, gaba_weights);

        // compute currents
        double I_int = -p.gL * (V - p.EL);
        double g_nmda_inf = _nmda_inf();
        double I_ext = -(g_AMPA + g_NMDA * g_nmda_inf) * (V - p.E_AMPA)
                       - g_GABA * (V - p.E_GABA);

        // calculate membrane potential
        double dV = (I_int + I_ext) / p.Cm * dt;
        V += dV;

        // see if spiked
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
// Vectorized population of N LIF neurons sharing one NeuronParams
// (lif_neuron_v01.py:LIFPopulation). This is the class cerebellum.py
// actually uses.
//
// Synaptic inputs (ampa_input / nmda_input / gaba_input) are per-neuron
// totals for this timestep -- already summed across any coincident
// presynaptic spikes, matching the numpy version's calling convention.
// Passing nullptr is equivalent to Python's `None` (skip that receptor
// type entirely, same as all-zero input but slightly cheaper).
// ---------------------------------------------------------------------
class LIFPopulation {
public:
    LIFPopulation(std::size_t n, const NeuronParams& params = NeuronParams())
        : n(n), p(params),
          V(n, params.EL),
          g_AMPA(n, 0.0),
          g_NMDA(n, 0.0),
          g_GABA(n, 0.0),
          refractory_time_left(n, 0.0),
          spike_counts(n, 0),
          t(0.0),
          spiked(n, 0) {}

    std::size_t size() const { return n; }

    std::size_t n;
    NeuronParams p;

    std::vector<double> V;
    std::vector<double> g_AMPA;
    std::vector<double> g_NMDA;
    std::vector<double> g_GABA;
    std::vector<double> refractory_time_left;
    std::vector<int64_t> spike_counts;
    double t;
    std::vector<uint8_t> spiked;  // last tick's spike flags (reused buffer, returned by step)

    static double _nmda_inf_scalar(double v) {
        // Eq. (10), vectorized version's formula (differs from the scalar
        // LIFNeuron class -- see note above LIFNeuron).
        return 1.0 / (1.0 + std::exp(-62.0 * v) * (1.2 / 3.57));
    }

    // ampa_input / nmda_input / gaba_input: per-neuron input for this tick, or nullptr for "None"
    // (skip that receptor type). Non-null pointers must point at n doubles. Raw pointers so the
    // cerebellum can pass its buffers without copying them (the GC input is 60000 doubles a tick);
    // the Python binding converts None / arrays to these. Returns the per-neuron spike flags.
    const std::vector<uint8_t>& step(double dt,
                                     const double* ampa_ptr = nullptr,
                                     const double* nmda_ptr = nullptr,
                                     const double* gaba_ptr = nullptr) {
        std::fill(spiked.begin(), spiked.end(), 0);

        const double decay_ampa = std::exp(-dt / p.tau_AMPA);
        const double decay_nmda = std::exp(-dt / p.tau_NMDA);
        const double decay_gaba = std::exp(-dt / p.tau_GABA);
        // membrane decay over the tick when only the leak conductance is on (most GCs, every tick).
        // Same expression as the general update below with G == gL, so using it doesn't change results.
        const double leak_decay = std::exp(-dt * p.gL / p.Cm);
        // V_inf with no synaptic conductance, by the same expression as the general update below
        // (gA = gN = gG = 0, G = gL), so the quiescent fast path gives bit-identical results
        const double zero = 0.0;
        const double leak_V_inf = (p.gL * p.EL + zero * p.E_AMPA + zero * p.E_GABA) / p.gL;

        for (std::size_t i = 0; i < n; ++i) {
            // 3) Advance refractory countdown
            refractory_time_left[i] = std::max(0.0, refractory_time_left[i] - dt);
            const bool active = refractory_time_left[i] <= 0.0;

            // 1) Decay existing conductances for every neuron (Eqs. 7-9)
            g_AMPA[i] *= decay_ampa;
            g_NMDA[i] *= decay_nmda;
            g_GABA[i] *= decay_gaba;

            // 2) Add new synaptic input -- applies to refractory neurons too,
            //    matching LIFNeuron's behavior.
            if (ampa_ptr) g_AMPA[i] += ampa_ptr[i];
            if (nmda_ptr) g_NMDA[i] += nmda_ptr[i];
            if (gaba_ptr) g_GABA[i] += gaba_ptr[i];

            if (!active) continue;

            // fast path: no conductance at all (almost every GC on any tick) -> only the leak acts
            if (g_AMPA[i] == 0.0 && g_NMDA[i] == 0.0 && g_GABA[i] == 0.0) {
                const double V_new = leak_V_inf + (V[i] - leak_V_inf) * leak_decay;
                if (V_new >= p.V_thr) {
                    spiked[i] = 1;
                    V[i] = p.EL;
                    refractory_time_left[i] = p.T_ref;
                    spike_counts[i] += 1;
                } else {
                    V[i] = V_new;
                }
                continue;
            }

            // 4) Integrate membrane potential (Eqs. 4-6), only for non-refractory neurons.
            //    Each conductance decays exponentially within the tick (Eqs. 7-9), so its
            //    tick mean is used -- every spike then delivers exactly w*tau, as in EDLUT.
            //    With those means held for the tick, V relaxes exponentially toward V_inf.
            //    (Forward Euler with the full post-spike conductance was unstable here and
            //    over-drove every synapse: PC inhibition acted as excitation.)
            const double Vi = V[i];
            const double gA = g_AMPA[i] * p.tau_AMPA / dt * (1.0 - decay_ampa);
            const double gN = g_NMDA[i] * p.tau_NMDA / dt * (1.0 - decay_nmda);
            const double gG = g_GABA[i] * p.tau_GABA / dt * (1.0 - decay_gaba);
            // no NMDA conductance -> skip the Mg-block exp (GCs and PCs never get NMDA input)
            const double g_exc = (gN != 0.0) ? gA + gN * _nmda_inf_scalar(Vi) : gA;
            const double G = p.gL + g_exc + gG;
            const double V_inf = (p.gL * p.EL + g_exc * p.E_AMPA + gG * p.E_GABA) / G;
            const double v_decay = (G == p.gL) ? leak_decay : std::exp(-dt * G / p.Cm);
            const double V_new = V_inf + (Vi - V_inf) * v_decay;

            // 5) Threshold check -> spike, reset, refractory period
            if (V_new >= p.V_thr) {
                spiked[i] = 1;
                V[i] = p.EL;
                refractory_time_left[i] = p.T_ref;
                spike_counts[i] += 1;
            } else {
                V[i] = V_new;
            }
        }

        t += dt;
        return spiked;
    }
};
