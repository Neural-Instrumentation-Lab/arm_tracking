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
// Passing std::nullopt is equivalent to Python's `None` (skip that
// receptor type entirely, same as all-zero input but slightly cheaper).
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
          t(0.0) {}

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

    static double _nmda_inf_scalar(double v) {
        // Eq. (10), vectorized version's formula (differs from the scalar
        // LIFNeuron class -- see note above LIFNeuron).
        return 1.0 / (1.0 + std::exp(-62.0 * v) * (1.2 / 3.57));
    }

    // ampa_input / nmda_input / gaba_input: pass std::nullopt for "None".
    // When provided, must have size() == n.
    std::vector<uint8_t> step(double dt,
                               const std::optional<std::vector<double>>& ampa_input = std::nullopt,
                               const std::optional<std::vector<double>>& nmda_input = std::nullopt,
                               const std::optional<std::vector<double>>& gaba_input = std::nullopt) {
        std::vector<uint8_t> spiked(n, 0);

        const double decay_ampa = std::exp(-dt / p.tau_AMPA);
        const double decay_nmda = std::exp(-dt / p.tau_NMDA);
        const double decay_gaba = std::exp(-dt / p.tau_GABA);

        const std::vector<double>* ampa_ptr = ampa_input ? &(*ampa_input) : nullptr;
        const std::vector<double>* nmda_ptr = nmda_input ? &(*nmda_input) : nullptr;
        const std::vector<double>* gaba_ptr = gaba_input ? &(*gaba_input) : nullptr;

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
            if (ampa_ptr) g_AMPA[i] += (*ampa_ptr)[i];
            if (nmda_ptr) g_NMDA[i] += (*nmda_ptr)[i];
            if (gaba_ptr) g_GABA[i] += (*gaba_ptr)[i];

            if (!active) continue;

            // 4) Integrate membrane potential (Eqs. 4-6), only for
            //    non-refractory neurons
            const double Vi = V[i];
            const double I_internal = -p.gL * (Vi - p.EL);
            const double g_nmda_inf = _nmda_inf_scalar(Vi);
            const double I_external = -(g_AMPA[i] + g_NMDA[i] * g_nmda_inf) * (Vi - p.E_AMPA)
                                       - g_GABA[i] * (Vi - p.E_GABA);
            const double dV = (I_internal + I_external) / p.Cm * dt;
            const double V_new = Vi + dV;

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
