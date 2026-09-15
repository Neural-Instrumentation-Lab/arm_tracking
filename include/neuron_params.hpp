#pragma once

// Direct port of lif_neuron_v01.py:NeuronParams
struct NeuronParams {
    double Cm       = 1.0e-9;   // membrane capacitance (F)
    double gL       = 0.1e-6;   // leak conductance (S)
    double EL       = -70e-3;   // resting potential, potential is reset to this (V)
    double E_AMPA   = 0.0;      // AMPA reversal potential (V)
    double E_GABA   = -80e-3;   // GABA reversal potential (V)
    double V_thr    = -50e-3;   // spike threshold (V)
    double T_ref    = 1e-3;     // absolute refractory period (s)
    double tau_AMPA = 1.0e-3;   // AMPA time constant (s)
    double tau_NMDA = 20e-3;    // NMDA time constant (s)
    double tau_GABA = 5.0e-3;   // GABA time constant (s)
};
