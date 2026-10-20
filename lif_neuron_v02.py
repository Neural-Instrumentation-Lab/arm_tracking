"""
dt-independent conductance-based LIF population (Eqs. 4-10 of Abadia et al. 2021).

What changed vs lif_neuron_v01 (LIFPopulation only; the scalar lif_neuron class is kept
for reference with its NMDA sign fixed):

1. Exact per-tick integration instead of forward Euler.
   Each synaptic conductance is an exponential g(s) = g0*exp(-s/tau) inside a tick, so
   its MEAN over the tick is g0*(tau/dt)*(1-exp(-dt/tau)).  The membrane equation with
   that frozen mean conductance is linear, so V is advanced with its closed form
        V <- V_inf + (V - V_inf) * exp(-dt * G_tot / Cm)
   This is stable for any dt (the Euler update is unstable once (gL+g)*dt/Cm > 2, which
   the GC and DCN were at 2 ms) and every spike delivers exactly w*tau of charge no
   matter what dt is.  (The v01 order -- decay, add, Euler -- delivered w*dt/(1-exp(-dt/tau))
   instead: 2.3x too much for GC AMPA, 4.1x for DCN AMPA at dt = 2 ms.)

2. `ampa_tonic` input: a constant AMPA conductance (Siemens) that does not go through the
   spike/decay state.  Use it for inputs that were injected "every tick" in v01 (the one-hot
   GC address drive, the constant MF->DCN drive).  Injecting an impulse every tick means an
   implied presynaptic rate of 1/dt, which changes when dt changes; a tonic conductance
   does not.

3. Refractory handling is time-exact: after a spike the neuron is held at reset for T_ref
   seconds and integrates only for the remainder of the tick it wakes up in.  (v01's
   decrement-then-check blocked ceil(T_ref/dt)-1 whole ticks: no refractory at all for
   GC/DCN at dt = 2 ms, and a cap of 1/dt instead of 1/T_ref on the firing rate.)

Spike times are still quantised to the tick (a spike is detected at the END of the tick in
which V crosses threshold).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class NeuronParams:
    """Per-neuron-type parameters"""
    Cm: float = 1.0e-9        # membrane capacitance (F)
    gL: float = 0.1e-6        # leak conductance (S)
    EL: float = -70e-3        # resting potential, potential is reset to this (V)
    E_AMPA: float = 0.0       # AMPA (and NMDA) reversal potential (V)
    E_GABA: float = -80e-3    # GABA reversal potential (V)
    V_thr: float = -50e-3     # spike threshold (V)
    T_ref: float = 1e-3       # absolute refractory period (s)
    tau_AMPA: float = 1.0e-3  # AMPA time constant (s)
    tau_NMDA: float = 20e-3   # NMDA time constant (s)
    tau_GABA: float = 5.0e-3  # GABA time constant (s)


def legacy_charge_gain(tau, dt0=2e-3):
    """
    Charge per spike delivered by the v01 update order (decay, add, forward Euler) at
    timestep dt0, divided by the exact w*tau:   dt0 / (tau * (1 - exp(-dt0/tau))).
    Multiply a spike-driven weight by this to reproduce the drive you had tuned at dt0.
    """
    return dt0 / (tau * (1.0 - np.exp(-dt0 / tau)))


def legacy_tonic_conductance(w, tau, dt0=2e-3):
    """
    Mean conductance produced by injecting weight `w` EVERY tick in the v01 update order at
    dt0 (steady-state sawtooth value right after the add):  w / (1 - exp(-dt0/tau)).
    Pass this as `ampa_tonic` to reproduce the drive you had tuned at dt0.
    """
    return w / (1.0 - np.exp(-dt0 / tau))


class lif_neuron:
    """Scalar reference neuron (unchanged from v01 except for the NMDA sign)."""

    def __init__(self, params: NeuronParams = None):
        self.p = params
        self.V = self.p.EL
        self.g_AMPA = 0.0
        self.g_NMDA = 0.0
        self.g_GABA = 0.0
        self.refact_t = 0.0
        self.spiked = False
        self.spike_times = []
        self.t = 0.0

    def _nmda_inf(self) -> float:
        """Jahr-Stevens Mg block: 1/(1 + [Mg]/3.57 * exp(-0.062 V_mV)); V in volts -> -62*V."""
        return 1.0 / (1.0 + np.exp(-62.0 * self.V) * (1.2 / 3.57))

    def _decay_conductances(self, dt):
        self.g_AMPA *= np.exp(-dt / self.p.tau_AMPA)
        self.g_NMDA *= np.exp(-dt / self.p.tau_NMDA)
        self.g_GABA *= np.exp(-dt / self.p.tau_GABA)

    def step(self, dt=2.0e-3, ampa_weights=(), nmda_weights=(), gaba_weights=()):
        p = self.p
        self.spiked = False
        self.g_AMPA += sum(ampa_weights)
        self.g_NMDA += sum(nmda_weights)
        self.g_GABA += sum(gaba_weights)
        e = [np.exp(-dt / t) for t in (p.tau_AMPA, p.tau_NMDA, p.tau_GABA)]
        gA = self.g_AMPA * p.tau_AMPA / dt * (1 - e[0])
        gN = self.g_NMDA * p.tau_NMDA / dt * (1 - e[1]) * self._nmda_inf()
        gG = self.g_GABA * p.tau_GABA / dt * (1 - e[2])
        self.g_AMPA *= e[0]; self.g_NMDA *= e[1]; self.g_GABA *= e[2]
        if self.refact_t > 1e-12:
            self.refact_t = max(0.0, self.refact_t - dt)
            self.t += dt
            return False
        G = p.gL + gA + gN + gG
        Vinf = (p.gL * p.EL + (gA + gN) * p.E_AMPA + gG * p.E_GABA) / G
        self.V = Vinf + (self.V - Vinf) * np.exp(-dt * G / p.Cm)
        if self.V >= p.V_thr:
            self.spiked = True
            self.V = p.EL
            self.refact_t = p.T_ref
            self.spike_times.append(self.t)
        self.t += dt
        return self.spiked


class LIFPopulation:
    """
    Vectorized population, N neurons sharing one NeuronParams.

    step(dt, ampa_input, nmda_input, gaba_input, ampa_tonic)
        *_input     : (n,) summed synaptic weights of the spikes ARRIVING this tick (S).
                      Each spike delivers exactly w*tau of conductance-time, any dt.
        ampa_tonic  : (n,) or scalar, constant AMPA conductance (S) for this tick.
    Returns bool array of neurons that spiked this tick.
    """

    # refractory countdown is considered finished below this (s)
    _REF_EPS = 1e-12

    def __init__(self, n: int, params: NeuronParams = None, legacy_refractory: bool = False):
        self.n = n
        # True reproduces v01's refractory rule (see step); only meaningful at the dt it was tuned at
        self.legacy_refractory = legacy_refractory
        self.p = params if params is not None else NeuronParams()
        p = self.p
        self.V = np.full(n, p.EL, dtype=np.float64)
        self.g_AMPA = np.zeros(n, dtype=np.float64)
        self.g_NMDA = np.zeros(n, dtype=np.float64)
        self.g_GABA = np.zeros(n, dtype=np.float64)
        self.refractory_time_left = np.zeros(n, dtype=np.float64)
        self.spike_counts = np.zeros(n, dtype=np.int64)
        self.t = 0.0
        self._dt = None

    def _nmda_inf(self, V: np.ndarray) -> np.ndarray:
        """Jahr-Stevens Mg block, Eq. (10): 1/(1 + 1.2/3.57 * exp(-62 V)), V in volts."""
        return 1.0 / (1.0 + np.exp(-62.0 * V) * (1.2 / 3.57))

    def _prepare(self, dt):
        p = self.p
        self._dt = dt
        taus = (p.tau_AMPA, p.tau_NMDA, p.tau_GABA)
        self._decay = tuple(float(np.exp(-dt / tau)) for tau in taus)
        # mean of exp(-s/tau) over one tick
        self._mean = tuple(tau / dt * (1.0 - d) for tau, d in zip(taus, self._decay))

    def step(self, dt: float, ampa_input=None, nmda_input=None, gaba_input=None, ampa_tonic=None):
        if dt != self._dt:
            self._prepare(dt)
        p = self.p
        dA, dN, dG = self._decay
        mA, mN, mG = self._mean

        # Refractory: a neuron integrates only for the part of this tick that is left after its
        # refractory time.  (v01 decremented first and then tested, which blocked only
        # ceil(T_ref/dt)-1 whole ticks -- i.e. NO refractory at all whenever T_ref <= dt.)
        if self.legacy_refractory:
            # v01 rule: whole ticks only, blocked while the countdown is still > 0 after decrementing
            self.refractory_time_left = np.maximum(0.0, self.refractory_time_left - dt)
            self.refractory_time_left[self.refractory_time_left <= self._REF_EPS] = 0.0
            t_act = np.where(self.refractory_time_left > 0.0, 0.0, dt)
        else:
            t_act = np.clip(dt - self.refractory_time_left, 0.0, dt)
            self.refractory_time_left = np.maximum(0.0, self.refractory_time_left - dt)
        active = t_act > self._REF_EPS

        # spikes arriving at the start of this tick (refractory neurons still receive them)
        if ampa_input is not None:
            self.g_AMPA += ampa_input
        if nmda_input is not None:
            self.g_NMDA += nmda_input
        if gaba_input is not None:
            self.g_GABA += gaba_input

        # mean conductances over the tick
        gA = self.g_AMPA * mA
        if ampa_tonic is not None:
            gA = gA + ampa_tonic
        gN = self.g_NMDA * mN * self._nmda_inf(self.V)
        gG = self.g_GABA * mG
        G = p.gL + gA + gN + gG
        V_inf = (p.gL * p.EL + (gA + gN) * p.E_AMPA + gG * p.E_GABA) / G
        # exact solution over the non-refractory part of the tick (t_act == 0 leaves V at reset)
        self.V = V_inf + (self.V - V_inf) * np.exp(-t_act * G / p.Cm)

        # conductances decay for the next tick
        self.g_AMPA *= dA
        self.g_NMDA *= dN
        self.g_GABA *= dG

        spiked = active & (self.V >= p.V_thr)
        self.V = np.where(spiked, p.EL, self.V)
        self.refractory_time_left = np.where(spiked, p.T_ref, self.refractory_time_left)
        self.spike_counts += spiked

        self.t += dt
        return spiked
