from dataclasses import dataclass
import numpy as np

@dataclass
class NeuronParams:
    """Per-neuron-type parameters"""
    Cm: float = 1.0e-9        # membrane capacitance (F)
    gL: float = 0.1e-6        # leak conductance (S)
    EL: float = -70e-3        # resting potential, potential is reset to this (V)
    E_AMPA: float = 0.0       # AMPA reversal potential (V)
    E_GABA: float = -80e-3    # GABA reversal potential (V)
    V_thr: float = -50e-3     # spike threshold (V)
    T_ref: float = 1e-3       # absolute refractory period (s)
    tau_AMPA: float = 1.0e-3  # AMPA time constant (s)
    tau_NMDA: float = 20e-3   # NMDA time constant (s)
    tau_GABA: float = 5.0e-3  # GABA time constant (s)


class lif_neuron:
    def __init__(self, params: NeuronParams = None):
        self.p = params

        # State variables
        self.V = self.p.EL          # membrane potential
        self.g_AMPA = 0.0           # AMPA conductance
        self.g_NMDA = 0.0           # NMDA conductance
        self.g_GABA = 0.0           # GABA conductance
        self.refact_t = 0.0
        self.spiked = False
 
        # Bookkeeping
        self.spike_times = []
        self.t = 0.0
 
    def _nmda_inf(self) -> float:
        """Voltage-dependent NMDA activation gate, Eq. (10)."""
        return (1.0 / (1.0 + np.exp(62.0 * self.V))) * (1.2 / 3.57)
 
    def _decay_conductances(self, dt: float = 2e-3):
        """Exponential decay of synaptic conductances between spike inputs. Eq (7-9)"""
        self.g_AMPA *= np.exp(-dt / self.p.tau_AMPA)
        self.g_NMDA *= np.exp(-dt / self.p.tau_NMDA)
        self.g_GABA *= np.exp(-dt / self.p.tau_GABA)
 
    def _apply_synaptic_inputs(self, ampa_weights, nmda_weights, gaba_weights):
        """Add instantaneous conductance jumps from incoming spikes (Dirac
        delta contributions in Eqs. 7-9)."""
        if ampa_weights:
            self.g_AMPA += sum(ampa_weights)
        if nmda_weights:
            self.g_NMDA += sum(nmda_weights)
        if gaba_weights:
            self.g_GABA += sum(gaba_weights)
 
    def step(self, dt: float = 2.0e-3, ampa_weights=(), nmda_weights=(), gaba_weights=()):
        """
        Args:
        dt : float
            Integration timestep in seconds (default 2ms)
        ampa_weights, nmda_weights, gaba_weights : array of floats
            Synaptic weights w_i for any presynaptic spikes arriving at this
            receptor type during this timestep. Pass an empty tuple/list if
            none arrived.
 
        Returns
        bool
            True if the neuron emitted a spike during this timestep.
        """
        p = self.p
        self.spiked = False
 
        # Refractory period: membrane potential held at reset, no integration
        if self.refact_t > 0.0:
            self.refact_t = max(0.0, self.refact_t - dt)
            # Conductances still decay / accumulate during refractory period
            self._decay_conductances(dt)
            self._apply_synaptic_inputs(ampa_weights, nmda_weights, gaba_weights)
            self.t += dt
            return False
 
        # 1: update conductances 
        self._decay_conductances(dt)
        self._apply_synaptic_inputs(ampa_weights, nmda_weights, gaba_weights)
 
        # compute currents 
        I_int = -p.gL * (self.V - p.EL)
        g_nmda_inf = self._nmda_inf()
        I_ext = (
            -(self.g_AMPA + self.g_NMDA * g_nmda_inf) * (self.V - p.E_AMPA)
            - self.g_GABA * (self.V - p.E_GABA)
        )
 
        # calculate membrane potential 
        dV = (I_int + I_ext) / p.Cm * dt
        self.V += dV
 
        # see if spiked
        if self.V >= p.V_thr:
            self.spiked = True
            self.V = p.EL
            self.refact_t = p.T_ref
            self.spike_times.append(self.t)
 
        self.t += dt
        return self.spiked

class LIFPopulation:
    """
    Vectorized version of LIFNeuron: simulates N neurons that share the same
    NeuronParams, with per-neuron state held as numpy arrays instead of one
    Python object per neuron. Same equations as LIFNeuron (Eqs. 4-10), just
    stepped for the whole population in one call.
 
    Use this instead of a list of LIFNeuron objects whenever N is large
    (hundreds+) -- e.g. the 10,000-neuron GC population per joint in this
    model. A Python loop over 10,000 LIFNeuron.step() calls, 500 times a
    second (2 ms bins) for a multi-second trial, is orders of magnitude
    slower than the equivalent numpy array ops here.
 
    Usage
    -----
    pop = LIFPopulation(n=10_000, params=GC_PARAMS)
    for t in time_steps:
        ampa_input = np.zeros(pop.n)
        ampa_input[winning_index] = 4 * W_MF_GC   # sum of 4 coincident synapses
        spiked = pop.step(dt, ampa_input=ampa_input)
        # spiked: bool array of shape (n,), True where that neuron fired
    """
 
    def __init__(self, n: int, params: NeuronParams = None):
        self.n = n
        self.p = params if params is not None else NeuronParams()
 
        p = self.p
        self.V = np.full(n, p.EL, dtype=np.float64)
        self.g_AMPA = np.zeros(n, dtype=np.float64)
        self.g_NMDA = np.zeros(n, dtype=np.float64)
        self.g_GABA = np.zeros(n, dtype=np.float64)
        self.refractory_time_left = np.zeros(n, dtype=np.float64)
 
        # Bookkeeping (lightweight -- no per-spike-time list, just counts)
        self.spike_counts = np.zeros(n, dtype=np.int64)
        self.t = 0.0
 
    def _nmda_inf(self, V: np.ndarray) -> np.ndarray:
        """Voltage-dependent NMDA activation gate, Eq. (10), vectorized."""
        return (1.0 / (1.0 + (np.exp(-62.0 * V) * (1.2 / 3.57))))
 
    def step(self, dt: float, ampa_input=None, nmda_input=None, gaba_input=None):
        """
        Advance all N neurons by one timestep dt (seconds).
 
        Parameters
        ----------
        dt : float
            Integration timestep in seconds.
        ampa_input, nmda_input, gaba_input : array of shape (n,) or None
            Total synaptic weight arriving at each neuron this timestep
            (already summed across any coincident presynaptic spikes for
            that neuron -- see LIFNeuron._apply_synaptic_inputs for the
            scalar equivalent). Pass None to skip a receptor type entirely
            (equivalent to all zeros, and slightly faster).
 
        Returns
        -------
        spiked : bool array of shape (n,)
            True for each neuron that emitted a spike this timestep.
        """
        # 3) Advance refractory countdown
        self.refractory_time_left = np.maximum(0.0, self.refractory_time_left - dt)

        p = self.p
        was_refractory = self.refractory_time_left > 0.0
        active = ~was_refractory
 
        # 1) Decay existing conductances for every neuron (Eqs. 7-9, decay term)
        self.g_AMPA *= np.exp(-dt / p.tau_AMPA)
        self.g_NMDA *= np.exp(-dt / p.tau_NMDA)
        self.g_GABA *= np.exp(-dt / p.tau_GABA)
 
        # 2) Add new synaptic input (Eqs. 7-9, spike term) -- applies to
        #    refractory neurons too, matching LIFNeuron's behavior.
        if ampa_input is not None:
            self.g_AMPA += ampa_input
        if nmda_input is not None:
            self.g_NMDA += nmda_input
        if gaba_input is not None:
            self.g_GABA += gaba_input
 
 
        # 4) Integrate membrane potential (Eqs. 4-6), only for non-refractory neurons
        I_internal = -p.gL * (self.V - p.EL)
        g_nmda_inf = self._nmda_inf(self.V)
        I_external = (
            -(self.g_AMPA + self.g_NMDA * g_nmda_inf) * (self.V - p.E_AMPA)
            - self.g_GABA * (self.V - p.E_GABA)
        )
        dV = (I_internal + I_external) / p.Cm * dt
        self.V = np.where(active, self.V + dV, self.V)
 
        # 5) Threshold check -> spike, reset, refractory period (only for
        #    neurons that were free to integrate this step)
        spiked = active & (self.V >= p.V_thr)
        self.V = np.where(spiked, p.EL, self.V)
        self.refractory_time_left = np.where(spiked, p.T_ref, self.refractory_time_left)
        self.spike_counts += spiked
 
        self.t += dt
        return spiked