"""
This is a recreation of the cerebellar control described in the paper
On Robot Compliance: A Cerebellar Control Approach authored by
Ignacio Abadía , Francisco Naveros , Jesús A. Garrido , Eduardo Ros , and Niceto R. Luque.
"""
import numpy as np
from dataclasses import dataclass

class BrainError(Exception): pass

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

# neuron parameters from table 2
GC_params = NeuronParams(Cm=2.0e-12, gL=1.0e-9, EL=-65e-3, E_AMPA=0, tau_AMPA=1.0e-3, V_thr=-50.0e-3, T_ref=1.0e-3)
PC_params = NeuronParams(Cm=100.0e-12, gL=6.0e-9, EL=-70e-3, E_AMPA=0, tau_AMPA=1.2e-3, V_thr=-52.0e-3, T_ref=2.0e-3)
DCN_params = NeuronParams(Cm=2.0e-12, gL=0.2e-9, EL=-70e-3, E_AMPA=0, E_GABA=-80.0e-3, tau_AMPA=0.5e-3, tau_NMDA=14.0e-3,
                           tau_GABA=10.0e-3, V_thr=-50.0e-3, T_ref=1.0e-3)

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
        return (1.0 / (1.0 + np.exp(-62.0 * self.V))) * (1.2 / 3.57)
 
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
            self.V = p.V_reset
            self.refact_t = p.T_ref
            self.spike_times.append(self.t)
 
        self.t += dt
        return self.spiked



class cerebellum:
    def __init__(self, n_dof=2):
        self.currPF = 0
        self.nPFs = 500
        self.pfIdx = 0
        self.nMuscles = n_dof*2 # one pair agonist-antagonist per joint
        self.purAct = np.zeros(self.nMuscles)
        self.pcLookup = np.zeros(self.nPFs) 
        self.pf_pc_weights = np.zeros((self.nPFs, self.nMuscles))
        self.mf_dcn_weights = np.zeros(self.nMuscles)
        self.pc_dcn_weights = np.zeros(self.nMuscles)
        self.dcnAct = np.zeros(self.nMuscles)
        self.pf_pc_only = False 
        # from garrido paper
        self.LTP_max = 0.01 # long-term potentiation
        self.LTD_max = 0.02 # long-term depression 
        self.LTP_max_dcn = 1e-3 # long-term potentiation
        self.LTD_max_dcn = 1e-4 # long-term depression 
        self.alpha = 1000 # LTP decay factor
        self.active_pf_pc = True
        self.active_pc_dcn = True
        self.active_mf_dcn = True

    def granuleLayer(self, state):
        '''
        The granular layer translates the 
        inputs to the parallel fibers,
        which act as a state machine (discritizing the motion)
        '''
        self.pfIdx = (state - 1) % self.nPFs
        self.currPF = (state) % self.nPFs 
        #print(f"{self.pfIdx} {self.currPF}")
        # if state >= self.nPFs:
        #     raise BrainError("state is greater than the number of PFs")

    def updatePF_PC(self, error):
        '''
        updates the synaptic weights between the parallel fibers
        and the purkinje cell
        '''
        self.pf_pc_weights[self.pfIdx,:] += (self.LTP_max / ((error+1)**self.alpha)) - self.LTD_max*error
        self.pf_pc_weights[self.pfIdx,:] = np.clip(self.pf_pc_weights[self.pfIdx,:], 0, 1)

    def purkinjeCompute(self):
        '''
        computes the purkinje cell firing rate [0 - 1]
        '''
        self.purAct = self.pf_pc_weights[self.currPF, :].copy() 
        self.purAct = np.clip(self.purAct, 0, 1)

    def getPC(self):
        return self.purAct

    def getDCN(self):
        return self.dcnAct

    def updateMF_DCN(self):
        '''
        updates the synaptic weights between the mossy fibers and the 
        deep cerebellar nuclei
        '''
        self.mf_dcn_weights += (self.LTP_max_dcn / ((self.purAct + 1)**self.alpha)) - self.LTD_max_dcn*self.purAct
        self.mf_dcn_weights = np.clip(self.mf_dcn_weights, 0, None) 

    def getMF_DCN(self):
        return(self.mf_dcn_weights)

    def getPC_DCN(self):
        return(self.pc_dcn_weights)

    def getPF_PC(self):
        return(self.pf_pc_weights)

    def getnPFs(self):
        return(self.nPFs)

    def setActiveSites(self, pf_pc, mf_dcn, pc_dcn):
        self.active_mf_dcn = mf_dcn
        self.active_pc_dcn = pc_dcn
        self.active_pf_pc  = pf_pc

    def updatePC_DCN(self):
        '''
        updates the weights between the purkinje cell
        and the deep cerebellar nuclei
        '''
        dcn_clipped = np.clip(self.dcnAct, 0, 1)
        self.pc_dcn_weights += ((self.LTP_max_dcn * self.purAct**self.alpha) * (1 - 1/(dcn_clipped + 1)**self.alpha)) - self.LTD_max_dcn*(1-self.purAct)
        self.pc_dcn_weights = np.clip(self.pc_dcn_weights, 0, None) 

    def DCNCompute(self):
        '''
        computes the deep cerebellar nuclei's activation,
        which is the torque outputs (for each muscle)
        '''
        self.dcnAct = self.mf_dcn_weights - self.purAct*self.pc_dcn_weights 
        self.dcnAct = np.clip(self.dcnAct, 0, None)

    def dcnToTorque(self):
        '''
        takes the DCN's output and adds antagonist and agonist together
        to make joint torque commands
        '''
        corr = self.dcnAct.copy()
        corr[1::2] *= -1
        corr = (corr).reshape(-1, 2).sum(axis=1) 
        return corr
    
    def compute(self, qError, qdError, state):
        '''
        updates the entire brain given the error signals
        '''
        # combine errors
        if state == 0:
            # no learning, just computation on first state
            self.currPF = 0
            self.purkinjeCompute()
            self.DCNCompute()
            return self.dcnToTorque()
        self.granuleLayer(state)
        posCon = np.ones(int(self.nMuscles/2))
        velCon = np.ones(int(self.nMuscles/2))
        error = posCon*qError + velCon*qdError 
        # error = np.tanh(error)
        agonist = np.maximum(error, 0)
        antagonist = np.maximum(-error, 0)
        error = np.stack([agonist, antagonist], axis=1).reshape(-1)  # (n_muscles,)
        error = np.clip(error, 0, 1) 
        #error = np.tanh(error) # clips errors to 0 - 1 BUT I DONT LIKE IT
        # for agonist / antagonist pairs
        if self.active_pf_pc:
            self.updatePF_PC(error)
        self.purkinjeCompute()
        if self.active_mf_dcn:
            self.updateMF_DCN()
        if self.active_pc_dcn:
            self.updatePC_DCN()
        self.DCNCompute()
        return self.dcnToTorque()

    def loadWts(self, init_pf_pc, init_mf_dcn, init_pc_dcn):
        self.pf_pc_weights  = init_pf_pc
        self.mf_dcn_weights = init_mf_dcn
        self.pc_dcn_weights = init_pc_dcn