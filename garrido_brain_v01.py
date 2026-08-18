"""
This is a recreation of the cerebellar control described in the paper
On Robot Compliance: A Cerebellar Control Approach authored by
Ignacio Abadía , Francisco Naveros , Jesús A. Garrido , Eduardo Ros , and Niceto R. Luque.
"""
import numpy as np
from dataclasses import dataclass
from lif_neuron_v01 import LIFPopulation, NeuronParams
from arm_assets_v02 import angle_diff
class BrainError(Exception): pass

# neuron parameters from table 2
GC_params = NeuronParams(Cm=2.0e-12, gL=1.0e-9, EL=-65e-3, E_AMPA=0, tau_AMPA=1.0e-3, V_thr=-50.0e-3, T_ref=1.0e-3)
PC_params = NeuronParams(Cm=100.0e-12, gL=6.0e-9, EL=-70e-3, E_AMPA=0, tau_AMPA=1.2e-3, V_thr=-52.0e-3, T_ref=2.0e-3)
DCN_params = NeuronParams(Cm=2.0e-12, gL=0.2e-9, EL=-70e-3, E_AMPA=0, E_GABA=-80.0e-3, tau_AMPA=0.5e-3, tau_NMDA=14.0e-3,
                           tau_GABA=10.0e-3, V_thr=-40.0e-3, T_ref=1.0e-3)

class CFsubcomplex:
    # from C_interface_for_robot_control.cpp
    MAX_AMPLITUDE = 3          # input_current >= 0.75
    MEDIUM_AMPLITUDE_UP = 3    # 0.50 < input_current < 0.75
    MEDIUM_AMPLITUDE_DOWN = 2  # 0.25 < input_current <= 0.50
    MIN_AMPLITUDE = 1          # input_current <= 0.25
    MAX_SPIKE_FREQ = 10

    def __init__(self, n_neurons=50):
        self.n_neurons = n_neurons
        self.max_spk_freq = CFsubcomplex.MAX_SPIKE_FREQ
        self.spikes_pending = np.zeros(n_neurons, dtype=np.int64)
        self.last_spk_time = np.full(n_neurons, -np.inf)
        self.t = 0.0
        self.rng = np.random.default_rng()

    @classmethod
    def burst_size(cls, I):
        if I >= 0.75:
            return cls.MAX_AMPLITUDE
        elif I > 0.50:
            return cls.MEDIUM_AMPLITUDE_UP
        elif I > 0.25:
            return cls.MEDIUM_AMPLITUDE_DOWN
        else:
            return cls.MIN_AMPLITUDE

    def step(self, dt, inp_I):
        spiked = self.spikes_pending > 0
        self.spikes_pending[spiked] -= 1

        idle = self.spikes_pending == 0
        idle_idx = np.nonzero(idle)[0]
 
        if idle_idx.size > 0:
            readiness = (self.t - self.last_spk_time[idle_idx]) * self.max_spk_freq
            np.clip(readiness, 0.0, 1.0, out=readiness)
 
            p_trigger = readiness * inp_I * dt * self.max_spk_freq
            draws = self.rng.random(size=idle_idx.size)
            triggered_local = p_trigger > draws
            triggered_idx = idle_idx[triggered_local]
 
            if triggered_idx.size > 0:
                num_spk = self.burst_size(inp_I)
                self.spikes_pending[triggered_idx] = num_spk
                self.last_spk_time[triggered_idx] = self.t + (num_spk + 1) * dt
 
        self.t += dt
        return spiked

class CFsubcomplexALT:
    # from ROSPoissonGenerator.cpp 
    SIGMA = 1
    MIN_SPIKE_FREQ = 1
    MAX_SPIKE_FREQ = 10
    MIN_ERROR = 0.001
    MAX_ERROR = 0.01

    def __init__(self, n_neurons=50):
        self.n_neurons = n_neurons
        self.centers = np.linspace(CFsubcomplexALT.MIN_ERROR, CFsubcomplexALT.MAX_ERROR, num=n_neurons) 
        self.widths = np.ones_like(self.centers) * CFsubcomplexALT.SIGMA*(CFsubcomplexALT.MAX_ERROR - CFsubcomplexALT.MIN_ERROR)/(n_neurons-1)
        self.rng = np.random.default_rng()
        self.spikes_pending = np.zeros(n_neurons, dtype=np.int64)

    def step(self, dt, inp_I):
        spiked = self.spikes_pending > 0
        self.spikes_pending[spiked] -= 1

        idle = self.spikes_pending == 0
        idle_idx = np.nonzero(idle)[0]

        norm_rof = (np.tanh((inp_I - self.centers) / self.widths) + 1) * 0.5 
        sp_rof = CFsubcomplexALT.MIN_SPIKE_FREQ + norm_rof[idle_idx]*(CFsubcomplexALT.MAX_SPIKE_FREQ - CFsubcomplexALT.MIN_SPIKE_FREQ)
        draws = self.rng.random(size=idle_idx.size)
        triggered = sp_rof*dt >= draws
        triggered_idx = idle_idx[triggered]
        self.spikes_pending[triggered_idx] = inp_I / (self.centers[triggered_idx])
        self.spikes_pending[triggered_idx] = np.clip(self.spikes_pending[triggered_idx], 1, 6)

        return spiked

class MFsubcomplex:
    SIGMA = 0.5

    def __init__(self, min, max, n_neurons=10):
        self.n_neurons = n_neurons
        self.centers = np.linspace(min, max, num=n_neurons) 
        self.widths = np.ones_like(self.centers) * MFsubcomplex.SIGMA*(max - min)/(n_neurons-1)
        self.spikes = np.zeros_like(self.centers, dtype=np.int64)

    def step(self, inp, dt):
        self.spikes.fill(0)
        current = 1 - np.abs((inp - self.centers)/self.widths)
        self.spikes = current > 0
        if not self.spikes.any():
            if inp > self.centers[-1]:
                self.spikes[-1] = 1
            else:
                self.spikes[0] = 1
        return self.spikes


class PFSpikeHistory:
    """
    Bounded, prunable record of recent PF (GC) spikes, stored as a list of
    (indices_array, spike_time) batches -- one batch per timestep that had
    at least one PF spike. Sparse by construction (at most a handful of the
    60,000 GCs fire per 2ms tick), so this stays small.
    """
    HISTORY_PRUNE_WINDOW = 200  # timesteps 

    def __init__(self, prune_window=HISTORY_PRUNE_WINDOW):
        self.prune_window = prune_window
        self._idx_batches = []
        self._times = []
 
    def record(self, indices: np.ndarray, t: float):
        if indices.size > 0:
            self._idx_batches.append(indices)
            self._times.append(t)
 
    def prune(self, current_t: float):
        cutoff = current_t - self.prune_window
        keep = [i for i, t in enumerate(self._times) if t >= cutoff]
        self._idx_batches = [self._idx_batches[i] for i in keep]
        self._times = [self._times[i] for i in keep]
 
    def flatten(self):
        """Return (all_indices, all_spike_times) as parallel 1D arrays."""
        if not self._idx_batches:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
        idx = np.concatenate(self._idx_batches)
        times = np.concatenate(
            [np.full(a.size, t) for a, t in zip(self._idx_batches, self._times)]
        )
        return idx, times
 
    def __len__(self):
        return len(self._idx_batches)

DK = 0.07          # kernel width parameter (s)
TAU_LTD = 0.1      # kernel time constant, aligned with sensorimotor delay (s)
def ltd_kernel(x, dk=DK, tau_ltd=TAU_LTD):
    """
    Eq. 13. x = t_PFspike - t_CFspike, i.e. how long BEFORE the CF spike a
    PF fired (x <= 0). Nonzero only for x < -dk; peaks at x = -tau_ltd with
    value 1.0; decays toward 0 for more negative x.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    mask = x < -dk
    z = (x[mask] + dk) / (tau_ltd - dk)
    out[mask] = -z * np.exp(z + 1.0)
    return out

def restrictAngle(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi 

class cerebellum:
    DEFAULT_MF_VALUE_RANGES = (
        (-1, 1),    # q      (actual position)
        (-1.5, 1.5),    # qd     (actual velocity)
        (-1, 1),    # q_des  (desired position)
        (-1, 1),    # qd_des (desired velocity)
    )
    ALPHA = 0.002e-9          # (S)
    BETA  = -0.001e-9         # (S)
    INIT_PF_PC_WT = 1.6e-9    # (S)
    W_MIN, W_MAX = 0.0, 5e-9  # pf-pc weight lims (S)

    def __init__(self, qMins, qdMins, qMaxs, qdMaxs, n_dof=6):
        self.nJoints = n_dof
        self.t = 0
        # these numbers are all PER JOINT
        self.nMF_subgroups    = 4
        self.nMF_per_subgroup = 10
        self.nMF              = self.nMF_per_subgroup * self.nMF_subgroups
        self.W_MF_GC   = 0.18e-9
        self.W_PC_DCN  = 1e-9
        self.W_MF_DCN   = 0.1e-9 
        self.W_CF_DCN_AMPA  = 0.5e-9
        self.W_CF_DCN_NMDA  = 0.25e-9
        self.TORQUE_ALPHA  = [0.75, 3.0, 0.375, 1.5, 0.05, 0.05]

        self.nGC      = self.nMF_per_subgroup ** self.nMF_subgroups 
        self.nCF      = 100
        self.nPC      = 100
        self.nDCN     = 100

        self.qmins = qMins
        self.qmaxs = qMaxs
        self.qdmins = qdMins
        self.qdmaxs = qdMaxs

        # this AMPA input is constant from MF firing rate being constant
        # self.mf_to_dcn = np.ones(self.nDCN*n_dof)*self.nMF_subgroups*n_dof*self.W_MF_DCN
        self.mf_to_dcn = np.ones(self.nDCN*n_dof)*self.W_MF_DCN
        # self.mf_to_dcn = np.ones(self.nDCN*n_dof)*0.12e-9

        self.gcNeurons     = LIFPopulation(n=self.nGC * n_dof, params=GC_params)
        self.gc_ampa_input = np.zeros(self.nGC * n_dof, dtype=np.float64)
        self.pf_history = PFSpikeHistory()

        self.CF_complexes  = [CFsubcomplexALT(n_neurons=int(self.nCF/2)) for _ in range(self.nJoints*2)] 
        self.cf_output     = np.zeros(self.nCF * n_dof, dtype=np.bool)

        self.MF_complexesQ      = [MFsubcomplex(mi, ma, self.nMF_per_subgroup) for (mi, ma) in zip(qMins, qMaxs)]
        self.MF_complexesQdes   = [MFsubcomplex(mi, ma, self.nMF_per_subgroup) for (mi, ma) in zip(qMins, qMaxs)]
        self.MF_complexesQd     = [MFsubcomplex(mi, ma, self.nMF_per_subgroup) for (mi, ma) in zip(qdMins, qdMaxs)]
        self.MF_complexesQddes  = [MFsubcomplex(mi, ma, self.nMF_per_subgroup) for (mi, ma) in zip(qdMins, qdMaxs)]
        self.mf_out     = np.zeros(self.nMF * n_dof, dtype=np.float64)

        self.pf_pc_wts = np.full((self.nGC * n_dof, self.nPC * n_dof), cerebellum.INIT_PF_PC_WT)
        self.pc_out = np.zeros(self.nPC * n_dof, dtype=bool)

        self.pcNeurons     = LIFPopulation(n=self.nPC * n_dof, params=PC_params)
        self.dcnNeurons    = LIFPopulation(n=self.nDCN * n_dof, params=DCN_params)

        self.prevDCN       = np.zeros((15, self.nJoints))

        self.timeStep = 2e-3

        # pre-calculate a bunch of these values
        self.kernelLookup = self.makeKernelLookup() 


    def makeKernelLookup(self):
        table = {}
        for x in np.arange(0, -201, -1):
            table[x] = ltd_kernel(x*self.timeStep)
        return table

    def encode_mf_address(self, state, value_ranges=DEFAULT_MF_VALUE_RANGES):
        """
        Convert (q, qd, q_des, qd_des) into 4 digits in [0, 9], one per MF
        subgroup, via uniform binning.
        Returns
        -------
        tuple of 4 ints, each in [0, N_MF_PER_SUBGROUP - 1]
        """
        digits = []
        for value, (lo, hi) in zip(state, value_ranges):
            frac = (value - lo) / (hi - lo)
            digit = int(frac * self.nMF_per_subgroup)
            digit = int(np.clip(digit, 0, self.nMF_per_subgroup - 1))  
            digits.append(digit)
        return tuple(digits)

    def address_to_gc_index(self, digits, nJoint):
        """Combine 4 base-10 digits into a single GC index in [0, 9999]."""
        d0, d1, d2, d3 = digits
        return (d0 + d1 * self.nMF_per_subgroup + d2 * self.nMF_per_subgroup**2 + d3 * self.nMF_per_subgroup**3) + nJoint*self.nGC

    def granularLayer(self, q, qd, qdes, qddes, dt):
        self.gc_ampa_input.fill(0.0)
        """
        RBF BASED MF IMPLEMENTATION (NOT WORKING YET)
        """
        # self.mf_out.fill(0.0)
        # step = self.nMF_per_subgroup * self.nJoints  
        # steep = self.nMF_per_subgroup
        # for i, (pos, mf_c) in enumerate(zip(q, self.MF_complexesQ)):
        #     self.mf_out[i*steep:(i+1)*steep] = mf_c.step(pos, dt)
        # for i, (pos, mf_c) in enumerate(zip(qdes, self.MF_complexesQdes)):
        #     self.mf_out[step+i*steep:step+(i+1)*steep] = mf_c.step(pos, dt)
        # for i, (pos, mf_c) in enumerate(zip(qd, self.MF_complexesQd)):
        #     self.mf_out[step*2+i*steep:step*2+(i+1)*steep] = mf_c.step(pos, dt)
        # for i, (pos, mf_c) in enumerate(zip(qddes, self.MF_complexesQddes)):
        #     self.mf_out[step*3+i*steep:step*3+(i+1)*steep] = mf_c.step(pos, dt)
        # self.mf_out = self.mf_out.reshape(self.nMF_subgroups, self.nJoints, self.nMF_per_subgroup)
        # # This part isn't right yet
        # chosen = np.nonzero(self.mf_out)
        # for joint in range(self.nJoints):
        #     idx = chosen[1] == joint
        #     subgroups, mf = (chosen[0])[idx], (chosen[2])[idx]
        #     for q_mf, q_mf_i in zip(subgroups[subgroups == 0], mf[subgroups == 0]):
        #         for qdes_mf, qdes_mf_i in zip(subgroups[subgroups == 1], mf[subgroups == 1]):
        #             for qd_mf, qd_mf_i in zip(subgroups[subgroups == 2], mf[subgroups == 2]):
        #                 for qddes_mf, qddes_mf_i in zip(subgroups[subgroups == 3], mf[subgroups == 3]):
        #                     digits = [q_mf_i, qdes_mf_i, qd_mf_i, qddes_mf_i]
        #                     addressed_index = self.address_to_gc_index(digits, joint)
        #                     self.gc_ampa_input[addressed_index] = self.W_MF_GC * self.nMF_subgroups
        """
        ONE-HOT IMPLEMENTATION (WORKING BUT I DON'T THINK THIS IS HOW THEY DO IT
        """
        for j in range(self.nJoints):
            digits = self.encode_mf_address((q[j], qd[j], qdes[j], qddes[j]), ((self.qmins[j], self.qmaxs[j]), 
                                                                               (self.qdmins[j], self.qdmaxs[j]), 
                                                                               (self.qmins[j], self.qmaxs[j]), 
                                                                               (self.qdmins[j], self.qdmaxs[j])))
            addressed_index = self.address_to_gc_index(digits, j)
            self.gc_ampa_input[addressed_index] = self.W_MF_GC * self.nMF_subgroups
        self.pfs = self.gcNeurons.step(dt=dt, ampa_input=self.gc_ampa_input)


    def errorCalc(self, error):
        max_error = 1 
        return 0.2+0.8*(1-np.exp(-error*90/max_error));

    def climbingFibers(self, qErr, qdErr, dt):
        self.cf_output.fill(False)
        kp = np.array([1.5, 2, 3, 2, 3, 3])
        kd = np.array([1.5, 1, 3, 1, 3, 0.5])
        # kp = np.ones(self.nJoints)*0.5
        # kd = np.ones(self.nJoints)*0.5/(2*np.pi)
        sigError = kp * (qErr) + kd * (qdErr)
        # pError = self.errorCalc(sigError)
        # nError = self.errorCalc(-sigError)
        pError = sigError
        nError = -sigError
        agonist = [x if y < 0 else 0 for (x,y) in zip(nError, sigError)]
        antagonist = [x if y >= 0 else 0 for (x,y) in zip(pError, sigError)]
        for j in range(self.nJoints):
            epsAgon  = agonist[j]
            epsAAgon = antagonist[j]
            startIdx = int(j * self.nCF)
            self.cf_output[startIdx:startIdx+int(self.nCF/2)] = self.CF_complexes[j*2].step(dt, epsAgon)
            self.cf_output[startIdx+int(self.nCF/2):startIdx+int(self.nCF)]  = self.CF_complexes[j*2+1].step(dt, epsAAgon)

    def update_pf_pc_weights(self, pf_pc_wts, pf_spike_idx, cf_spike_idx, current_t, history):
        """
        Advance PF-PC weights by one timestep's worth of LTP + LTD.
    
        Parameters
        ----------
        pf_pc_wts : np.ndarray, shape (n_gc_total, n_pc_total)
            Weight matrix, modified IN PLACE.
        pf_spike_idx : np.ndarray of int
            Flat GC/PF indices that fired THIS timestep.
        cf_spike_idx : np.ndarray of int
            Flat CF indices that fired THIS timestep -- assumed to equal the PC
            column index each maps to (one-to-one CF-PC connectivity per Table I).
        current_t : float
            Current simulation time; CF spikes this call are treated as
            occurring exactly at current_t.
        history : PFSpikeHistory
            Persistent across calls -- pass the same object every timestep.
        """
        # --- LTP: fixed jump of ALPHA at every PF spike, applied to ALL PC
        #     columns (every PF connects to every PC). No dt factor -- this is
        #     a discrete per-spike jump (Eq. 11's Dirac delta), not a rate. ---
        if pf_spike_idx.size > 0:
            pf_pc_wts[pf_spike_idx, :] += cerebellum.ALPHA
            pf_pc_wts[pf_spike_idx, :] = np.clip(pf_pc_wts[pf_spike_idx, :], cerebellum.W_MIN, cerebellum.W_MAX)
    
        # --- Record this step's PF spikes for future LTD lookups, then prune
        #     anything old enough to be kernel-negligible. ---
        history.record(pf_spike_idx, current_t)
        history.prune(current_t)
    
        # --- LTD: every CF spike this step looks back at PF history and applies
        #     a kernel-weighted decrement, restricted to that CF's own PC column. ---
        if cf_spike_idx.size > 0:
            hist_idx, hist_time = history.flatten()
            if hist_idx.size > 0:
                x = hist_time - current_t  
                # k_vals = ltd_kernel(x)
                k_vals = np.array([self.kernelLookup[x_val] for x_val in x])
                nonzero = k_vals != 0.0
                if np.any(nonzero):
                    rows = hist_idx[nonzero]
                    k_nonzero = k_vals[nonzero]
                    # Outer-expand (history rows) x (currently-firing CF columns).
                    # np.add.at is required (not fancy-index +=) because `rows`
                    # can contain duplicates -- the same GC may appear in
                    # multiple history batches, and each occurrence must
                    # contribute its own kernel-weighted term; plain fancy
                    # assignment silently drops repeated-index contributions.
                    R = np.repeat(rows, cf_spike_idx.size)
                    C = np.tile(cf_spike_idx, rows.size)
                    V = np.repeat(cerebellum.BETA * k_nonzero, cf_spike_idx.size)
                    np.add.at(pf_pc_wts, (R, C), V)
                    pf_pc_wts[R, C] = np.clip(pf_pc_wts[R, C], cerebellum.W_MIN, cerebellum.W_MAX)

    def PCstep(self, dt):
        pf_idx = np.nonzero(self.pfs)[0]
        cf_idx = np.nonzero(self.cf_output)[0]
        self.update_pf_pc_weights(self.pf_pc_wts, pf_idx, cf_idx, current_t=self.t, history=self.pf_history)
        pc_ampa_inp = np.sum(self.pf_pc_wts[pf_idx], axis=0)
        self.pc_out = self.pcNeurons.step(dt=dt, ampa_input=pc_ampa_inp)

    def dcnToTorque(self, dcnOut, dt):
        fix = dcnOut.reshape(self.nJoints*2, -1)
        torques = np.sum(dcnOut.reshape(self.nJoints*2, -1), axis=1)
        torques[1::2] *= -1
        torques = np.sum(torques.reshape(-1, 2), axis=1)
        self.prevDCN[:-1] = self.prevDCN[1:]
        self.prevDCN[-1] = torques
        corr = (self.TORQUE_ALPHA) * np.mean(self.prevDCN, axis=0)
        return corr

    def compute(self, q, qd, qdes, qddes, qErr, qdErr, dt=2e-3):
        # input fiber layers
        self.granularLayer(restrictAngle(q), qd, restrictAngle(qdes), qddes, dt)
        self.climbingFibers(qErr, qdErr, dt)
        # now we have spikes from the PFs and the CFs in cf_output and pfs
        self.PCstep(dt)
        dcnOut = self.dcnNeurons.step(dt=dt, ampa_input=(self.W_CF_DCN_AMPA*self.cf_output + self.mf_to_dcn), 
                             nmda_input=(self.W_CF_DCN_NMDA*self.cf_output),
                             gaba_input=(self.pc_out*self.W_PC_DCN))
        if dcnOut[150:199].any():
            pass 
        # if not dcnOut.all() and dcnOut.any():
        #     print("waaaaaah")
        corr = self.dcnToTorque(dcnOut, dt)
        self.t += 1
        return(corr)