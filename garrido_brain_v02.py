"""
This is a recreation of the cerebellar control described in the paper
On Robot Compliance: A Cerebellar Control Approach authored by
Ignacio Abadía , Francisco Naveros , Jesús A. Garrido , Eduardo Ros , and Niceto R. Luque.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from lif_neuron_v02 import LIFPopulation, NeuronParams, legacy_charge_gain, legacy_tonic_conductance
from arm_assets_v02 import angle_diff
class BrainError(Exception): pass

# --- dt handling -------------------------------------------------------------------------
# Everything that used to be "N ticks" is now defined in seconds and converted with the
# brain's dt.  LEGACY_DT is the timestep the v01 constants (W_MF_GC*4, mf_to_dcn, TORQUE_ALPHA,
# alpha/beta, ...) were tuned at.
LEGACY_DT = 2e-3
# Set to False to use Table-I weights as-is (each spike then delivers exactly w*tau).
# True keeps the *effective* drive you tuned at 2 ms, so a dt change alone shouldn't move behavior.
KEEP_LEGACY_DRIVE = True
# True uses v01's refractory rule (blocks ceil(T_ref/dt)-1 whole ticks), plus v01's W_MF_DCN and
# TORQUE_ALPHA, so at dt = 2 ms this brain reproduces v01 exactly.  At 2 ms that rule blocks
# nothing (every T_ref here is <= 2 ms), so results will NOT hold up as dt shrinks.
# False uses exact refractory periods, which give consistent results across dt.
KEEP_LEGACY_REFRACTORY = True
PF_HISTORY_S = 0.4        # PF spike history kept for LTD (was 200 ticks @ 2 ms)
TORQUE_FILTER_S = 0.030   # mean-filter window (was 15 taps @ 2 ms)
CF_BURST_ISI_S = 0.002    # spacing of spikes inside a CF burst (was 1 tick @ 2 ms)

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
    MAX_ERROR = 0.1

    def __init__(self, n_neurons=50):
        self.n_neurons = n_neurons
        self.centers = np.linspace(CFsubcomplexALT.MIN_ERROR, CFsubcomplexALT.MAX_ERROR, num=n_neurons) 
        self.widths = np.ones_like(self.centers) * CFsubcomplexALT.SIGMA*(CFsubcomplexALT.MAX_ERROR - CFsubcomplexALT.MIN_ERROR)/(n_neurons-1)
        self.rng = np.random.default_rng()
        self.spikes_pending = np.zeros(n_neurons, dtype=np.int64)
        self.wait = np.zeros(n_neurons, dtype=np.int64)   # ticks until the next burst spike

    def step(self, dt, inp_I):
        # spikes inside a burst are CF_BURST_ISI_S apart (1 tick at the legacy 2 ms)
        isi_ticks = max(1, int(round(CF_BURST_ISI_S / dt)))
        pending = self.spikes_pending > 0
        self.wait[pending] -= 1
        spiked = pending & (self.wait <= 0)
        self.spikes_pending[spiked] -= 1
        self.wait[spiked] = isi_ticks

        idle = self.spikes_pending == 0
        idle_idx = np.nonzero(idle)[0]

        norm_rof = (np.tanh((inp_I - self.centers) / self.widths) + 1) * 0.5
        sp_rof = CFsubcomplexALT.MIN_SPIKE_FREQ + norm_rof[idle_idx]*(CFsubcomplexALT.MAX_SPIKE_FREQ - CFsubcomplexALT.MIN_SPIKE_FREQ)
        draws = self.rng.random(size=idle_idx.size)
        triggered = sp_rof*dt >= draws
        triggered_idx = idle_idx[triggered]
        self.spikes_pending[triggered_idx] = inp_I / (self.centers[triggered_idx])
        self.spikes_pending[triggered_idx] = np.clip(self.spikes_pending[triggered_idx], 1, 6)
        self.wait[triggered_idx] = 1     # first spike of the burst on the next tick, as before

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
    HISTORY_PRUNE_WINDOW = 200  # timesteps (legacy default; the brain passes round(PF_HISTORY_S/dt))

    def __init__(self, prune_window=HISTORY_PRUNE_WINDOW):
        self.prune_window = prune_window
        # times are recorded in non-decreasing order, so pruning is a popleft loop (O(removed))
        self._idx_batches = deque()
        self._times = deque()

    def record(self, indices: np.ndarray, t: float):
        if indices.size > 0:
            self._idx_batches.append(indices)
            self._times.append(t)

    def prune(self, current_t: float):
        cutoff = current_t - self.prune_window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._idx_batches.popleft()
 
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

class cerebellum:
    DEFAULT_MF_VALUE_RANGES = (
        (-2, 2),    # q      (actual position)
        (-1, 1),    # q_des  (desired position)
        (-1, 1),    # q_des  (desired position)
        (-1, 1),    # q_des  (desired position)
    )
    ALPHA = 0.002e-9          # (S)
    BETA  = -0.001e-9         # (S)
    INIT_PF_PC_WT = 1.6e-9    # (S)
    W_MIN, W_MAX = 0.0, 5e-9  # pf-pc weight lims (S)

    def __init__(self, qMins, qMaxs, n_dof=6, dt=LEGACY_DT):
        self.nJoints = n_dof
        self.dt = dt
        self.t = 0            # tick counter; time in seconds is self.t * self.dt
        # these numbers are all PER JOINT
        self.nMF_subgroups    = 2
        self.nMF_per_subgroup = 20
        self.nMF              = self.nMF_per_subgroup * self.nMF_subgroups
        self.W_MF_GC   = 0.18e-9
        self.W_PC_DCN  = 1e-9
        # exact refractory blocks PCs for a whole 2 ms tick after each spike, so PCs inhibit the
        # DCN less and DCN output roughly doubles; these were retuned for that
        if KEEP_LEGACY_REFRACTORY:
            self.W_MF_DCN      = 0.1e-9
            self.TORQUE_ALPHA  = [.24, .24]
        else:
            self.W_MF_DCN      = 0.2e-9
            self.TORQUE_ALPHA  = [.12, .12]
        self.W_CF_DCN_AMPA  = 0.5e-9
        self.W_CF_DCN_NMDA  = 0.25e-9

        self.nGC      = self.nMF_per_subgroup ** self.nMF_subgroups 
        self.nCF      = 100
        self.nPC      = 100
        self.nDCN     = 100

        self.qmins = qMins
        self.qmaxs = qMaxs

        self.effDelay = 50e-3
        self.affDelay = 50e-3

        # temp stuff to keep for saving brain results
        self.dcnOut = np.zeros(self.nDCN*n_dof)

        # this AMPA input is constant from MF firing rate being constant
        # self.mf_to_dcn = np.ones(self.nDCN*n_dof)*self.nMF_subgroups*n_dof*self.W_MF_DCN
        # Constant drives are now tonic conductances (S), not an impulse injected every tick
        # (an impulse every tick implies a presynaptic rate of 1/dt).  With KEEP_LEGACY_DRIVE the
        # values equal the mean conductance the v01 code produced at 2 ms; otherwise they are the
        # physical mean of one MF spike (weight W) every LEGACY_DT.
        if KEEP_LEGACY_DRIVE:
            self.gc_tonic_value  = legacy_tonic_conductance(self.W_MF_GC * 4, GC_params.tau_AMPA, LEGACY_DT)
            self.mf_dcn_value    = legacy_tonic_conductance(self.W_MF_DCN, DCN_params.tau_AMPA, LEGACY_DT)
        else:
            self.gc_tonic_value  = self.W_MF_GC * 4 * GC_params.tau_AMPA / LEGACY_DT
            self.mf_dcn_value    = self.W_MF_DCN * DCN_params.tau_AMPA / LEGACY_DT
        self.mf_to_dcn = np.full(self.nDCN*n_dof, self.mf_dcn_value)

        # per-spike charge compensation: v01 delivered w*dt0/(tau*(1-exp(-dt0/tau))) * (w*tau) per
        # spike at dt0 = 2 ms; the exact integrator delivers w*tau.  These gains restore the old drive.
        if KEEP_LEGACY_DRIVE:
            self.gain_pc_ampa  = legacy_charge_gain(PC_params.tau_AMPA,  LEGACY_DT)
            self.gain_dcn_ampa = legacy_charge_gain(DCN_params.tau_AMPA, LEGACY_DT)
            self.gain_dcn_nmda = legacy_charge_gain(DCN_params.tau_NMDA, LEGACY_DT)
            self.gain_dcn_gaba = legacy_charge_gain(DCN_params.tau_GABA, LEGACY_DT)
        else:
            self.gain_pc_ampa = self.gain_dcn_ampa = self.gain_dcn_nmda = self.gain_dcn_gaba = 1.0

        self.gcNeurons     = LIFPopulation(n=self.nGC * n_dof, params=GC_params, legacy_refractory=KEEP_LEGACY_REFRACTORY)
        self.gc_tonic      = np.zeros(self.nGC * n_dof, dtype=np.float64)
        self.pf_history = PFSpikeHistory(prune_window=int(round(PF_HISTORY_S / dt)))

        self.CF_complexes  = [CFsubcomplexALT(n_neurons=int(self.nCF/2)) for _ in range(self.nJoints*2)] 
        self.cf_output     = np.zeros(self.nCF * n_dof, dtype=np.bool)

        # self.MF_complexesQ      = [MFsubcomplex(mi, ma, self.nMF_per_subgroup) for (mi, ma) in zip(qMins, qMaxs)]
        # self.MF_complexesQdes   = [MFsubcomplex(mi, ma, self.nMF_per_subgroup) for (mi, ma) in zip(qMins, qMaxs)]
        # self.MF_complexesQd     = [MFsubcomplex(mi, ma, self.nMF_per_subgroup) for (mi, ma) in zip(qdMins, qdesMaxs)]
        # self.MF_complexesQddes  = [MFsubcomplex(mi, ma, self.nMF_per_subgroup) for (mi, ma) in zip(qdMins, qdesMaxs)]
        self.mf_out     = np.zeros(self.nMF * n_dof, dtype=np.float64)

        self.pf_pc_wts = np.full((self.nGC * n_dof, self.nPC * n_dof), cerebellum.INIT_PF_PC_WT)
        self.pc_out = np.zeros(self.nPC * n_dof, dtype=bool)

        self.pcNeurons     = LIFPopulation(n=self.nPC * n_dof, params=PC_params, legacy_refractory=KEEP_LEGACY_REFRACTORY)
        self.dcnNeurons    = LIFPopulation(n=self.nDCN * n_dof, params=DCN_params, legacy_refractory=KEEP_LEGACY_REFRACTORY)

        self.nTaps         = max(1, int(round(TORQUE_FILTER_S / dt)))
        self.prevDCN       = np.zeros((self.nTaps, self.nJoints))

        self.timeStep = dt

        # pre-calculate a bunch of these values
        self.kernelLookup = self.makeKernelLookup()


    def makeKernelLookup(self):
        """kernel_table[lag] = ltd_kernel(-lag*dt) for lag = 0..history window (in ticks)."""
        n_lag = self.pf_history.prune_window + 1
        return ltd_kernel(-np.arange(n_lag) * self.timeStep)

    def encode_mf_address(self, state, value_ranges=DEFAULT_MF_VALUE_RANGES):
        """
        Convert (q, qd, q_des, qd_des) into 4 digits in [0, 9], one per MF
        subgroup, via uniform binning.
        Returns
        -------
        tuple of 2 ints, each in [0, N_MF_PER_SUBGROUP - 1]
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
        d0, d1 = digits
        return (d0 + d1 * self.nMF_per_subgroup) + nJoint*self.nGC

    def granularLayer(self, q, qdes, dt):
        self.gc_tonic.fill(0.0)
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
            digits = self.encode_mf_address((q[j], qdes[j],), ((self.qmins[j], self.qmaxs[j]),
                                                            (self.qmins[j], self.qmaxs[j])))
            addressed_index = self.address_to_gc_index(digits, j)
            self.gc_tonic[addressed_index] = self.gc_tonic_value   # constant AMPA conductance (S)
        self.pfs = self.gcNeurons.step(dt=dt, ampa_tonic=self.gc_tonic)


    def errorCalc(self, error):
        max_error = 1 
        return 0.2+0.8*(1-np.exp(-error*90/max_error));

    def climbingFibers(self, qErr, dt):
        self.cf_output.fill(False)
        kp = np.array([3, 3])
        # kp = np.ones(self.nJoints)*0.5
        # kd = np.ones(self.nJoints)*0.5/(2*np.pi)
        sigError = kp * (qErr)
        # pError = self.errorCalc(sigError)
        # nError = self.errorCalc(-sigError)
        pError = sigError
        nError = -sigError
        # positive error -> agonist microzone (DCN half that pushes +). v01 had these swapped, which
        # only worked because its Euler-unstable DCN turned PC inhibition into excitation.
        agonist = [x if y >= 0 else 0 for (x,y) in zip(pError, sigError)]
        antagonist = [x if y < 0 else 0 for (x,y) in zip(nError, sigError)]
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
                lags = current_t - hist_time          # ticks since each PF spike (int, 0..window)
                k_vals = self.kernelLookup[lags]      # vectorized table lookup
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
        pc_ampa_inp = self.gain_pc_ampa * np.sum(self.pf_pc_wts[pf_idx], axis=0)
        self.pc_out = self.pcNeurons.step(dt=dt, ampa_input=pc_ampa_inp)

    def dcnToTorque(self, dcnOut, dt):
        torques = np.sum(dcnOut.reshape(self.nJoints*2, -1), axis=1)
        # spikes per tick shrink as dt shrinks; convert to a rate and re-express per legacy tick
        # so TORQUE_ALPHA means the same thing at any dt.
        torques = torques * (LEGACY_DT / dt)
        torques[1::2] *= -1
        torques = np.sum(torques.reshape(-1, 2), axis=1)
        self.prevDCN[:-1] = self.prevDCN[1:]
        self.prevDCN[-1] = torques
        corr = (self.TORQUE_ALPHA) * np.mean(self.prevDCN, axis=0)
        return corr

    def compute(self, q, qdes, qErr, dt=None):
        # dt is fixed at construction (kernel table, history window, filter taps depend on it)
        if dt is not None and abs(dt - self.dt) > 1e-12:
            raise BrainError(f"compute(dt={dt}) != brain dt={self.dt}; build cerebellum(..., dt=dt)")
        dt = self.dt
        # input fiber layers
        self.granularLayer(q, qdes, dt)
        self.climbingFibers(qErr, dt)
        # now we have spikes from the PFs and the CFs in cf_output and pfs
        self.PCstep(dt)
        self.dcnOut = self.dcnNeurons.step(dt=dt,
                             ampa_input=(self.gain_dcn_ampa*self.W_CF_DCN_AMPA*self.cf_output),
                             ampa_tonic=self.mf_to_dcn,
                             nmda_input=(self.gain_dcn_nmda*self.W_CF_DCN_NMDA*self.cf_output),
                             gaba_input=(self.gain_dcn_gaba*self.pc_out*self.W_PC_DCN))
        # print(self.dcnOut.any())
        corr = self.dcnToTorque(self.dcnOut, dt)
        self.t += 1
        return(corr)