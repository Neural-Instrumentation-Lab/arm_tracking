"""
Brain-machine interfaces: turn the cerebellum's DCN spikes into a joystick command.

In the simulation pipeline the brain is stepped, then the BMI reads `brain.dcnOut`
and decodes the correction torque that is sent to the arm:

    brain.compute(...)               # step the cerebellum
    torque = bmi.decode(brain.dcnOut)

Every BMI is set up from the brain it reads (`bmi.setup(brain)`) before the run, so the
same BMI object can be put in the experiment registry before any brain exists.

DCN layout (same as the brain): dcnOut has nDCN*nJoints entries, split into 2*nJoints
groups of nDCN/2 -- [joint 0 agonist, joint 0 antagonist, joint 1 agonist, ...].
Agonist spikes push the joint positive, antagonist spikes push it negative.
"""
import numpy as np

# timestep TORQUE_ALPHA was tuned at; spike counts are re-expressed per LEGACY_DT tick
# so the gain means the same thing at any brain timestep (same as the brain's own readout)
LEGACY_DT = 2e-3


class BMIError(Exception): pass


class BMI:
    """
    Base class. Subclasses choose how a DCN group's spikes become a number by
    overriding group_counts(); the base class turns those into a filtered torque
    the same way the brain does (rate * TORQUE_ALPHA, mean over the last nTaps ticks).
    """

    def setup(self, brain):
        """Read sizes and gains from the brain and clear the filter. Call before every run."""
        self.nJoints = brain.nJoints
        self.nPerGroup = brain.nDCN // 2
        self.dt = brain.dt
        self.gain = np.asarray(brain.TORQUE_ALPHA, dtype=np.float64)
        self.prevDCN = np.zeros((brain.nTaps, self.nJoints))
        return self

    def group_counts(self, groups):
        """groups: bool array (2*nJoints, nDCN/2). Returns the spike count of each group."""
        raise NotImplementedError

    def decode(self, dcnOut):
        """DCN spikes from one brain tick -> torque command, shape (nJoints,)."""
        groups = np.asarray(dcnOut, dtype=bool).reshape(self.nJoints * 2, self.nPerGroup)
        counts = self.group_counts(groups) * (LEGACY_DT / self.dt)
        torques = counts[0::2] - counts[1::2]          # agonist - antagonist, per joint
        self.prevDCN[:-1] = self.prevDCN[1:]
        self.prevDCN[-1] = torques
        return self.gain * np.mean(self.prevDCN, axis=0)


class FullBMI(BMI):
    """Records every DCN. Decodes exactly like the brain's own dcnToTorque."""

    def group_counts(self, groups):
        return groups.sum(axis=1).astype(np.float64)

    def __repr__(self):
        return "FullBMI()"


class DroppedDCNBMI(BMI):
    """
    Records only some DCNs: n_drop random DCNs are dropped from every agonist/antagonist
    group (each group loses its own random cells). The remaining counts are rescaled to the
    full group size, like a rate decoder, so the command gain doesn't depend on how many
    DCNs are recorded -- only its noise does.
    """

    def __init__(self, n_drop, seed=None):
        self.n_drop = int(n_drop)
        self.seed = seed

    def setup(self, brain):
        super().setup(brain)
        if not 0 <= self.n_drop < self.nPerGroup:
            raise BMIError(f"n_drop must be between 0 and {self.nPerGroup - 1} (got {self.n_drop})")
        rng = np.random.default_rng(self.seed)
        n_keep = self.nPerGroup - self.n_drop
        self.kept = np.array([np.sort(rng.choice(self.nPerGroup, size=n_keep, replace=False))
                              for _ in range(self.nJoints * 2)])
        return self

    def group_counts(self, groups):
        kept = np.take_along_axis(groups, self.kept, axis=1)
        return kept.sum(axis=1) * (self.nPerGroup / kept.shape[1])

    def __repr__(self):
        return f"DroppedDCNBMI(n_drop={self.n_drop}, seed={self.seed})"
