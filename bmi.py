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
from dataclasses import dataclass
from typing import Callable

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


class SpikeDropBMI(BMI):
    """
    Loses every DCN spike independently with probability drop_prob (an unreliable recording).
    The surviving count of each group is divided by (1 - drop_prob) so the command matches on
    average, but capped at the group size: the most the normal brain could produce in one tick.
    """

    def __init__(self, drop_prob, seed=None):
        self.drop_prob = float(drop_prob)
        self.seed = seed

    def setup(self, brain):
        super().setup(brain)
        if not 0 <= self.drop_prob < 1:
            raise BMIError(f"drop_prob must be at least 0 and below 1 (got {self.drop_prob})")
        self.rng = np.random.default_rng(self.seed)
        return self

    def group_counts(self, groups):
        kept = groups & (self.rng.random(groups.shape) >= self.drop_prob)
        return np.minimum(kept.sum(axis=1) / (1 - self.drop_prob), self.nPerGroup)

    def __repr__(self):
        return f"SpikeDropBMI(drop_prob={self.drop_prob}, seed={self.seed})"


class ClusterBMI(BMI):
    """
    Can't tell the DCNs of a cluster apart: each agonist/antagonist group is split into clusters of
    cluster_size neighbouring DCNs (the last is smaller if the size doesn't divide the group), and a
    cluster registers one spike if any of its DCNs spiked. The count is scaled by nDCNs / nClusters,
    so all clusters spiking reads the same as all DCNs spiking in the normal brain.
    """

    def __init__(self, cluster_size):
        self.cluster_size = int(cluster_size)

    def setup(self, brain):
        super().setup(brain)
        if not 1 <= self.cluster_size <= self.nPerGroup:
            raise BMIError(f"cluster_size must be between 1 and {self.nPerGroup} (got {self.cluster_size})")
        self.starts = np.arange(0, self.nPerGroup, self.cluster_size)
        return self

    def group_counts(self, groups):
        active = np.logical_or.reduceat(groups, self.starts, axis=1)
        return active.sum(axis=1) * (self.nPerGroup / len(self.starts))

    def __repr__(self):
        return f"ClusterBMI(cluster_size={self.cluster_size})"


def _whole(value, name):
    if float(value) != int(value):
        raise BMIError(f"{name} values must be whole numbers of DCNs (got {value})")
    return int(value)


def _make_num_drop(value, seed):
    return DroppedDCNBMI(_whole(value, "num_drop"), seed=seed)


def _make_cluster(value, seed):
    return ClusterBMI(_whole(value, "cluster"))


@dataclass(frozen=True)
class BMISweep:
    """How dcn_sweep.py varies one kind of BMI. Add an entry to BMI_SWEEPS to make a new BMI sweepable."""
    make: Callable             # (value, seed) -> BMI
    values: tuple              # values swept by default
    label: str                 # what a value means, used on the plots


BMI_SWEEPS = {
    "num_drop": BMISweep(_make_num_drop, (0, 20, 30, 40, 45, 48, 49),
                         "DCNs dropped per agonist/antagonist group (of 50)"),
    "spike_drop": BMISweep(SpikeDropBMI, (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98),
                           "Probability of dropping each DCN spike"),
    "cluster": BMISweep(_make_cluster, (1, 2, 5, 10, 25, 50),
                        "DCNs per cluster (of 50 per agonist/antagonist group)"),
}
