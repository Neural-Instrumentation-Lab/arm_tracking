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


# what the BMI does with each DCN, see BMI.dcn_roles(); used to colour the DCN plots
DCN_ROLES = ("right", "wrong", "both", "unread")


class BMIError(Exception): pass


class ProbabilityError(BMIError):
    """A BMI's probabilities don't add up to 1 (dcn_sweep.py skips those combinations)."""


def resolve_probs(probs):
    """
    probs: {name: probability or None}, which must add up to 1. Any one can be None and is then
    1 minus the others. Returns the filled-in dict.
    """
    names = ", ".join(probs)
    missing = [n for n, v in probs.items() if v is None]
    if len(missing) > 1:
        given = [n for n in probs if n not in missing]
        raise BMIError(f"give at least {len(probs) - 1} of {names} (the last is 1 minus the others); "
                       f"got {', '.join(given) or 'none'}")
    probs = {n: None if v is None else float(v) for n, v in probs.items()}
    for n, v in probs.items():
        if v is not None and not 0 <= v <= 1:
            raise BMIError(f"{n} must be between 0 and 1 (got {v:g})")
    total = sum(v for v in probs.values() if v is not None)
    shown = ", ".join(f"{n}={v:g}" for n, v in probs.items() if v is not None)
    if missing:
        if total > 1 + 1e-9:
            raise ProbabilityError(f"{shown} add up to {total:g}, more than 1, leaving nothing for {missing[0]}")
        probs[missing[0]] = max(1 - total, 0.0)
    elif abs(total - 1) > 1e-9:
        raise ProbabilityError(f"{names} must add up to 1 ({shown} add up to {total:g})")
    return probs


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

    def dcn_roles(self):
        """
        How the BMI reads each DCN, as an index into DCN_ROLES, laid out like dcnOut (and like the
        PCs, since PC i inhibits DCN i): read only as its own muscle ("right"), only as another
        muscle ("wrong"), as both, or not at all ("unread"). Call after setup(). None for BMIs that
        read every DCN as its own muscle.
        """
        return None

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


class CrossClusterBMI(BMI):
    """
    Like ClusterBMI, but a cluster can pick up DCNs from other muscles (the 2*nJoints
    agonist/antagonist groups). Each cluster slot is still read from the same DCN position, but
    with probability right_prob that DCN belongs to the cluster's own muscle; otherwise it belongs
    to one of the other muscles, chosen uniformly. So right_prob = 1 is ClusterBMI and
    right_prob = 1 / (2*nJoints) (0.25 with 2 joints) is a cluster of DCNs from random muscles.
    The wiring is drawn once in setup() and fixed for the run.
    """

    def __init__(self, cluster_size, right_prob, seed=None):
        self.cluster_size = int(cluster_size)
        self.right_prob = float(right_prob)
        self.seed = seed

    def setup(self, brain):
        super().setup(brain)
        if not 1 <= self.cluster_size <= self.nPerGroup:
            raise BMIError(f"cluster_size must be between 1 and {self.nPerGroup} (got {self.cluster_size})")
        if not 0 <= self.right_prob <= 1:
            raise BMIError(f"right_prob must be between 0 and 1 (got {self.right_prob})")
        self.starts = np.arange(0, self.nPerGroup, self.cluster_size)
        nGroups = self.nJoints * 2
        rng = np.random.default_rng(self.seed)
        own = np.arange(nGroups)[:, None]
        # wrong muscle: an offset of 1..nGroups-1 from the right one, so every other muscle is equally likely
        wrong = (own + rng.integers(1, nGroups, size=(nGroups, self.nPerGroup))) % nGroups
        right = rng.random((nGroups, self.nPerGroup)) < self._dcn_right_prob()
        self.source = np.where(right, own, wrong)                # muscle each slot reads from
        self.slot = np.broadcast_to(np.arange(self.nPerGroup), self.source.shape)
        return self

    def dcn_roles(self):
        n_groups = self.nJoints * 2
        own = np.arange(n_groups)[:, None]
        reads_dcn = ~getattr(self, "noise", np.zeros(self.source.shape, dtype=bool))   # random-neuron slots read no DCN
        right = np.zeros((n_groups, self.nPerGroup), dtype=bool)
        wrong = np.zeros_like(right)
        is_right = reads_dcn & (self.source == own)
        is_wrong = reads_dcn & (self.source != own)
        right[self.source[is_right], self.slot[is_right]] = True
        wrong[self.source[is_wrong], self.slot[is_wrong]] = True
        roles = np.full(right.shape, DCN_ROLES.index("unread"))
        roles[right] = DCN_ROLES.index("right")
        roles[wrong] = DCN_ROLES.index("wrong")
        roles[right & wrong] = DCN_ROLES.index("both")
        return roles.ravel()

    def _dcn_right_prob(self):
        """Chance a slot that reads a DCN reads one of its own muscle."""
        return self.right_prob

    def read(self, groups):
        """What each cluster slot sees this tick: (2*nJoints, nDCN/2) bool, laid out like groups."""
        return groups[self.source, self.slot]

    def group_counts(self, groups):
        active = np.logical_or.reduceat(self.read(groups), self.starts, axis=1)
        return active.sum(axis=1) * (self.nPerGroup / len(self.starts))

    def __repr__(self):
        return (f"CrossClusterBMI(cluster_size={self.cluster_size}, "
                f"right_prob={self.right_prob}, seed={self.seed})")


class NoisyCrossClusterBMI(CrossClusterBMI):
    """
    Like CrossClusterBMI, but some cluster slots pick up a random neuron instead of a DCN. Each slot
    is a DCN of the cluster's own muscle with probability right_prob, a DCN of another muscle (chosen
    uniformly) with probability cross_prob, or a random neuron with probability noise_prob. The three
    must add up to 1: give any two and the third is 1 minus their sum, or give all three. A random
    neuron fires independently every tick with probability noise_rate * dt, so on average at
    noise_rate Hz (at most 1/dt, e.g. 500 Hz at dt = 2 ms). Untrained DCNs fire at about 500 Hz and
    trained ones a few hundred Hz less. The wiring is drawn once in setup() and fixed for the run.
    """

    def __init__(self, cluster_size, noise_rate, right_prob=None, cross_prob=None, noise_prob=None, seed=None):
        probs = resolve_probs({"right_prob": right_prob, "cross_prob": cross_prob, "noise_prob": noise_prob})
        super().__init__(cluster_size, probs["right_prob"], seed=seed)
        self.cross_prob = probs["cross_prob"]
        self.noise_prob = probs["noise_prob"]
        self.noise_rate = float(noise_rate)

    def _dcn_right_prob(self):
        # the noise slots are drawn separately, so among the DCN slots right : cross keeps its ratio
        dcn = self.right_prob + self.cross_prob
        return self.right_prob / dcn if dcn > 0 else 1.0

    def setup(self, brain):
        super().setup(brain)
        if not 0 <= self.noise_rate * self.dt <= 1:
            raise BMIError(f"noise_rate must be between 0 and {1 / self.dt:g} Hz at dt={self.dt:g} s "
                           f"(got {self.noise_rate:g})")
        # own stream, so the noise doesn't reuse the random numbers that drew the DCN wiring
        self.noise_rng = np.random.default_rng(None if self.seed is None else [self.seed, 1])
        self.noise = self.noise_rng.random(self.source.shape) < self.noise_prob   # slots that are random neurons
        self.spike_prob = self.noise_rate * self.dt
        return self

    def read(self, groups):
        seen = super().read(groups)
        if not self.noise.any():
            return seen
        spikes = self.noise_rng.random(seen.shape) < self.spike_prob
        return np.where(self.noise, spikes, seen)

    def __repr__(self):
        return (f"NoisyCrossClusterBMI(cluster_size={self.cluster_size}, noise_rate={self.noise_rate:g}, "
                f"right_prob={self.right_prob:g}, cross_prob={self.cross_prob:g}, "
                f"noise_prob={self.noise_prob:g}, seed={self.seed})")


class NoisyClusterBMI(NoisyCrossClusterBMI):
    """
    Like ClusterBMI, but some cluster slots pick up a random neuron instead of a DCN: each slot is a
    DCN of the cluster's own muscle with probability right_prob, or a random neuron with probability
    noise_prob. The two must add up to 1: give either one, or both. A random neuron fires
    independently every tick with probability noise_rate * dt, so on average at noise_rate Hz (at
    most 1/dt, e.g. 500 Hz at dt = 2 ms). Untrained DCNs fire at about 500 Hz and trained ones a few
    hundred Hz less. noise_prob = 0 is ClusterBMI.
    """

    def __init__(self, cluster_size, noise_rate, right_prob=None, noise_prob=None, seed=None):
        probs = resolve_probs({"right_prob": right_prob, "noise_prob": noise_prob})
        super().__init__(cluster_size, noise_rate, cross_prob=0, seed=seed, **probs)

    def __repr__(self):
        return (f"NoisyClusterBMI(cluster_size={self.cluster_size}, noise_rate={self.noise_rate:g}, "
                f"right_prob={self.right_prob:g}, noise_prob={self.noise_prob:g}, seed={self.seed})")


class NoisyMixBMI(NoisyCrossClusterBMI):
    """
    Records one neuron per DCN slot of each muscle (the 2*nJoints agonist/antagonist groups, 4 with 2
    joints), but some of those neurons are the wrong one. Each recorded neuron is a DCN of its own
    muscle with probability right_prob, a DCN of another muscle (chosen uniformly) with probability
    cross_prob, or a random neuron with probability noise_prob. The three must add up to 1: give any
    two and the third is 1 minus their sum, or give all three. Unlike the cluster BMIs every neuron
    counts on its own, so a muscle's count is how many of its recorded neurons spiked (like FullBMI,
    which is right_prob = 1). A random neuron fires independently every tick with probability
    noise_rate * dt, so on average at noise_rate Hz (at most 1/dt, e.g. 500 Hz at dt = 2 ms).
    Untrained DCNs fire at about 500 Hz and trained ones a few hundred Hz less. The wiring is drawn
    once in setup() and fixed for the run.
    """

    def __init__(self, noise_rate, right_prob=None, cross_prob=None, noise_prob=None, seed=None):
        super().__init__(1, noise_rate, right_prob=right_prob, cross_prob=cross_prob,
                         noise_prob=noise_prob, seed=seed)

    def __repr__(self):
        return (f"NoisyMixBMI(noise_rate={self.noise_rate:g}, right_prob={self.right_prob:g}, "
                f"cross_prob={self.cross_prob:g}, noise_prob={self.noise_prob:g}, seed={self.seed})")


class NeurodesBMI(BMI):
    """
    Each muscle (the 2*nJoints agonist/antagonist groups) is read by n_neurodes neurodes, and each
    neurode works like one noisy cross-talk cluster: it has cluster_size slots, and it spikes in a
    tick if any of its slots spiked. Each slot is a DCN of the neurode's own muscle with probability
    right_prob, a DCN of another muscle (chosen uniformly) with probability cross_prob, or a random
    neuron with probability noise_prob. The three must add up to 1: give any two and the third is 1
    minus their sum, or give all three. A random neuron fires independently every tick with
    probability noise_rate * dt, so on average at noise_rate Hz (at most 1/dt, e.g. 500 Hz at
    dt = 2 ms).

    A neurode's slots read cluster_size different DCN positions, picked at random (so neurodes can
    share DCNs, and n_neurodes * cluster_size can be more than the group size). Like SpikeDropBMI,
    every neurode spike is then lost independently with probability drop_prob.

    The count of surviving neurode spikes is scaled by nDCNs / n_neurodes, so all neurodes spiking
    reads the same as all DCNs spiking in the normal brain, and divided by (1 - drop_prob) so the
    command matches on average, capped at the group size. The wiring is drawn once in setup() and
    fixed for the run.
    """

    def __init__(self, n_neurodes, cluster_size, noise_rate, right_prob=None, cross_prob=None,
                 noise_prob=None, drop_prob=0.0, seed=None):
        probs = resolve_probs({"right_prob": right_prob, "cross_prob": cross_prob, "noise_prob": noise_prob})
        self.n_neurodes = int(n_neurodes)
        self.cluster_size = int(cluster_size)
        self.noise_rate = float(noise_rate)
        self.right_prob = probs["right_prob"]
        self.cross_prob = probs["cross_prob"]
        self.noise_prob = probs["noise_prob"]
        self.drop_prob = float(drop_prob)
        self.seed = seed

    def setup(self, brain):
        super().setup(brain)
        if self.n_neurodes < 1:
            raise BMIError(f"n_neurodes must be at least 1 (got {self.n_neurodes})")
        if not 1 <= self.cluster_size <= self.nPerGroup:
            raise BMIError(f"cluster_size must be between 1 and {self.nPerGroup} (got {self.cluster_size})")
        if not 0 <= self.noise_rate * self.dt <= 1:
            raise BMIError(f"noise_rate must be between 0 and {1 / self.dt:g} Hz at dt={self.dt:g} s "
                           f"(got {self.noise_rate:g})")
        if not 0 <= self.drop_prob < 1:
            raise BMIError(f"drop_prob must be at least 0 and below 1 (got {self.drop_prob})")

        nGroups = self.nJoints * 2
        shape = (nGroups, self.n_neurodes, self.cluster_size)
        rng = np.random.default_rng(self.seed)
        # DCN position of each slot: cluster_size different positions per neurode
        order = np.argsort(rng.random((nGroups, self.n_neurodes, self.nPerGroup)), axis=-1)
        self.slot = order[..., :self.cluster_size]
        # what each slot is: right DCN, cross-talk DCN or random neuron
        kind = rng.random(shape)
        own = np.arange(nGroups)[:, None, None]
        # wrong muscle: an offset of 1..nGroups-1 from the right one, so every other muscle is equally likely
        wrong = (own + rng.integers(1, nGroups, size=shape)) % nGroups if nGroups > 1 else np.broadcast_to(own, shape)
        self.source = np.where(kind < self.right_prob, own, wrong)              # muscle each slot reads from
        self.noise = kind >= self.right_prob + self.cross_prob                  # slots that are random neurons
        # own stream, so the per-tick noise and drops don't reuse the random numbers that drew the wiring
        self.rng = np.random.default_rng(None if self.seed is None else [self.seed, 1])
        self.spike_prob = self.noise_rate * self.dt
        return self

    def dcn_roles(self):
        n_groups = self.nJoints * 2
        own = np.arange(n_groups)[:, None, None]
        right = np.zeros((n_groups, self.nPerGroup), dtype=bool)
        wrong = np.zeros_like(right)
        is_right = ~self.noise & (self.source == own)
        is_wrong = ~self.noise & (self.source != own)
        right[self.source[is_right], self.slot[is_right]] = True
        wrong[self.source[is_wrong], self.slot[is_wrong]] = True
        roles = np.full(right.shape, DCN_ROLES.index("unread"))
        roles[right] = DCN_ROLES.index("right")
        roles[wrong] = DCN_ROLES.index("wrong")
        roles[right & wrong] = DCN_ROLES.index("both")
        return roles.ravel()

    def group_counts(self, groups):
        seen = groups[self.source, self.slot]                   # (nGroups, n_neurodes, cluster_size)
        if self.noise.any():
            seen = np.where(self.noise, self.rng.random(seen.shape) < self.spike_prob, seen)
        fired = seen.any(axis=2)                                # neurode spikes
        if self.drop_prob > 0:
            fired &= self.rng.random(fired.shape) >= self.drop_prob
        counts = fired.sum(axis=1) * (self.nPerGroup / self.n_neurodes) / (1 - self.drop_prob)
        return np.minimum(counts, self.nPerGroup)

    def __repr__(self):
        return (f"NeurodesBMI(n_neurodes={self.n_neurodes}, cluster_size={self.cluster_size}, "
                f"noise_rate={self.noise_rate:g}, right_prob={self.right_prob:g}, "
                f"cross_prob={self.cross_prob:g}, noise_prob={self.noise_prob:g}, "
                f"drop_prob={self.drop_prob:g}, seed={self.seed})")


@dataclass(frozen=True)
class SweepParam:
    """One setting of a BMI that dcn_sweep.py can vary."""
    default: float | None      # value used when this setting isn't swept (None: worked out by the BMI)
    values: tuple              # values swept by default
    label: str                 # what a value means, used on the plots
    whole: bool = False        # must be a whole number (e.g. a count of DCNs)


@dataclass(frozen=True)
class BMISweep:
    """
    How dcn_sweep.py builds one kind of BMI. Add an entry to BMI_SWEEPS to make a new BMI sweepable:
    make(seed, **settings) returns the BMI, and params names each setting it takes. The first
    entry of params is the one swept when dcn_sweep.py isn't told which to sweep.

    linked lists groups of settings that depend on each other, like probabilities that must add up
    to 1: as soon as any setting of a group is set or swept, the others drop their defaults (they
    become None, for the BMI to work out), so a default never silently fills in one of them.

    The BMI must store each setting as an attribute of the same name: dcn_sweep.py reads them back
    so its results show the values the BMI actually used (e.g. a probability it worked out).
    """
    make: Callable             # (seed, **settings) -> BMI
    params: dict               # setting name -> SweepParam
    linked: tuple = ()         # tuples of setting names, see above

    def build(self, seed, **settings):
        """The BMI for these settings; settings left out use their defaults."""
        unknown = set(settings) - set(self.params)
        if unknown:
            raise BMIError(f"unknown setting(s) {sorted(unknown)}; this BMI takes {list(self.params)}")
        full = {}
        for name, param in self.params.items():
            value = settings.get(name, param.default)
            if param.whole and value is not None:
                if float(value) != int(value):
                    raise BMIError(f"{name} must be a whole number (got {value})")
                value = int(value)
            full[name] = value
        return self.make(seed, **full)


_N_PER_GROUP = "(of 50 per agonist/antagonist group)"
_NOISE_PROB = SweepParam(0.25, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1), "Probability a cluster slot is a random neuron")
_CLUSTER_SIZE = SweepParam(10, (1, 2, 5, 10, 25, 50), f"DCNs per cluster {_N_PER_GROUP}", whole=True)
_NOISE_RATE = SweepParam(100, (0, 10, 50, 100, 200, 350, 500), "Firing rate of the random neurons (Hz)")

BMI_SWEEPS = {
    "num_drop": BMISweep(
        lambda seed, n_drop: DroppedDCNBMI(n_drop, seed=seed),
        {"n_drop": SweepParam(0, (0, 20, 30, 40, 45, 48, 49), f"DCNs dropped per group {_N_PER_GROUP}", whole=True)}),
    "spike_drop": BMISweep(
        lambda seed, drop_prob: SpikeDropBMI(drop_prob, seed=seed),
        {"drop_prob": SweepParam(0, (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98), "Probability of dropping each DCN spike")}),
    "cluster": BMISweep(
        lambda seed, cluster_size: ClusterBMI(cluster_size),
        {"cluster_size": SweepParam(1, (1, 2, 5, 10, 25, 50), f"DCNs per cluster {_N_PER_GROUP}", whole=True)}),
    "cross_cluster": BMISweep(
        lambda seed, right_prob, cluster_size: CrossClusterBMI(cluster_size, right_prob, seed=seed),
        {"right_prob": SweepParam(1, (0.25, 0.5, 0.75, 0.9, 1), "Probability a clustered DCN is from the right muscle"),
         "cluster_size": SweepParam(10, (1, 2, 5, 10, 25, 50), f"DCNs per cluster {_N_PER_GROUP}", whole=True)}),
    "noisy_cluster": BMISweep(
        lambda seed, noise_prob, noise_rate, right_prob, cluster_size:
            NoisyClusterBMI(cluster_size, noise_rate, right_prob=right_prob, noise_prob=noise_prob, seed=seed),
        {"noise_prob": SweepParam(0.25, _NOISE_PROB.values, _NOISE_PROB.label),
         "noise_rate": _NOISE_RATE,
         "right_prob": SweepParam(None, (0, 0.25, 0.5, 0.75, 0.9, 1), "Probability a cluster slot is a DCN of the right muscle"),
         "cluster_size": _CLUSTER_SIZE},
        linked=(("right_prob", "noise_prob"),)),
    "noisy_cross_cluster": BMISweep(
        lambda seed, noise_rate, right_prob, cross_prob, noise_prob, cluster_size:
            NoisyCrossClusterBMI(cluster_size, noise_rate, right_prob=right_prob, cross_prob=cross_prob,
                                 noise_prob=noise_prob, seed=seed),
        {"noise_rate": _NOISE_RATE,
         "right_prob": SweepParam(0.5, (0, 0.25, 0.5, 0.75, 0.9, 1), "Probability a cluster slot is a DCN of the right muscle"),
         "cross_prob": SweepParam(None, (0, 0.1, 0.25, 0.5, 0.75, 1), "Probability a cluster slot is a DCN of another muscle"),
         "noise_prob": SweepParam(0.25, _NOISE_PROB.values, _NOISE_PROB.label),
         "cluster_size": _CLUSTER_SIZE},
        linked=(("right_prob", "cross_prob", "noise_prob"),)),
    "noisy_mix": BMISweep(
        lambda seed, noise_rate, right_prob, cross_prob, noise_prob:
            NoisyMixBMI(noise_rate, right_prob=right_prob, cross_prob=cross_prob, noise_prob=noise_prob, seed=seed),
        {"noise_rate": _NOISE_RATE,
         "right_prob": SweepParam(0.5, (0, 0.25, 0.5, 0.75, 0.9, 1), "Probability a recorded neuron is a DCN of the right muscle"),
         "cross_prob": SweepParam(None, (0, 0.1, 0.25, 0.5, 0.75, 1), "Probability a recorded neuron is a DCN of another muscle"),
         "noise_prob": SweepParam(0.25, _NOISE_PROB.values, "Probability a recorded neuron is a random neuron")},
        linked=(("right_prob", "cross_prob", "noise_prob"),)),
    "neurodes": BMISweep(
        lambda seed, n_neurodes, cluster_size, right_prob, noise_prob, noise_rate, cross_prob, drop_prob:
            NeurodesBMI(n_neurodes, cluster_size, noise_rate, right_prob=right_prob, cross_prob=cross_prob,
                        noise_prob=noise_prob, drop_prob=drop_prob, seed=seed),
        {"n_neurodes": SweepParam(10, (1, 2, 5, 10, 25, 50, 100), "Neurodes per agonist/antagonist group", whole=True),
         "cluster_size": SweepParam(5, (1, 2, 5, 10, 25, 50), f"Neurons per neurode {_N_PER_GROUP}", whole=True),
         "right_prob": SweepParam(0.5, (0, 0.25, 0.5, 0.75, 0.9, 1), "Probability a neurode slot is a DCN of the right muscle"),
         "noise_prob": SweepParam(0.25, _NOISE_PROB.values, "Probability a neurode slot is a random neuron"),
         "noise_rate": _NOISE_RATE,
         "cross_prob": SweepParam(None, (0, 0.1, 0.25, 0.5, 0.75, 1), "Probability a neurode slot is a DCN of another muscle"),
         "drop_prob": SweepParam(0, (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98), "Probability of dropping each neurode spike")},
        linked=(("right_prob", "cross_prob", "noise_prob"),)),
}
