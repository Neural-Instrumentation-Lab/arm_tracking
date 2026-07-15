from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import pandas as pd
import numpy as np
import yaml
from typing import Literal

WeightGroup = Literal["pf_pc", "mf_dcn", "pc_dcn"]
ALL_WEIGHT_GROUPS: tuple[WeightGroup, ...] = ("pf_pc", "mf_dcn", "pc_dcn")

@dataclass(frozen=True)
class Experiment:
    name: str                # short, unique, CLI-friendly id (e.g. "baseline_v1")
    description: str         # one-line human explanation, shown in --list
    trajectory: Path         # the 1 trajectory file
    actualArm: Path          # the 2 arm files
    illusoryArm: Path
    results: Path            # where .pkl of arm trajectories is stored
    finalWts : Path          # where .npz of brain weights is stored
    graphs : Path            # where .pngs of results are stored 
    nDof : int # how many degrees of freedom arm has
    nTrials : int # how many times the traj is repeated
    load_weight_groups: tuple[WeightGroup, ...] = field(default_factory=tuple) # which sites are to be loaded from initialWts file
    initialWts: Path | None = None        # .npz of initial weights of brain, only sites specified from load_weight_groups loaded
    defaultWts: Path | None = None        # .npz file with initial weights of the sites not in load_weight_groups. If not given defaults to 0s
    plastic: tuple[WeightGroup, ...] = ALL_WEIGHT_GROUPS # what sites in the brain are plastic

    def validate(self, assume_produced: frozenset[Path] = frozenset()) -> list[str]:
        """Return a list of problems (missing files etc). Empty list = OK.
 
        Args:
            assume_produced: paths to treat as OK even though they don't exist
                yet — e.g. when batch-running a set of experiments where one
                produces a results file another consumes as load_weights_from.
                Standalone/single-experiment runs should leave this empty.
        """
        problems = []
        for label, path in [
            ("trajectory", self.trajectory),
            ("actualArm", self.actualArm),
            ("illusoryArm", self.illusoryArm),
        ]:
            if not path.exists():
                problems.append(f"{label} file not found: {path}")
 
        unknown = set(self.load_weight_groups) - set(ALL_WEIGHT_GROUPS)
        if unknown:
            problems.append(
                f"load_weight_groups has unknown group(s) {sorted(unknown)}; "
                f"valid options are {ALL_WEIGHT_GROUPS}"
            )
        if self.load_weight_groups and self.initialWts is None:
            problems.append("load_weight_groups is set but initialWts is None")
        if self.initialWts is not None:
            already_exists = self.initialWts.exists()
            will_be_produced = self.initialWts in assume_produced
            if not already_exists and not will_be_produced:
                problems.append(f"initialWts file not found: {self.initialWts}")
        if self.defaultWts is not None:
            already_exists = self.defaultWts.exists()
            will_be_produced = self.defaultWts in assume_produced
            if not already_exists and not will_be_produced:
                problems.append(f"defaultWts file not found: {self.defaultWts}")
        return problems


def build_dependencies(to_run: list[Experiment], all_experiments: dict[str, Experiment]) -> dict[str, set[str]]:
    """Map each experiment name -> names of experiments whose results it needs
    first, inferred by matching initialWts against another experiment's
    results path. Only creates an edge if the file doesn't already exist on
    disk — if it's already there, there's no ordering requirement."""
    producer_by_results = {
        exp.finalWts: exp.name for exp in all_experiments.values() if exp.finalWts is not None
    }
    deps: dict[str, set[str]] = {}
    for exp in to_run:
        dep_names = set()
        if exp.initialWts is not None and not exp.initialWts.exists():
            producer = producer_by_results.get(exp.initialWts)
            if producer is not None and producer != exp.name:
                dep_names.add(producer)
        if exp.defaultWts is not None and not exp.defaultWts.exists():
            producer = producer_by_results.get(exp.defaultWts)
            if producer is not None and producer != exp.name:
                dep_names.add(producer)
        deps[exp.name] = dep_names
    return deps

"""
Trajectory loading with a self-describing column convention.
 
Instead of hardcoding "column 2 is x, column 5 is joint velocity", trajectory
CSVs use a column-name convention, and this module groups columns by prefix
automatically. Anything that can't be inferred from column names (units,
coordinate frame, notes) lives in an optional YAML sidecar file next to the
CSV.
 
--------------------------------------------------------------------------
COLUMN NAMING CONVENTION (all columns except `time` are optional)
--------------------------------------------------------------------------
    time                          required, seconds
 
    pos_x, pos_y, pos_z           end-effector position   (any subset of xyz)
    vel_x, vel_y, vel_z           end-effector velocity
    acc_x, acc_y, acc_z           end-effector acceleration
 
    joint_pos_1, joint_pos_2, ... joint positions, 1-indexed, any count
    joint_vel_1, joint_vel_2, ...  joint velocities
 
Add a new kind of data later (e.g. torque) by adding a new prefix group
below — existing files and callers are unaffected.
 
--------------------------------------------------------------------------
OPTIONAL SIDECAR: traj_004.csv  ->  traj_004.yaml (same stem, .yaml)
--------------------------------------------------------------------------
    units: meters
    coordinate_frame: world
    sample_rate_hz: 100
    notes: recalibrated 2024-03-01, see experiment log
 
If no sidecar exists, `Trajectory.meta` is just an empty dict — nothing
breaks, the sidecar is purely additive.
"""
 
 
# Ordered so position/velocity/acceleration triples come out as [x, y, z]
_AXES = ["x", "y", "z"]
_JOINT_PATTERN = re.compile(r"^joint_(pos|vel|acc)_(\d+)$")
 
@dataclass
class Trajectory:
    """All fields are plain numpy arrays (pandas is only used internally to parse
    the CSV header). time is shape (N,); position/velocity/acceleration are shape
    (N, k) for k in 1..3; joint_position/joint_velocity are shape (N, n_dof)."""
 
    time: np.ndarray
    position: np.ndarray | None             # columns x, y, z (whichever present)
    velocity: np.ndarray | None
    acceleration: np.ndarray | None
    joint_position: np.ndarray | None       # columns 1..N, sorted numerically
    joint_velocity: np.ndarray | None
    joint_acceleration: np.ndarray | None
    meta: dict = field(default_factory=dict)
    source: Path | None = None
 
    @property
    def n_dof(self) -> int | None:
        """Inferred joint count, if joint columns are present."""
        if self.joint_position is not None:
            return self.joint_position.shape[1]
        if self.joint_velocity is not None:
            return self.joint_velocity.shape[1]
        if self.joint_acceleration is not None:
            return self.joint_acceleration.shape[1]
        return None
 
    @property
    def available_fields(self) -> list[str]:
        fields = ["time"]
        for name in ["position", "velocity", "acceleration", "joint_position", "joint_velocity", "joint_acceleration"]:
            if getattr(self, name) is not None:
                fields.append(name)
        return fields
 
    def __repr__(self) -> str:
        return (
            f"Trajectory(source={self.source}, n_samples={len(self.time)}, "
            f"fields={self.available_fields}, n_dof={self.n_dof})"
        )
 
 
def _extract_axes(df: pd.DataFrame, prefix: str) -> np.ndarray | None:
    cols = [f"{prefix}_{axis}" for axis in _AXES if f"{prefix}_{axis}" in df.columns]
    if not cols:
        return None
    return df[cols].to_numpy()
 
 
def _extract_joint_group(df: pd.DataFrame, kind: str) -> np.ndarray | None:
    """kind is 'pos' or 'vel'. Picks up joint_pos_1, joint_pos_2, ... in numeric order."""
    matches = []
    for col in df.columns:
        m = _JOINT_PATTERN.match(col)
        if m and m.group(1) == kind:
            matches.append((int(m.group(2)), col))
    if not matches:
        return None
    matches.sort(key=lambda t: t[0])
    ordered_cols = [col for _, col in matches]
    return df[ordered_cols].to_numpy()
 
 
def load_trajectory(path: Path | str) -> Trajectory:
    path = Path(path)
    df = pd.read_csv(path)
 
    if "time" not in df.columns:
        raise ValueError(f"{path}: trajectory CSV must have a 'time' column")
 
    meta_path = path.with_suffix(".yaml")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
 
    return Trajectory(
        time=df["time"].to_numpy(),
        position=_extract_axes(df, "pos"),
        velocity=_extract_axes(df, "vel"),
        acceleration=_extract_axes(df, "acc"),
        joint_position=_extract_joint_group(df, "pos"),
        joint_velocity=_extract_joint_group(df, "vel"),
        joint_acceleration=_extract_joint_group(df, "acc"),
        meta=meta,
        source=path,
    )

"""
Saving/loading final brain weights.
 
Weights are a single snapshot (not a time series like trajectories), made up
of a few named groups of possibly different lengths:
    pf_pc   - parallel fiber -> Purkinje cell weights, length n_pf
    mf_dcn  - mossy fiber -> DCN weights, length n_muscle
    pc_dcn  - Purkinje cell -> DCN weights, length n_muscle
 
Stored as a numpy .npz archive (each group kept as its own named array, no
flattening/reassembly needed) plus an optional YAML sidecar for anything
that isn't inferable from the arrays themselves (e.g. training notes).
"""
@dataclass
class Weights:
    pf_pc: np.ndarray    # shape (n_pf,)
    mf_dcn: np.ndarray   # shape (n_muscle,)
    pc_dcn: np.ndarray   # shape (n_muscle,)
    meta: dict = field(default_factory=dict)
    source: Path | None = None
 
    @property
    def n_pf(self) -> int:
        return self.pf_pc.shape[0]
 
    @property
    def n_muscle(self) -> int:
        return self.mf_dcn.shape[0]
 
    def __repr__(self) -> str:
        return (
            f"Weights(source={self.source}, n_pf={self.n_pf}, "
            f"n_muscle={self.n_muscle}, total={self.n_pf + 2 * self.n_muscle})"
        )
 
def save_weights(
    fname: str | Path,
    pf_pc: np.ndarray,
    mf_dcn: np.ndarray,
    pc_dcn: np.ndarray,
    meta: dict | None = None,
) -> None:
    """Save final weights to an .npz archive, with an optional YAML metadata sidecar.
 
    Args:
        fname: output path, e.g. "results/weights_exp_02.npz"
        pf_pc: parallel fiber -> Purkinje cell weights, shape (n_pf,)
        mf_dcn: mossy fiber -> DCN weights, shape (n_muscle,)
        pc_dcn: Purkinje cell -> DCN weights, shape (n_muscle,)
        meta: optional dict written to a sidecar fname.yaml (e.g. training notes,
            n_epochs, final_error)
 
    Returns:
        none
    """
    fname = Path(fname)
    np.savez(fname, pf_pc=np.asarray(pf_pc), mf_dcn=np.asarray(mf_dcn), pc_dcn=np.asarray(pc_dcn))
 
    if meta:
        meta_path = fname.with_suffix(".yaml")
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
 
 
def load_weights(fname: str | Path) -> Weights:
    fname = Path(fname)
    data = np.load(fname)
 
    meta_path = fname.with_suffix(".yaml")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
 
    return Weights(
        pf_pc=data["pf_pc"],
        mf_dcn=data["mf_dcn"],
        pc_dcn=data["pc_dcn"],
        meta=meta,
        source=fname,
    )

def prepare_brain_weights(exp) -> tuple[dict[str, bool], Weights | None]:
    """Resolve an Experiment's weight-loading config into what a sim needs to
    initialize a brain: which sites are plastic (trainable), and the actual
    arrays to load (real values for frozen sites, zeros for plastic ones).
 
    Replaces the old pattern of manually building/inverting a plasticSites
    dict and multiplying arrays by `(not plastic)` to zero them out.
 
    Args:
        exp: an Experiment with load_weight_groups and load_weights_from set
 
    Returns:
        (plastic, weights)
        plastic: dict of {"pf_pc": bool, "mf_dcn": bool, "pc_dcn": bool} —
            True means this site should be trained/plastic during the sim.
        weights: a Weights with real values for frozen (non-plastic) sites
            and zero arrays (matching shape) for plastic sites, or None if
            exp.load_weights_from is None (no file to load at all — nothing
            is frozen, let the brain use its own default init).
 
    Example:
        plastic, wts = prepare_brain_weights(exp)
        brain.setActiveSites(plastic["pf_pc"], plastic["mf_dcn"], plastic["pc_dcn"])
        if wts is not None:
            brain.loadWts(wts.pf_pc, wts.mf_dcn, wts.pc_dcn)
    """
    plastic = {g: g in exp.plastic for g in ALL_WEIGHT_GROUPS}
    load =    {g: g in exp.load_weight_groups for g in ALL_WEIGHT_GROUPS}
 
    if exp.initialWts is None:
        return plastic, None
 
    # Load everything so plastic sites can be zeroed to the *correct* shape
    # rather than loaded and discarded.
    full = load_weights(exp.initialWts)
    if exp.defaultWts is None:
        zeroed = Weights(
            pf_pc=full.pf_pc if load["pf_pc"] else np.zeros_like(full.pf_pc),
            mf_dcn=full.mf_dcn if load["mf_dcn"] else np.zeros_like(full.mf_dcn),
            pc_dcn=full.pc_dcn if load["pc_dcn"] else np.zeros_like(full.pc_dcn),
            meta=full.meta,
            source=full.source,
        )
    else:
        default = load_weights(exp.defaultWts)
        zeroed = Weights(
            pf_pc=full.pf_pc if load["pf_pc"] else default.pf_pc,
            mf_dcn=full.mf_dcn if load["mf_dcn"] else default.mf_dcn, 
            pc_dcn=full.pc_dcn if load["pc_dcn"] else default.pc_dcn, 
            meta=full.meta,
            source=full.source,
        )

    return plastic, zeroed
