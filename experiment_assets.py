from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import pandas as pd
import numpy as np
import yaml
from bmi import BMI, FullBMI


class brainData:
    def __init__(self, nTrials, trajLen, nDCNs, nGCs, nPCs):
        self.dcn     = np.zeros((nTrials, trajLen, nDCNs), dtype=bool)   # DCN spikes are true/false
        self.pf_pc   = np.zeros((nTrials, nGCs, nPCs)) 

@dataclass(frozen=True)
class Experiment:
    name: str                # short, unique, CLI-friendly id (e.g. "baseline_v1")
    description: str         # one-line human explanation, shown in --list
    trajectory: Path         # the 1 trajectory file
    results: Path            # where .pkl of arm trajectories is stored
    finalWts : Path          # where .npz of brain weights is stored
    graphs : Path            # where .pngs of results are stored 
    videos : Path            # where .mp4 of video results are stored 
    nDof : int # how many degrees of freedom arm has
    nTrials : int # how many times the traj is repeated
    qMins : tuple[float, ...] # per-joint lower bound of the brain's MF encoding range, length nDof
    qMaxs : tuple[float, ...] # per-joint upper bound of the brain's MF encoding range, length nDof
    brain_timestep : float = 2e-3 # (s) how often the brain is stepped during the sim
    bmi : BMI = field(default_factory=FullBMI) # decodes the brain's DCN spikes into the arm command, see bmi.py
    initialWts: Path | None = None # brain weights .npz (another experiment's finalWts) whose final PF-PC weights the brain starts from. None = default init

    def validate(self, assume_produced: frozenset[Path] = frozenset()) -> list[str]:
        """Return a list of problems (missing files etc). Empty list = OK.
 
        Args:
            assume_produced: paths to treat as OK even though they don't exist
                yet — e.g. when batch-running a set of experiments where one
                produces a results file another consumes as load_weights_from.
                Standalone/single-experiment runs should leave this empty.
        """
        problems = []
        if not self.trajectory.exists():
            problems.append(f"trajectory file not found: {self.trajectory}")

        if len(self.qMins) != self.nDof or len(self.qMaxs) != self.nDof:
            problems.append(
                f"qMins/qMaxs must have nDof={self.nDof} entries "
                f"(got {len(self.qMins)} and {len(self.qMaxs)})"
            )
        elif any(lo >= hi for lo, hi in zip(self.qMins, self.qMaxs)):
            problems.append(f"every qMins entry must be < its qMaxs entry: {self.qMins} vs {self.qMaxs}")
        if self.brain_timestep <= 0:
            problems.append(f"brain_timestep must be positive (got {self.brain_timestep})")
 
        if self.initialWts is not None:
            already_exists = self.initialWts.exists()
            will_be_produced = self.initialWts in assume_produced
            if not already_exists and not will_be_produced:
                problems.append(f"initialWts file not found: {self.initialWts}")
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
Saving/loading brain results.

Stored as a compressed numpy .npz archive:
    pf_pc       - parallel fiber -> Purkinje cell weights after every trial, shape (nTrials, n_pf, n_pc)
    dcn_packed  - DCN spikes of every step, bit-packed (read them with load_dcn)
plus an optional YAML sidecar for anything that isn't inferable from the
arrays themselves (e.g. training notes).

An experiment can start from another one's weights by setting its initialWts
to that experiment's finalWts; load_pf_pc() gives the weights after the last trial.
"""
def save_weights(
    fname: str | Path,
    results : brainData,
    meta: dict | None = None,
) -> None:
    """Save brain results to an .npz archive, with an optional YAML metadata sidecar.

    Args:
        fname: output path, e.g. "results/weights_exp_02.npz"
        results: brainData from runSimulation (PF-PC weights per trial, DCN spikes per step)
        meta: optional dict written to a sidecar fname.yaml (e.g. training notes,
            n_epochs, final_error)
 
    Returns:
        none
    """
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    # DCN spikes are bit-packed along the neuron axis (8 spikes per byte); load_dcn() unpacks them.
    # savez_compressed is lossless and shrinks pf_pc a lot, since rows of unvisited GCs never change.
    dcn = np.asarray(results.dcn, dtype=bool)
    np.savez_compressed(fname, pf_pc=np.asarray(results.pf_pc),
                        dcn_packed=np.packbits(dcn, axis=-1), dcn_n=dcn.shape[-1])
 
    if meta:
        meta_path = fname.with_suffix(".yaml")
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
 
 
def load_dcn(fname: str | Path) -> np.ndarray:
    """DCN spikes saved by save_weights, as a bool array of shape (nTrials, trajLen, nDCNs)."""
    data = np.load(fname)
    if "dcn" in data:                           # files saved before bit-packing
        return data["dcn"].astype(bool)
    return np.unpackbits(data["dcn_packed"], axis=-1, count=int(data["dcn_n"])).astype(bool)


def load_pf_pc(fname: str | Path) -> np.ndarray:
    """PF-PC weights after the last trial saved by save_weights, shape (n_pf, n_pc)."""
    pf_pc = np.load(fname)["pf_pc"]
    return pf_pc[-1] if pf_pc.ndim == 3 else pf_pc
