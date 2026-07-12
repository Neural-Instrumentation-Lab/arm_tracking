from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import pandas as pd
import numpy as np
import yaml

@dataclass(frozen=True)
class Experiment:
    name: str                # short, unique, CLI-friendly id (e.g. "baseline_v1")
    description: str         # one-line human explanation, shown in --list
    trajectory: Path         # the 1 trajectory file
    actualArm: Path          # the 2 arm files
    illusoryArm: Path
    results: Path # where .pkl of results is stored
    nDof : int # how many degrees of freedom arm has

    def validate(self) -> list[str]:
        """Return a list of problems (missing files etc). Empty list = OK."""
        problems = []
        for label, path in [
            ("trajectory", self.trajectory),
            ("actualArm", self.actualArm),
            ("illusoryArm", self.illusoryArm),
        ]:
            if not path.exists():
                problems.append(f"{label} file not found: {path}")
        return problems

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
