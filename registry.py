from pathlib import Path
from experiment_assets import Experiment

# --- Central, self-documenting registry ------------------------------------
# This replaces the "type 3 for experiment 3" problem: every experiment has
# a memorable name and a description right next to its file paths.

ARMS = Path("models")
TRAJS = Path("trajectories")
RESUL = Path("results/arm_paths")

EXPERIMENTS: dict[str, Experiment] = {
    e.name: e
    for e in [
        Experiment(
            name="cntrl_exp",
            description="10kg mass manipulated over 1500 trials with no preset weights",
            trajectory=TRAJS / "traj_009.csv",
            actualArm=ARMS / "arm_3dofLargeMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "cntrl_exp.pkl",
            nDof=3,
            nTrials=1500,
        ),
        Experiment(
            name="diff_traj",
            description="10kg mass manipulated over 1500 trials with no preset weights. Uses the same 8-figure shape trajectory but translated",
            trajectory=TRAJS / "traj_008.csv",
            actualArm=ARMS / "arm_3dofLargeMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "diff_traj.pkl",
            nDof=3,
            nTrials=1500,
        ),
        Experiment(
            name="test",
            description="test, only simulates one trial",
            trajectory=TRAJS / "traj_009.csv",
            actualArm=ARMS / "arm_3dofLargeMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "test.pkl",
            nDof=3,
            nTrials=1,
        ),
    ]
}
