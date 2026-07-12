from pathlib import Path
from experiment_assets import Experiment

# --- Central, self-documenting registry ------------------------------------
# This replaces the "type 3 for experiment 3" problem: every experiment has
# a memorable name and a description right next to its file paths.

ARMS  = Path("models")
TRAJS = Path("trajectories")
RESUL = Path("results/arm_paths")
WTS   = Path("results/brain_weights")

EXPERIMENTS: dict[str, Experiment] = {
    e.name: e
    for e in [
        Experiment(
            name="full_10",
            description="10kg mass manipulated over 1500 trials with no preset weights",
            trajectory=TRAJS / "traj_009.csv",
            actualArm=ARMS / "arm_3dofLargeMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "cntrl_exp.pkl",
            initialWts=None,
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
            initialWts=None,
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
            initialWts=None,
            nDof=3,
            nTrials=1,
        ),
        Experiment(
            name="full_1_5",
            description="1.5kg mass manipulated over the full 1500 trials with no init weights",
            trajectory=TRAJS / "traj_009.csv",
            actualArm=ARMS / "arm_3dofMedMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "test.pkl",
            initialWts=None,
            nDof=3,
            nTrials=1500,
        ),
        Experiment(
            name="full_0_5",
            description="0.5kg mass manipulated over the full 1500 trials with no init weights",
            trajectory=TRAJS / "traj_009.csv",
            actualArm=ARMS / "arm_3dofSmallMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "test.pkl",
            initialWts=None,
            nDof=3,
            nTrials=1500,
        ),
        Experiment(
            name="pf_pc_1_5",
            description="1.5kg mass manipulated over 150 trials with init weights for 1.5kg",
            trajectory=TRAJS / "traj_009.csv",
            actualArm=ARMS / "arm_3dofMedMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "test.pkl",
            initialWts=WTS / "1_5kg.csv",
            nDof=3,
            nTrials=150,
        ),
        Experiment(
            name="pf_pc_10",
            description="1.5kg mass manipulated over 150 trials with init weights for 10kg",
            trajectory=TRAJS / "traj_009.csv",
            actualArm=ARMS / "arm_3dofMedMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "test.pkl",
            initialWts=WTS / "10kg.csv",
            nDof=3,
            nTrials=150,
        ),
        Experiment(
            name="pf_pc_0_5",
            description="1.5kg mass manipulated over 150 trials with init weights for 0.5kg",
            trajectory=TRAJS / "traj_009.csv",
            actualArm=ARMS / "arm_3dofMedMass.urdf",
            illusoryArm=ARMS / "arm_3dof.urdf",
            results=RESUL / "test.pkl",
            initialWts=WTS / "0_5kg.csv",
            nDof=3,
            nTrials=150,
        ),

    ]
}
