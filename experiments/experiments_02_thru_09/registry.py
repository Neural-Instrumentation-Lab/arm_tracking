from pathlib import Path
from experiment import Experiment

# --- Central, self-documenting registry ------------------------------------
# This replaces the "type 3 for experiment 3" problem: every experiment has
# a memorable name and a description right next to its file paths.

ARMS = Path("experiments/experiments_02_thru_09/models")
TRAJS = Path("experiments/experiments_02_thru_09/trajectories")
RESUL = Path("experiments/experiments_02_thru_09/results/arm_paths")

EXPERIMENTS: dict[str, Experiment] = {
    e.name: e
    for e in [
        Experiment(
            name="2dof_len_err",
            description="2dof arm with length error. Brain converges and reduces error.",
            trajectory=TRAJS / "traj_004.csv",
            actualArm=ARMS / "arm_2dof.urdf",
            illusoryArm=ARMS / "arm_2dofBigIllusion.urdf",
            results=RESUL / "path_exp_02.pkl",
            nDof=2,
        ),
        Experiment(
            name="2dof_small_len_err",
            description="2dof Arm with small length error. Brain converges but increases error.",
            trajectory=TRAJS / "traj_004.csv",
            actualArm=ARMS / "arm_2dof.urdf",
            illusoryArm=ARMS / "arm_2dofIllusoryLengths.urdf",
            results=RESUL / "path_exp_03.pkl",
            nDof=2,
        ),
        Experiment(
            name="3dof_2dtraj_len_err",
            description="3dof with 2d trajectory and length error. Brain converges but increases error.",
            trajectory=TRAJS / "traj_004.csv",
            actualArm=ARMS / "arm_3dof.urdf",
            illusoryArm=ARMS / "arm_3dofIllusion.urdf",
            results=RESUL / "path_exp_04.pkl",
            nDof=3,
        ),
        Experiment(
            name="2dof_complex_traj_len_err",
            description="2dof arm with more complicated trajectory and length error. Brain converges and reduces error.",
            trajectory=TRAJS / "traj_005.csv",
            actualArm=ARMS / "arm_2dof.urdf",
            illusoryArm=ARMS / "arm_2dofBigIllusion.urdf",
            results=RESUL / "path_exp_05.pkl",
            nDof=2,
        ),
        Experiment(
            name="3dof_3dtraj_len_err",
            description="3dof with 3d trajectory and length error. Brain flails but converges at end.",
            trajectory=TRAJS / "traj_003.csv",
            actualArm=ARMS / "arm_3dof.urdf",
            illusoryArm=ARMS / "arm_3dofIllusion.urdf",
            results=RESUL / "path_exp_06.pkl",
            nDof=3,
        ),
        Experiment(
            name="2dof_mass_small",
            description="2dof Arm with Illusory Mass .5kg. Brain converges and reduces error.",
            trajectory=TRAJS / "traj_004.csv",
            actualArm=ARMS / "arm_2dof.urdf",
            illusoryArm=ARMS / "arm_2dofIllusoryMass.urdf",
            results=RESUL / "path_exp_07.pkl",
            nDof=2,
        ),
        Experiment(
            name="2dof_mass_large",
            description="2dof Arm with Illusory Mass 5kg. Brain converges and reduces error.",
            trajectory=TRAJS / "traj_004.csv",
            actualArm=ARMS / "arm_2dof.urdf",
            illusoryArm=ARMS / "arm_2dofBigIllusoryMass.urdf",
            results=RESUL / "path_exp_08.pkl",
            nDof=2,
        ),
        Experiment(
            name="3dof_mass_small",
            description="3dof Arm with Illusory Mass .5kg. Brain converges but increases error.",
            trajectory=TRAJS / "traj_004.csv",
            actualArm=ARMS / "arm_3dof.urdf",
            illusoryArm=ARMS / "arm_3dofIllusoryMass.urdf",
            results=RESUL / "path_exp_09.pkl",
            nDof=3,
        ),
    ]
}
