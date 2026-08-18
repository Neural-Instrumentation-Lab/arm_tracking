from pathlib import Path
from experiment_assets import Experiment

# --- Central, self-documenting registry ------------------------------------
# This replaces the "type 3 for experiment 3" problem: every experiment has
# a memorable name and a description right next to its file paths.

ARMS  = Path("baxter_description/urdf")
TRAJS = Path("trajectories")
RESUL = Path("results/arm_paths")
WTS   = Path("results/brain_weights")
GRAPH = Path("results/graphs")
VIDEO = Path("results/videos")

EXPERIMENTS: dict[str, Experiment] = {
    e.name: e
    for e in [
        Experiment(
            name="test",
            description="trying out baxter",
            trajectory=TRAJS / "circleTrajInterp.csv",
            actualArm=ARMS / "baxter_fixed.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "test.pkl",
            finalWts=WTS / "test.npz",
            graphs=GRAPH / "test.png",
            videos=VIDEO / "test.mp4",
            nDof=7,
            nTrials=100,
            load_weight_groups = (),
        ),
        Experiment(
            name="test2",
            description="simple 2dof arm for brain test",
            trajectory=TRAJS / "traj_004.csv",
            actualArm=ARMS / "arm_2dof.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "test2.pkl",
            finalWts=WTS / "test2.npz",
            graphs=GRAPH / "test2.png",
            videos=VIDEO / "test2.mp4",
            nDof=2,
            nTrials=1,
            load_weight_groups = (),
        ),

    ]
}
