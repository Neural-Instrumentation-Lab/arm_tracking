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
            trajectory=TRAJS / "circleTrajInterp8.csv",
            actualArm=ARMS / "baxter_fixed.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "test.pkl",
            finalWts=WTS / "test.npz",
            graphs=GRAPH / "test.png",
            videos=VIDEO / "test.mp4",
            nDof=7,
            nTrials=1,
            load_weight_groups = (),
        ),
        Experiment(
            name="trial_100",
            description="100 trials of the baxter on circular trajectory",
            trajectory=TRAJS / "circleTrajInterp8.csv",
            actualArm=ARMS / "baxter_fixed.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "trial_100.pkl",
            finalWts=WTS / "trial_100.npz",
            graphs=GRAPH / "trial_100.png",
            videos=VIDEO / "trial_100.mp4",
            nDof=7,
            nTrials=100,
            load_weight_groups = (),
        ),
        Experiment(
            name="trial_500",
            description="500 trials of the baxter on circular trajectory",
            trajectory=TRAJS / "circleTrajInterp8.csv",
            actualArm=ARMS / "baxter_fixed.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "trial_500.pkl",
            finalWts=WTS / "trial_500.npz",
            graphs=GRAPH / "trial_500.png",
            videos=VIDEO / "trial_500.mp4",
            nDof=7,
            nTrials=500,
            load_weight_groups = (),
        ),
        Experiment(
            name="eight_300",
            description="300 trials of the baxter on the paper's eight-like trajectory (2 s per trial)",
            trajectory=TRAJS / "eightTrajInterp8.csv",
            actualArm=ARMS / "baxter_fixed.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "eight_300.pkl",
            finalWts=WTS / "eight_300.npz",
            graphs=GRAPH / "eight_300.png",
            videos=VIDEO / "eight_300.mp4",
            nDof=7,
            nTrials=300,
            load_weight_groups = (),
        ),
        Experiment(
            name="target_reaching_100",
            description="100 trials of the paper's 8 target reaches in a fixed order (16 s per trial, 2 s per reach)",
            trajectory=TRAJS / "targetReachingInterp8.csv",
            actualArm=ARMS / "baxter_fixed.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "target_reaching_100.pkl",
            finalWts=WTS / "target_reaching_100.npz",
            graphs=GRAPH / "target_reaching_100.png",
            videos=VIDEO / "target_reaching_100.mp4",
            nDof=7,
            nTrials=100,
            load_weight_groups = (),
        ),

    ]
}
