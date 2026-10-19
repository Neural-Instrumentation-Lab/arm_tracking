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
            nTrials=1,
            load_weight_groups = (),
        ),
        Experiment(
            name="joystick",
            description="wooo yeah",
            trajectory=TRAJS / "joystick.csv",
            actualArm=ARMS / "baxter_fixed.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "joystick.pkl",
            finalWts=WTS / "joystick.npz",
            graphs=GRAPH / "joystick.png",
            videos=VIDEO / "joystick.mp4",
            nDof=2,
            nTrials=50,
            load_weight_groups = (),
        ),
        Experiment(
            name="small_timestep",
            description="test small timestep to make sure neuron sim is working",
            trajectory=TRAJS / "small_timestep.csv",
            actualArm=ARMS / "baxter_fixed.urdf",
            illusoryArm=ARMS / "baxter_fixed.urdf",
            results=RESUL / "small_timestep.pkl",
            finalWts=WTS / "small_timestep.npz",
            graphs=GRAPH / "small_timestep.png",
            videos=VIDEO / "small_timestep.mp4",
            nDof=2,
            nTrials=150,
            load_weight_groups = (),
        )
    ]
}
