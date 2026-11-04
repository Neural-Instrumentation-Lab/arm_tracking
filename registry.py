from pathlib import Path
from experiment_assets import Experiment
from bmi import DroppedDCNBMI

# --- Central, self-documenting registry ------------------------------------
# This replaces the "type 3 for experiment 3" problem: every experiment has
# a memorable name and a description right next to its file paths.

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
            trajectory=TRAJS / "joystick.csv",
            results=RESUL / "test.pkl",
            finalWts=WTS / "test.npz",
            graphs=GRAPH / "test.png",
            videos=VIDEO / "test.mp4",
            nDof=2,
            nTrials=3,
            qMins=(-2, -1),
            qMaxs=(2, 1),
            brain_timestep=2e-3,
        ),
        Experiment(
            name="joystick",
            description="wooo yeah",
            trajectory=TRAJS / "joystick.csv",
            results=RESUL / "joystick.pkl",
            finalWts=WTS / "joystick.npz",
            graphs=GRAPH / "joystick.png",
            videos=VIDEO / "joystick.mp4",
            nDof=2,
            nTrials=200,
            qMins=(-2, -1),
            qMaxs=(2, 1),
            brain_timestep=2e-3,
        ),
        Experiment(
            name="small_timestep",
            description="small timestep traj but with normal dt",
            trajectory=TRAJS / "small_timestep.csv",
            results=RESUL / "small_timestep.pkl",
            finalWts=WTS / "small_timestep.npz",
            graphs=GRAPH / "small_timestep.png",
            videos=VIDEO / "small_timestep.mp4",
            nDof=2,
            nTrials=100,
            qMins=(-2, -1),
            qMaxs=(2, 1),
            brain_timestep=2e-3,
        ),
        Experiment(
            name="small_dt_brain",
            description="test small timestep to make sure neuron sim is working",
            trajectory=TRAJS / "small_timestep.csv",
            results=RESUL / "small_dt_brain.pkl",
            finalWts=WTS / "small_dt_brain.npz",
            graphs=GRAPH / "small_dt_brain.png",
            videos=VIDEO / "small_dt_brain.mp4",
            nDof=2,
            nTrials=200,
            qMins=(-2, -1),
            qMaxs=(2, 1),
            brain_timestep=0.5e-3,
        ),
        Experiment(
            name="complex_periodic",
            description="joystick on a figure 8 with harmonics, harder than the circle",
            trajectory=TRAJS / "complex_periodic.csv",
            results=RESUL / "complex_periodic.pkl",
            finalWts=WTS / "complex_periodic.npz",
            graphs=GRAPH / "complex_periodic.png",
            videos=VIDEO / "complex_periodic.mp4",
            nDof=2,
            nTrials=200,
            qMins=(-2, -1),
            qMaxs=(2, 1),
            brain_timestep=2e-3,
        ),
        Experiment(
            name="joystick_few_dcn",
            description="joystick circle, but the BMI only records 5 of the 50 DCNs in each agonist/antagonist group",
            trajectory=TRAJS / "joystick.csv",
            results=RESUL / "joystick_few_dcn.pkl",
            finalWts=WTS / "joystick_few_dcn.npz",
            graphs=GRAPH / "joystick_few_dcn.png",
            videos=VIDEO / "joystick_few_dcn.mp4",
            nDof=2,
            nTrials=50,
            qMins=(-2, -1),
            qMaxs=(2, 1),
            brain_timestep=2e-3,
            bmi=DroppedDCNBMI(n_drop=45, seed=0),
        ),
        Experiment(
            name="joystick_pretrained",
            description="joystick circle starting from the PF-PC weights learned by 'joystick' (run that with --save first)",
            trajectory=TRAJS / "joystick.csv",
            results=RESUL / "joystick_pretrained.pkl",
            finalWts=WTS / "joystick_pretrained.npz",
            graphs=GRAPH / "joystick_pretrained.png",
            videos=VIDEO / "joystick_pretrained.mp4",
            nDof=2,
            nTrials=10,
            qMins=(-2, -1),
            qMaxs=(2, 1),
            brain_timestep=2e-3,
            initialWts=WTS / "joystick.npz",
        ),
        Experiment(
            name="complex_pretrained",
            description="joystick on a figure 8 with harmonics, harder than the circle",
            trajectory=TRAJS / "complex_periodic.csv",
            results=RESUL / "complex_periodic_bmi.pkl",
            finalWts=WTS / "complex_periodic_bmi.npz",
            graphs=GRAPH / "complex_periodic_bmi.png",
            videos=VIDEO / "complex_periodic_bmi.mp4",
            nDof=2,
            nTrials=50,
            qMins=(-2, -1),
            qMaxs=(2, 1),
            brain_timestep=2e-3,
            bmi=DroppedDCNBMI(n_drop=45, seed=0),
            initialWts=WTS / "complex_periodic.npz",
        ),
    ]
}
