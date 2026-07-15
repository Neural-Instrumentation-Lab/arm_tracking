# Arm Tracking

Code from the arm-tracking project originally developed through Iyad Obeid's NSF Career grant

## Installation Instructions

### Prerequisites: 
1. Local installation of `Docker Desktop`
1. WSL (if running on Windows)
1. VSCode
1. In VSCode, install the `Dev Containers` extension
1. In VSCode, install the `WSL` extension (if on Windows)

### Actions
1. Run Docker
1. Run WSL, and clone this repositiory
1. In VSCode, open folder for the repository
1. A pop-up should appear asking to re-open this folder in a
Dev Container - click `Reopen in Container`

## Running experiments

```bash
python runExperiment.py --list                            # see all experiments + descriptions
python runExperiment.py --experiment full_10              # run one
python runExperiment.py -e full_10 --dry-run    # validate paths only, don't run
python runExperiment.py --experiment full_10 --save --hide_output
python runExperiment.py --experiment full_10 --view
```

- `--save` — save results after running (weights, graphs, trajectories, etc.)
- `--hide_output` — suppress the sim's normal console/plot output
- `--view` — does not re-run experiment, only brings up previous data to be viewed 

Some experiments require other experiment's results for their own initial conditions. An experiment will error and tell you which experiments to run to construct their dependencies.
Alternatively, you can run all the experiments to have all their results already. Warning: running all the experiments takes ~15 mins. 

```bash
python run_all.py
```

## Files

| File | Purpose |
|---|---|
| `experiment_assets.py` | useful dataclasses used by most files. |
| `registry.py` | The list of named experiments. |
| `model_v01.py` | actual sim logic. |
| `runExperiment.py` | CLI to run one experiment by name. |
| `runAll.py` | CLI to run every (or a chosen subset of) experiment, in dependency order. |
| `plot_trajectory.py` | CLI to plot a trajectory in joint + cartesian space. |
| `play_arm_path.py` | used when --view is passed to runExperiment.py |
| `cerebellum.hpp` + `garrido_brain_v00.py` | brain classes. Have identical results but C++ is faster and default. Can swap them out by uncommenting 2 lines in model_v01.py |
| `bindings.cpp` | binds Python & C++ versions of brain with pybind11 |
| `arm_assets_v01.py` | contains the arm class. Powered by Python/C++ Library Pinocchio. |
| `makefile` | rebuild cerebellum if code is changed with `make brain`. Should auto make on container build. |
| `requirments.txt` | python libraries docker must install |
 

## Adding a new experiment

Add one `Experiment(...)` entry to the list in `registry.py`:

```python
Experiment(
    name="my_new_experiment",
    description="One-line summary of what this tests and what should happen.",
    trajectory=TRAJS / "traj_009.csv",         # the trajectory being followed
    actualArm=ARMS / "arm_3dof.urdf",          # this is the arm that is actually being manipulated
    illusoryArm=ARMS / "arm_3dofMedMass.urdf", # this is the arm the inverse dynamics are based on
    results=RESUL / "new_exp.pkl",             # name of trajectory results file
    finalWts=WTS / "new_exp.npz",              # name of brain weight results file
    graphs=GRAPH / "new_exp.png",              # name of resulting graphs (will be split into new_exp_brain.png and new_exp_arm.png)
    nDof=3,
    nTrials=1500,
    load_weight_groups=(),                     # which synapses should load from initWts (OPTIONAL)
    initWts=None,                              # where initial weights are stored (OPTIONAL)
    defaultWts=None,                           # synapses not in load_weight_groups are loaded from this file. If not given, default is 0 for all weights. (OPTIONAL)
    plastic=('pf_pc', 'mf_dcn', 'pc_dcn'),     # which synaptic sites are plastic during the experiment. Default is all (OPTIONAL) 
),
```

`name` is what you pass to `--experiment`. Run `python runExperiment.py
--list` to confirm it shows up with the right description.

## Trajectory files
can be recreated by running 
```bash
python create_trajectories --trajectory_id=<Trajectory_Number_To_Create>
```

if creating new trajectories, including all possible info (cartesian pos, joint pos, and analytical derivatives) produces the best results.
However, only `time` and `pos` are required, the rest if omitted will be created by the model. 