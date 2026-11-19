# Arm Tracking

Code from the arm-tracking project originally developed through Iyad Obeid's NSF Career grant.

This branch drives a simulated Baxter arm (right arm locked; the left arm's 7 joints move) along a
circular end-effector trajectory. A spiking cerebellar model generates the arm's joint torques.
The cerebellum is a recreation of the
model in *On Robot Compliance: A Cerebellar Control Approach* (Abadía, Naveros, Garrido, Ros,
Luque), see `reading/`, with spike-timing-dependent plasticity added at the MF–DCN and PC–DCN
synapses as in *Distributed Cerebellar Motor Learning: A Spike-Timing-Dependent Plasticity Model*
(Luque et al. 2016, `reading/luque_2016.pdf`).

## Installation Instructions

### Prerequisites:
1. Local installation of `Docker Desktop`
1. WSL (if running on Windows)
1. VSCode
1. In VSCode, install the `Dev Containers` extension
1. In VSCode, install the `WSL` extension (if on Windows)

### Actions
1. Run Docker
1. Run WSL, and clone this repository
1. In VSCode, open folder for the repository
1. A pop-up should appear asking to re-open this folder in a
Dev Container - click `Reopen in Container`

When the container is created it runs `./build.sh`, which compiles the C++ cerebellum
(`include/*.hpp` + `src/bindings.cpp`) into `garrido_brain.cpython-*.so` with CMake and pybind11.
If you change the C++ code, rerun it by hand:

```bash
./build.sh
```

## Running experiments

```bash
python runExperiment.py --list                            # see all experiments + descriptions
python runExperiment.py --experiment trial_100            # run one
python runExperiment.py -e trial_100 --dry-run            # validate paths only, don't run
python runExperiment.py --experiment trial_100 --save --hide_output
python runExperiment.py --experiment trial_100 --view
```

- `--save`: save results after running (arm trajectories to `results/arm_paths/`, graphs to `results/graphs/`, final PF-PC, MF-DCN and PC-DCN weights to the experiment's `finalWts` `.npz`, readable with `experiment_assets.load_weights`)
- `--hide_output`: don't show plots or the 3D viewer; just print when the simulation is done
- `--view`: does not re-run the experiment, only shows the saved graphs (arm, error, PF-PC weights, DCN raster, DCN weights) and then plays the saved trajectories. If no saved results exist, it runs the experiment with `--save` first
- `--no_grav`: turn gravity off in the arm model
- `--makeMovie`: accepted, but not used by `model_v02.py` on this branch yet

Each run simulates two things:
1. The arm following the trajectory with the cerebellum + a PD velocity term, repeated `nTrials` times without resetting the arm between trials.
1. A control run with no brain, for comparison.

It then plots the joint and end-effector paths, the final PF-PC weight matrix, the mean absolute
joint error per trial (red dashed line = the no-brain error), a DCN spike raster, and the MF-DCN and
PC-DCN weights vs trial for each joint's agonist and antagonist DCNs (`<graphs>_dcn_wts.png`; line =
group mean, band = min to max, dashed = initial value).

### Watching the arm

The 3D arm is drawn with Meshcat in the browser. When an experiment finishes (without
`--hide_output`), or with `--view`, the terminal asks which trajectory to play:

1. desired trajectory
1. trajectory with no brain
1. trajectory with brain (last trial)
1. trajectory the inverse dynamics were computed for

Open **http://127.0.0.1:7000/static/** (not `http://127.0.0.1:7000`); in VSCode, CTRL + click the link.

## Files

| File | Purpose |
|---|---|
| `runExperiment.py` | CLI to run one experiment by name. |
| `registry.py` | The list of named experiments. |
| `experiment_assets.py` | `Experiment` dataclass, trajectory CSV loading, weight saving/loading, dependency checks between experiments. |
| `model_v02.py` | The actual sim: inverse dynamics, the closed control loop (`runSimulation`), plotting, and playback. |
| `arm_assets_v02.py` | Arm classes, powered by the Python/C++ library Pinocchio. `baxter_reduced` is the arm used on this branch. |
| `garrido_brain_v01.py` + `lif_neuron_v01.py` | Python version of the cerebellum and its LIF neurons. |
| `include/*.hpp` + `src/bindings.cpp` | C++ version of the cerebellum and its pybind11 bindings. Builds to `garrido_brain.cpython-*.so` and is what `model_v02.py` uses by default. |
| `build.sh` / `CMakeLists.txt` | Build the C++ cerebellum. |
| `play_arm_path.py` | Plays saved trajectories in Meshcat; used by `--view`. |
| `plot_trajectory.py` | CLI to plot a trajectory file in joint + cartesian space. |
| `create_trajectories.py` | Generators for the older 2D/3-DOF arm trajectories (`--trajectory_id 3`–`9`). |
| `interpolate_traj.py` | Doubles the sample rate of `circleTrajPos.csv` by linear interpolation to make `circleTrajInterp.csv`. |
| `testBaxter.py` | Shows the Baxter model in its neutral pose in Meshcat, as a quick install check. |
| `requirements.txt` | Python libraries Docker must install. |
| `baxter_description/` | Baxter URDF and meshes. |
| `reading/` | Papers the model is based on. |
| `prev_experiments/`, `ARCHIVE/` | Older models (incl. the 2013 Garrido brain) and notes, kept for reference. |

### Switching between the C++ and Python brain

`model_v02.py` imports the compiled brain as `gb`. To use the Python version instead (slower,
but easier to debug), change the import near the top of `model_v02.py`:

```python
# import garrido_brain as gb
import garrido_brain_v01 as gb
```

Both take the same constructor arguments, `cerebellum(qMins, qdMins, qMaxs, qdMaxs, n_dof, qdCurMins, qdCurMaxs)`.
`qdMins`/`qdMaxs` set the desired-velocity MF range; the optional `qdCurMins`/`qdCurMaxs` set the
measured-velocity range and default to the desired-velocity range.
The brain controls the first 6 joints; joint `w2` is left uncontrolled.

### Synaptic plasticity

Three sites learn. Each can be turned off with `brain.setActiveSites(pf_pc, mf_dcn, pc_dcn)`, which
`model_v02.py` calls from the experiment's `plastic` field.

| Site | Rule | Weights |
|---|---|---|
| PF–PC | LTP: fixed `ALPHA` per PF spike. LTD: per CF spike, kernel over PF spikes ~100 ms earlier | `brain.pf_pc_wts` |
| MF–DCN | Luque 2016 Eqs. 5–6. LTP: fixed `MF_DCN_ALPHA` per MF spike. LTD: per PC spike reaching a DCN, symmetric kernel `exp(-\|βx\|)cos²x` over MF spikes before and after it. PCs above `MF_DCN_PC_RATE` weaken the MF drive to their DCN, PCs below strengthen it | `brain.mf_dcn_wts` (240 MFs × 600 DCN, all-to-all) |
| PC–DCN | Luque 2016 Eq. 8, symmetric i-STDP. Each PC spike depresses by `PC_DCN_LTD_MAX`, each nearby PC/DCN spike pair potentiates by `LTP_MAX·k(Δt)`, pulling each DCN toward `PC_DCN_DCN_RATE` | `brain.pc_dcn_wts` (one per PC→DCN pair) |

The MF–DCN input uses the same one-hot MF code that drives the GCs (one MF per subgroup per joint
fires every tick). The weights start where the MF–DCN and PC–DCN drives were fixed before, so with
only PF–PC plastic the model behaves as it did before. The paper gives no values for the DCN rules.
The values here are set so the rules balance near the untrained model's PC (~46 Hz) and DCN
(~153 Hz) rates. The weight caps (1 nS per MF–DCN synapse, 30 nS per PC–DCN synapse) are the per-synapse peak
conductances from the paper's Table A1. The constants are at the top of `cerebellum` in `garrido_brain_v01.py`
and `include/cerebellum.hpp`, so a change has to be made in both.

### Key simulation settings

These sit at the top of `model_v02.py` and in `runSimulation`:

- `TIMESTEP = 2e-3`: brain update period (s). The brain runs once every `TIMESTEP`, even if the trajectory is sampled faster.
- `DEL_AFF`, `DEL_EFF = 50e-3`: afferent (sensory) and efferent (motor) delays (s).
- `kp`, `kd`: gains of the PD term added to the brain's correction.

## Adding a new experiment

Add one `Experiment(...)` entry to the list in `registry.py`:

```python
Experiment(
    name="my_new_experiment",
    description="One-line summary of what this tests and what should happen.",
    trajectory=TRAJS / "circleTrajInterp.csv",  # the trajectory being followed
    actualArm=ARMS / "baxter_fixed.urdf",       # this is the arm that is actually being manipulated
    illusoryArm=ARMS / "baxter_fixed.urdf",     # this is the arm the inverse dynamics are based on
    results=RESUL / "new_exp.pkl",              # name of trajectory results file
    finalWts=WTS / "new_exp.npz",               # name of brain weight results file
    graphs=GRAPH / "new_exp.png",               # base name of resulting graphs (saved as new_exp_arm.png etc.)
    videos=VIDEO / "new_exp.mp4",               # name of resulting video
    nDof=7,
    nTrials=100,
    load_weight_groups=(),                      # which synapses should load from initialWts (OPTIONAL)
    initialWts=None,                            # where initial weights are stored (OPTIONAL)
    defaultWts=None,                            # synapses not in load_weight_groups are loaded from this file. If not given, default is 0 for all weights. (OPTIONAL)
    plastic=('pf_pc', 'mf_dcn', 'pc_dcn'),      # which synaptic sites are plastic during the experiment. Default is all (OPTIONAL)
    s1_spring_torque=30.0,                      # s1 spring torque in Nm, see baxter_reduced in arm_assets_v02.py (OPTIONAL)
),
```

`name` is what you pass to `--experiment`. Run `python runExperiment.py --list` to confirm it
shows up with the right description. If an experiment loads another experiment's weights,
`runExperiment.py` tells you which experiments to run with `--save` first.

## Trajectory files

Trajectories are CSVs in `trajectories/` with a `time` column plus any of:

| Columns | Meaning |
|---|---|
| `pos_x, pos_y, pos_z` | end-effector position |
| `vel_*`, `acc_*` | end-effector velocity / acceleration |
| `joint_pos_1 … joint_pos_7` | joint positions |
| `joint_vel_*`, `joint_acc_*` | joint velocities / accelerations |

Only `time` and position are required. Missing derivatives are computed numerically, and missing joint
positions come from inverse kinematics. Including everything (especially analytic derivatives)
gives the smoothest results, because double numerical differentiation adds noise.

The Baxter circle files:
- `circleTrajPos.csv`: the original circle with joint positions and velocities
- `circleTrajPosCartesianOnly.csv`: the same circle with end-effector positions only
- `circleTrajInterp.csv`: `circleTrajPos.csv` at double the sample rate (`python interpolate_traj.py`)

To look at a trajectory:

```bash
python plot_trajectory.py trajectories/circleTrajInterp.csv
```
