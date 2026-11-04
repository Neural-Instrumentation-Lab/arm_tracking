"""
BMI sweep

Trains a brain for each value of one BMI's setting (see BMI_SWEEPS in bmi.py) and plots the
final MAE against it. Each brain starts from the experiment's initialWts if it has any
(otherwise from the default weights). Only the readout changes: the brain always simulates
all of its DCNs, and the BMI decides what it gets to see of them.

Usage:
    python dcn_sweep.py                                   # joystick experiment, num_drop BMI
    python dcn_sweep.py --bmi spike_drop --values 0 0.5 0.9 --trials 100 --repeats 3
    python dcn_sweep.py -e joystick_pretrained            # start from that experiment's initialWts
    python dcn_sweep.py -e joystick --bmi num_drop --plot_only   # re-plot that sweep's saved results

Results go to results/dcn_sweep/<experiment>_<bmi>/ (csv of final MAE, npz of every learning curve, plots).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from simulate import runSimulation
from plot_results import apply_mae_ylim
from bmi import BMI_SWEEPS, BMIError
from registry import EXPERIMENTS
from experiment_assets import load_trajectory
from arm_assets_v02 import armTraj, JoystickDotSimulator
from model_v02 import fillTrajectory, build_brain

OUT = Path("results/dcn_sweep")


def run_one(job):
    """One full training run with one BMI setting. Returns the per-trial MAE."""
    exp_name, bmi_name, value, seed, n_trials, reset = job
    np.random.seed(seed)                      # random trial start points
    exp = EXPERIMENTS[exp_name]
    joystick = JoystickDotSimulator()
    traj = fillTrajectory(load_trajectory(exp.trajectory), joystick)
    desired = armTraj(pos=traj.position, vel=traj.velocity)

    brain = build_brain(exp, seed)      # same brain (and initialWts) as runExperiment
    bmi = BMI_SWEEPS[bmi_name].make(value, seed)
    _, errorTot, _, _, _ = runSimulation(joystick, desired, traj.time, brain, bmi, n_trials, brain_dt=exp.brain_timestep,
                                      reset=reset, record_brain=False, verbose=False)
    mae = errorTot / (len(traj.time) * (brain.nJoints - 1))   # same normalization as plot_brain_results
    print(f"  {bmi} seed={seed}: final MAE {mae[-10:].mean():.3f}", flush=True)
    return value, seed, mae


def plot(values, final, curves, sweep, title, out, show):
    labels = [f"{v:g}" for v in values]
    mean, std = final.mean(axis=1), final.std(axis=1)

    # evenly spaced, not to scale: the informative values bunch up near the extremes
    fig, ax = plt.subplots()
    ax.errorbar(range(len(values)), mean, yerr=std if final.shape[1] > 1 else None, marker="o", capsize=3)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(sweep.label)
    ax.set_ylabel("Final MAE")
    ax.set_title(f"{title}: final MAE")
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "final_mae.png", dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots()
    for label, curve in zip(labels, curves):
        ax2.plot(curve.mean(axis=0), label=label)
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("MAE")
    apply_mae_ylim(ax2, curves.mean(axis=1))
    ax2.set_title(f"{title}: learning curves")
    ax2.legend(title=sweep.label, fontsize=8, title_fontsize=8)
    fig2.savefig(out / "learning_curves.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", "-e", default="joystick", choices=sorted(EXPERIMENTS),
                        help="experiment whose trajectory, initialWts (and nTrials, unless --trials) are used")
    parser.add_argument("--bmi", default="num_drop", choices=sorted(BMI_SWEEPS), help="which BMI to sweep")
    parser.add_argument("--values", type=float, nargs="+", default=None,
                        help="BMI settings to test (default: that BMI's values in BMI_SWEEPS)")
    parser.add_argument("--trials", type=int, default=None, help="trials per run (default: experiment's nTrials)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per value with different seeds (error bars)")
    parser.add_argument("--last", type=int, default=10, help="final MAE = mean MAE of the last N trials")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel runs")
    parser.add_argument("--noReset", action="store_true",
                        help="don't reset the arm to a random start point each trial (same as runExperiment.py)")
    parser.add_argument("--plot_only", action="store_true", help="re-plot saved results without running")
    parser.add_argument("--hide_output", action="store_true", help="save plots without showing them")
    args = parser.parse_args()

    if args.hide_output:
        plt.switch_backend("Agg")
    sweep = BMI_SWEEPS[args.bmi]
    out = OUT / f"{args.experiment}_{args.bmi}"
    title = f"{args.experiment}, {args.bmi}"

    if args.plot_only:
        if not (out / "results.npz").exists():
            parser.error(f"no saved sweep at {out}/")
        saved = np.load(out / "results.npz")
        curves = saved["curves"]
        final = curves[:, :, -int(saved["last"]):].mean(axis=2)
        plot(saved["values"], final, curves, sweep, title, out, not args.hide_output)
        return

    exp = EXPERIMENTS[args.experiment]
    problems = exp.validate()
    if problems:
        parser.error(f"cannot run '{exp.name}': " + "; ".join(problems))
    values = sorted(set(sweep.values if args.values is None else args.values))
    brain = build_brain(exp)
    try:
        for v in values:
            sweep.make(v, 0).setup(brain)
    except BMIError as e:
        parser.error(f"--values: {e}")
    n_trials = args.trials or exp.nTrials
    jobs = [(args.experiment, args.bmi, v, seed, n_trials, not args.noReset) for v in values for seed in range(args.repeats)]
    print(f"Running {len(jobs)} runs of {n_trials} trials on '{args.experiment}' with BMI '{args.bmi}' ({args.workers} workers)...")
    print(f"Starting weights: {exp.initialWts or 'default'}")

    curves = np.zeros((len(values), args.repeats, n_trials))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for value, seed, mae in pool.map(run_one, jobs):
            curves[values.index(value), seed] = mae
    final = curves[:, :, -args.last:].mean(axis=2)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "results.npz", values=values, curves=curves, last=args.last)
    with open(out / "results.csv", "w") as f:
        f.write("value,seed,final_mae\n")
        for i, v in enumerate(values):
            for s in range(args.repeats):
                f.write(f"{v:g},{s},{final[i, s]:.5f}\n")
    print(f"Saved results to {out}/")
    plot(values, final, curves, sweep, title, out, not args.hide_output)


if __name__ == "__main__":
    main()
