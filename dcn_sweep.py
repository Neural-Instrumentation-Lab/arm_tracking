"""
DCN readout sweep (BMI experiment)

Trains a fresh brain for each number of recorded DCNs and plots the final MAE
against it. Only the readout changes: the brain always simulates all of its DCNs,
but the BMI decodes the joystick command from n of them per agonist/antagonist group
(a DroppedDCNBMI dropping the rest, see bmi.py).

Usage:
    python dcn_sweep.py                                   # defaults: joystick experiment, 1..50 DCNs
    python dcn_sweep.py --counts 50 20 5 1 --trials 100 --repeats 3
    python dcn_sweep.py --plot_only                       # re-plot the saved csv

Results go to results/dcn_sweep/ (csv of final MAE, npz of every learning curve, plots).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from simulate import runSimulation
from bmi import DroppedDCNBMI
from registry import EXPERIMENTS
from experiment_assets import load_trajectory
from arm_assets_v02 import armTraj, JoystickDotSimulator
from model_v02 import cerebellum, fillTrajectory, build_brain

OUT = Path("results/dcn_sweep")


def run_one(job):
    """One full training run with n recorded DCNs per group. Returns the per-trial MAE."""
    exp_name, n_dcn, seed, n_trials, reset = job
    np.random.seed(seed)                      # random trial start points
    exp = EXPERIMENTS[exp_name]
    joystick = JoystickDotSimulator()
    traj = fillTrajectory(load_trajectory(exp.trajectory), joystick)
    desired = armTraj(pos=traj.position, vel=traj.velocity)

    brain = build_brain(exp, seed)      # same brain (and initialWts) as runExperiment
    bmi = DroppedDCNBMI(n_drop=brain.nDCN // 2 - n_dcn, seed=seed)
    _, errorTot, _, _ = runSimulation(joystick, desired, traj.time, brain, bmi, n_trials, brain_dt=exp.brain_timestep,
                                      reset=reset, record_brain=False, verbose=False)
    mae = errorTot / (len(traj.time) * (brain.nJoints - 1))   # same normalization as plot_brain_results
    print(f"  n_dcn={n_dcn:3d} seed={seed}: final MAE {mae[-10:].mean():.3f}", flush=True)
    return n_dcn, seed, mae


def plot(counts, final, curves, n_per_group, show):
    counts = np.asarray(counts)
    mean, std = final.mean(axis=1), final.std(axis=1)

    fig, ax = plt.subplots()
    ax.errorbar(counts, mean, yerr=std if final.shape[1] > 1 else None, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel(f"DCNs recorded per agonist/antagonist group (of {n_per_group})")
    ax.set_ylabel("Final MAE")
    ax.set_title("BMI readout: final MAE vs recorded DCNs")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(OUT / "mae_vs_dcn.png", dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots()
    for c, curve in zip(counts, curves):
        ax2.plot(curve.mean(axis=0), label=f"{c} DCNs")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("MAE")
    ax2.set_title("Learning curves by number of recorded DCNs")
    ax2.legend()
    fig2.savefig(OUT / "learning_curves.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", "-e", default="joystick", choices=sorted(EXPERIMENTS),
                        help="experiment whose trajectory (and nTrials, unless --trials) is used")
    parser.add_argument("--counts", type=int, nargs="+", default=[50, 30, 20, 10, 5, 2, 1],
                        help="numbers of recorded DCNs per agonist/antagonist group")
    parser.add_argument("--trials", type=int, default=None, help="trials per run (default: experiment's nTrials)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per count with different seeds (error bars)")
    parser.add_argument("--last", type=int, default=10, help="final MAE = mean MAE of the last N trials")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel runs")
    parser.add_argument("--noReset", action="store_true",
                        help="don't reset the arm to a random start point each trial (same as runExperiment.py)")
    parser.add_argument("--plot_only", action="store_true", help="re-plot saved results without running")
    parser.add_argument("--hide_output", action="store_true", help="save plots without showing them")
    args = parser.parse_args()

    if args.hide_output:
        plt.switch_backend("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    n_per_group = cerebellum.nDCN // 2

    if args.plot_only:
        saved = np.load(OUT / "dcn_sweep.npz")
        curves = saved["curves"]
        final = curves[:, :, -int(saved["last"]):].mean(axis=2)
        plot(saved["counts"], final, curves, n_per_group, not args.hide_output)
        return

    bad = [c for c in args.counts if not 1 <= c <= n_per_group]
    if bad:
        parser.error(f"--counts must be between 1 and {n_per_group}; got {bad}")
    counts = sorted(set(args.counts), reverse=True)
    n_trials = args.trials or EXPERIMENTS[args.experiment].nTrials
    jobs = [(args.experiment, c, seed, n_trials, not args.noReset) for c in counts for seed in range(args.repeats)]
    print(f"Running {len(jobs)} runs of {n_trials} trials on '{args.experiment}' ({args.workers} workers)...")

    curves = np.zeros((len(counts), args.repeats, n_trials))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n_dcn, seed, mae in pool.map(run_one, jobs):
            curves[counts.index(n_dcn), seed] = mae
    final = curves[:, :, -args.last:].mean(axis=2)

    np.savez(OUT / "dcn_sweep.npz", counts=counts, curves=curves, last=args.last)
    with open(OUT / "dcn_sweep.csv", "w") as f:
        f.write("n_dcn_per_group,seed,final_mae\n")
        for i, c in enumerate(counts):
            for s in range(args.repeats):
                f.write(f"{c},{s},{final[i, s]:.5f}\n")
    print(f"Saved results to {OUT}/")
    plot(counts, final, curves, n_per_group, not args.hide_output)


if __name__ == "__main__":
    main()
