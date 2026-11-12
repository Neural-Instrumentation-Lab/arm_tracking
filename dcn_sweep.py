"""
BMI sweep

Trains a brain for every combination of the swept BMI settings (see BMI_SWEEPS in bmi.py).
Sweeping one setting plots the final MAE against it; sweeping two or more only saves the
results (csv/npz), to be analysed before deciding how to plot them. Each brain starts from the experiment's initialWts if it has any
(otherwise from the default weights). Only the readout changes: the brain always simulates
all of its DCNs, and the BMI decides what it gets to see of them.

Usage:
    python dcn_sweep.py                                   # joystick experiment, num_drop BMI
    python dcn_sweep.py --bmi spike_drop --values 0 0.5 0.9 --trials 100 --repeats 3
    python dcn_sweep.py -e joystick_pretrained            # start from that experiment's initialWts
    python dcn_sweep.py -e joystick --bmi num_drop --plot_only   # re-plot that sweep's saved results

    # BMIs with several settings: the first one is swept by default, the rest stay at their defaults
    python dcn_sweep.py --bmi cross_cluster                         # right_prob, cluster_size=10
    python dcn_sweep.py --bmi cross_cluster --sweep cluster_size --set right_prob=0.5
    python dcn_sweep.py --bmi cross_cluster --sweep right_prob cluster_size     # full grid, csv only
    python dcn_sweep.py --bmi cross_cluster --set right_prob=0.25,1 --set cluster_size=5,10

    python dcn_sweep.py --list                      # every BMI and its settings
    python dcn_sweep.py --bmi cluster --list        # one BMI's settings, defaults and swept values

--set NAME=V1,V2,... gives the values of one setting; giving more than one value sweeps it.
--sweep NAME ... sweeps those settings over their default values from BMI_SWEEPS.
--values V1 V2 ... is shorthand for --set on the one swept setting.

A sweep can also be saved in registry.py as a SweepExperiment and run with
`python runExperiment.py -e <name>`, which calls this script with its base experiment and args.

Results go to results/dcn_sweep/<experiment>_<bmi>[_<swept settings>]/ (or <--name>/) (csv of final MAE with every
setting, npz of every learning curve, plots when one setting is swept).
"""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from simulate import runSimulation
from plot_results import apply_mae_ylim
from bmi import BMI_SWEEPS, BMIError, ProbabilityError
from registry import EXPERIMENTS
from experiment_assets import load_trajectory, Experiment
from arm_assets_v02 import armTraj, JoystickDotSimulator
from model_v02 import fillTrajectory, build_brain

OUT = Path("results/dcn_sweep")
# experiments a sweep can be based on (the registry also holds SweepExperiments, which run through this script)
BASES = {name: exp for name, exp in EXPERIMENTS.items() if isinstance(exp, Experiment)}


def run_one(job):
    """One full training run with one combination of BMI settings. Returns the per-trial MAE."""
    exp_name, bmi_name, index, settings, seed, n_trials, reset = job
    np.random.seed(seed)                      # random trial start points
    exp = EXPERIMENTS[exp_name]
    joystick = JoystickDotSimulator()
    traj = fillTrajectory(load_trajectory(exp.trajectory), joystick)
    desired = armTraj(pos=traj.position, vel=traj.velocity)

    brain = build_brain(exp, seed)      # same brain (and initialWts) as runExperiment
    bmi = BMI_SWEEPS[bmi_name].build(seed, **settings)
    _, errorTot, _, _, _ = runSimulation(joystick, desired, traj.time, brain, bmi, n_trials, brain_dt=exp.brain_timestep,
                                      reset=reset, record_brain=False, verbose=False)
    mae = errorTot / (len(traj.time) * (brain.nJoints - 1))   # same normalization as plot_brain_results
    print(f"  {bmi} seed={seed}: final MAE {mae[-10:].mean():.3f}", flush=True)
    return index, seed, mae


def plot(values, final, curves, label, title, out, show):
    labels = [f"{v:g}" for v in values]
    mean, std = final.mean(axis=1), final.std(axis=1)

    # evenly spaced, not to scale: the informative values bunch up near the extremes
    fig, ax = plt.subplots()
    ax.errorbar(range(len(values)), mean, yerr=std if final.shape[1] > 1 else None, marker="o", capsize=3)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(label)
    ax.set_ylabel("Final MAE")
    ax.set_title(f"{title}: final MAE")
    ax.grid(True, alpha=0.3)
    fig.savefig(out / "final_mae.png", dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots()
    for lab, curve in zip(labels, curves):
        ax2.plot(curve.mean(axis=0), label=lab)
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("MAE")
    apply_mae_ylim(ax2, curves.mean(axis=1))
    ax2.set_title(f"{title}: learning curves")
    ax2.legend(title=label, fontsize=8, title_fontsize=8)
    fig2.savefig(out / "learning_curves.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close("all")


def parse_set(text):
    """'name=v1,v2,...' -> (name, [v1, v2, ...])"""
    name, sep, values = text.partition("=")
    try:
        if not sep or not name:
            raise ValueError
        return name.strip(), [float(v) for v in values.split(",") if v.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected NAME=V1,V2,... (got '{text}')")


def setting_values(sweep, args, parser):
    """Values of every setting of the BMI: {name: sorted list}. Settings with >1 value are swept."""
    names = list(sweep.params)
    for name in (args.sweep or []) + [n for n, _ in args.set]:
        if name not in sweep.params:
            parser.error(f"BMI '{args.bmi}' has no setting '{name}' (it has: {', '.join(names)})")
    given = dict(args.set)
    if args.sweep is not None:
        swept = list(args.sweep)
    elif given:
        swept = []                            # only what --set gives several values to
    else:
        swept = [names[0]]
    values = {}
    for name, param in sweep.params.items():
        if name in given:
            values[name] = given[name]
        elif name in swept:
            values[name] = list(param.values)
        else:
            values[name] = [param.default]
    chosen = set(given) | set(swept)
    if args.values is not None:
        multi = [n for n in names if len(values[n]) > 1] or swept or [names[0]]
        if len(multi) != 1:
            parser.error(f"--values needs exactly one swept setting (sweeping {', '.join(multi)}); use --set NAME=...")
        values[multi[0]] = list(args.values)
        chosen.add(multi[0])
    for group in sweep.linked:               # choosing any of a linked group drops the others' defaults
        if chosen & set(group):
            for name in set(group) - chosen:
                values[name] = [None]
    return {n: sorted(set(v)) for n, v in values.items()}


def fmt(value):
    """A setting's value for printing; None is one the BMI works out from the others."""
    return "auto" if value is None else f"{value:g}"


def print_settings(bmi_names):
    """What each BMI is and the settings dcn_sweep.py can give it."""
    for name in bmi_names:
        sweep = BMI_SWEEPS[name]
        doc = (type(sweep.build(0)).__doc__ or "").strip().split("\n\n")[0]
        print(f"{name}:")
        print("\n".join("    " + line.strip() for line in doc.splitlines()))
        width = max(len(n) for n in sweep.params)
        for i, (pname, param) in enumerate(sweep.params.items()):
            kind = "whole number" if param.whole else "number"
            first = " (swept when no setting is chosen)" if i == 0 and len(sweep.params) > 1 else ""
            print(f"  {pname:<{width}}  {param.label} [{kind}]{first}")
            default = "worked out from the others" if param.default is None else f"{param.default:g}"
            print(f"  {'':<{width}}  default {default}; swept over {', '.join(f'{v:g}' for v in param.values)}")
        for group in sweep.linked:
            print(f"  {', '.join(group)} must add up to 1: set or sweep any {len(group) - 1} of them (or all {len(group)})")
            print(f"  and the last is 1 minus the others. Setting any of them drops the others' defaults; in a sweep,")
            print(f"  combinations that don't add up to 1 are skipped.")
        print()


def main(argv=None):
    """argv: the command-line arguments as a list (default: sys.argv). runExperiment.py runs registry sweeps this way."""
    parser = argparse.ArgumentParser(prog="dcn_sweep.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", "-e", default="joystick", choices=sorted(BASES),
                        help="experiment whose trajectory, initialWts (and nTrials, unless --trials) are used")
    parser.add_argument("--bmi", default=None, choices=sorted(BMI_SWEEPS), help="which BMI to sweep (default: num_drop)")
    parser.add_argument("--list", action="store_true",
                        help="list the BMIs (or just --bmi) with their settings, defaults and swept values, then exit")
    parser.add_argument("--sweep", nargs="+", default=None, metavar="NAME",
                        help="BMI settings to sweep over their default values (default: the BMI's first setting)")
    parser.add_argument("--set", type=parse_set, action="append", default=[], metavar="NAME=V1,V2,...",
                        help="values of one BMI setting (repeatable); more than one value sweeps it")
    parser.add_argument("--values", type=float, nargs="+", default=None,
                        help="values of the one swept setting (default: its values in BMI_SWEEPS)")
    parser.add_argument("--trials", type=int, default=None, help="trials per run (default: experiment's nTrials)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per combination with different seeds (error bars)")
    parser.add_argument("--last", type=int, default=10, help="final MAE = mean MAE of the last N trials")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel runs")
    parser.add_argument("--noReset", action="store_true",
                        help="don't reset the arm to a random start point each trial (same as runExperiment.py)")
    parser.add_argument("--plot_only", action="store_true", help="re-plot saved results without running")
    parser.add_argument("--hide_output", action="store_true", help="save plots without showing them")
    parser.add_argument("--name", default=None,
                        help="name of the results folder and plot titles (default: <experiment>_<bmi>[_<swept settings>])")
    parser.add_argument("--dry_run", action="store_true", help="check the settings and print the runs without running them")
    args = parser.parse_args(argv)
    if args.list:
        print_settings([args.bmi] if args.bmi else sorted(BMI_SWEEPS))
        return
    args.bmi = args.bmi or "num_drop"

    if args.hide_output:
        plt.switch_backend("Agg")
    sweep = BMI_SWEEPS[args.bmi]
    values = setting_values(sweep, args, parser)
    names = list(values)
    swept = [n for n in names if len(values[n]) > 1]
    fixed = {n: values[n][0] for n in names if n not in swept}

    if args.name:
        out = OUT / args.name
        title = args.name
    else:
        out = OUT / f"{args.experiment}_{args.bmi}"
        if len(names) > 1:                   # keep sweeps of different settings apart
            out = out.with_name(out.name + "_" + ("_".join(swept) or "fixed"))
        title = f"{args.experiment}, {args.bmi}"
    if len(names) > 1 and fixed:
        title += " (" + ", ".join(f"{n}={fmt(v)}" for n, v in fixed.items()) + ")"
    plot_name = swept[0] if swept else names[0]
    can_plot = len(swept) <= 1

    if args.plot_only:
        if not (out / "results.npz").exists():
            parser.error(f"no saved sweep at {out}/")
        saved = np.load(out / "results.npz")
        if "grid" in saved:
            saved_names = list(saved["names"])
            if "swept" in saved:
                saved_swept = list(saved["swept"])
            else:                             # saved before the swept settings were stored
                saved_swept = [n for i, n in enumerate(saved_names) if len(set(saved["grid"][:, i])) > 1]
            if len(saved_swept) > 1:
                parser.error(f"{out}/ sweeps {', '.join(saved_swept)} together; there is no plot for that, use results.csv")
            plot_values = saved["grid"][:, saved_names.index(saved_swept[0] if saved_swept else saved_names[0])]
        else:                                 # saved before sweeps had named settings
            plot_values = saved["values"]
        curves = saved["curves"]
        final = curves[:, :, -int(saved["last"]):].mean(axis=2)
        plot(plot_values, final, curves, sweep.params[plot_name].label, title, out, not args.hide_output)
        return

    exp = BASES[args.experiment]
    problems = exp.validate()
    if problems:
        parser.error(f"cannot run '{exp.name}': " + "; ".join(problems))
    combos = [dict(zip(names, c)) for c in itertools.product(*(values[n] for n in names))]
    brain = build_brain(exp)
    runnable, used, skipped = [], [], []      # used: the values each BMI really runs with (worked-out ones filled in)
    for settings in combos:
        try:
            bmi = sweep.build(0, **settings).setup(brain)
        except ProbabilityError as e:
            skipped.append(f"{', '.join(f'{n}={fmt(v)}' for n, v in settings.items())}: {e}")
            continue
        except BMIError as e:
            parser.error(f"BMI settings: {e}")
        runnable.append(settings)
        used.append({n: getattr(bmi, n, settings[n]) for n in names})
    if skipped:
        print(f"Skipping {len(skipped)} of {len(combos)} combinations whose probabilities don't add up to 1:")
        for line in skipped:
            print(f"  {line}")
    if not runnable:
        parser.error("no combination of the BMI settings is valid (see above)")
    combos = runnable
    n_trials = args.trials or exp.nTrials
    jobs = [(args.experiment, args.bmi, i, settings, seed, n_trials, not args.noReset)
            for i, settings in enumerate(combos) for seed in range(args.repeats)]
    print(f"Running {len(jobs)} runs of {n_trials} trials on '{args.experiment}' with BMI '{args.bmi}' ({args.workers} workers)...")
    print(f"Sweeping: {', '.join(f'{n}={values[n]}' for n in swept) or 'nothing'}"
          + (f"; fixed: {', '.join(f'{n}={fmt(v)}' for n, v in fixed.items())}" if fixed else ""))
    print(f"Starting weights: {exp.initialWts or 'default'}")
    print(f"Results folder: {out}/")
    if args.dry_run:
        print("[dry run] settings are valid, nothing was run")
        return

    curves = np.zeros((len(combos), args.repeats, n_trials))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, seed, mae in pool.map(run_one, jobs):
            curves[index, seed] = mae
    final = curves[:, :, -args.last:].mean(axis=2)

    out.mkdir(parents=True, exist_ok=True)
    grid = np.array([[u[n] for n in names] for u in used], dtype=np.float64)
    np.savez(out / "results.npz", names=names, swept=swept, grid=grid, curves=curves, last=args.last)
    with open(out / "results.csv", "w") as f:
        f.write(",".join(names) + ",seed,final_mae\n")
        for i, settings in enumerate(used):
            for s in range(args.repeats):
                f.write(",".join(f"{settings[n]:g}" for n in names) + f",{s},{final[i, s]:.5f}\n")
    print(f"Saved results to {out}/")
    if can_plot:
        plot(grid[:, names.index(plot_name)], final, curves, sweep.params[plot_name].label, title, out,
             not args.hide_output)
    else:
        print(f"Swept {len(swept)} settings together: not plotting, see {out / 'results.csv'}")


if __name__ == "__main__":
    main()
