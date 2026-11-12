"""
Experiment runner

Usage:
    python runExperiment.py --list
    python runExperiment.py --experiment joystick
    python runExperiment.py --experiment joystick --dry-run

BMI sweeps (SweepExperiment entries) run through dcn_sweep.py:
    python runExperiment.py -e sweep_cross_cluster                  # run the sweep
    python runExperiment.py -e sweep_cross_cluster --dry-run        # check it, list the runs
    python runExperiment.py -e sweep_cross_cluster --view           # re-plot its saved results
    python runExperiment.py -e sweep_cross_cluster -- --repeats 3   # args after -- are added to dcn_sweep's

Add a new experiment by adding one Experiment(...) or SweepExperiment(...) entry to EXPERIMENTS in registry.py.
"""

from __future__ import annotations
import argparse
import dataclasses
import sys
from dataclasses import dataclass
from pathlib import Path
from registry import EXPERIMENTS
import model_v02
import play_arm_path
import matplotlib.pyplot as plt
from experiment_assets import build_dependencies, SweepExperiment
import dcn_sweep
from bmi import DroppedDCNBMI, SpikeDropBMI, ClusterBMI

def print_list() -> None:
    width = max(len(name) for name in EXPERIMENTS) + 2
    print("Available experiments:\n")
    for exp in EXPERIMENTS.values():
        kind = "[sweep] " if isinstance(exp, SweepExperiment) else ""
        print(f"  {exp.name:<{width}} {kind}{exp.description}")
    print(f"\nRun one with: python {Path(__file__).name} --experiment <name>")

def check_runnable(exp) -> bool:
    """Print why exp can't run (missing files, experiments to run first). True if it can."""
    problems = exp.validate()
    if not problems:
        return True
    deps = build_dependencies([exp], EXPERIMENTS)[exp.name]
    if deps:
        print(f"dependency issue. Run the following experiments with --save to produce the required dependencies: {sorted(deps)}")
    print(f"Cannot run '{exp.name}':")
    for p in problems:
        print(f"  - {p}")
    return False


def run_sweep(exp: SweepExperiment, args, extra: list[str], parser) -> None:
    """Run a registry sweep through dcn_sweep.py, with its base experiment and args."""
    unsupported = [flag for flag, used in (
        ("--drop_dcn", args.drop_dcn is not None), ("--drop_spike", args.drop_spike is not None),
        ("--cluster_dcn", args.cluster_dcn is not None), ("--makeMovie", args.makeMovie is not None),
        ("--no_grav", args.no_grav)) if used]
    if unsupported:
        parser.error(f"'{exp.name}' is a BMI sweep, so {', '.join(unsupported)} don't apply "
                     f"(BMI settings go in its args in registry.py, or after --)")
    try:
        exp = dataclasses.replace(exp, args=exp.args + tuple(extra))   # re-checks the args
    except ValueError as e:
        parser.error(str(e))
    if not args.view and not check_runnable(EXPERIMENTS[exp.base]):
        sys.exit(1)
    argv = ["-e", exp.base, "--name", exp.name, *exp.args]
    if args.noReset:
        argv.append("--noReset")
    if args.hide_output:
        argv.append("--hide_output")
    if args.dry_run:
        argv.append("--dry_run")
    if args.view:
        argv.append("--plot_only")
    print(f"'{exp.name}': python dcn_sweep.py {' '.join(argv)}")
    dcn_sweep.main(argv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment", "-e",
        choices=sorted(EXPERIMENTS),   # argparse validates + shows valid names on error
        help="Name of the experiment to run.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available experiments with descriptions and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate file paths and print the config without running anything.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Stores the results in the result file after experiment runs.")
    parser.add_argument(
        "--view", action="store_true",
        help="Does not rerun experiment, but opens simulator to view the resulting trajectories"
    )
    parser.add_argument(
        "--hide_output", action="store_true",
        help="Reruns experiment but suppresses all output"
    )
    parser.add_argument(
        "--no_grav", action="store_true",
        help="Turns gravity off for the experiment"
    )
    parser.add_argument(
        "--makeMovie", type=int, nargs="?", const=5, default=None, metavar="N",
        help="saves an mp4 timelapse of the learning to the experiment's videos path, showing every Nth trial (default 5)"
    )
    parser.add_argument(
        "--drop_dcn", type=int, default=None, metavar="N",
        help="BMI drops N random DCNs from every agonist/antagonist group (overrides the experiment's bmi)"
    )
    parser.add_argument(
        "--drop_spike", type=float, default=None, metavar="P",
        help="BMI loses each DCN spike with probability P, in [0, 1) (overrides the experiment's bmi)"
    )
    parser.add_argument(
        "--cluster_dcn", type=int, default=None, metavar="N",
        help="BMI reads clusters of N DCNs, a spike from any of them counts as one (overrides the experiment's bmi)"
    )
    parser.add_argument(
        "--noReset", action="store_true",
        help="Don't reset the arm to a random start point each trial; trials run back to back"
    )

    # anything after a bare -- is passed on to dcn_sweep.py (sweep experiments only)
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)

    if args.list or not args.experiment:
        print_list()
        sys.exit(0 if args.list else 1)

    exp = EXPERIMENTS[args.experiment]
    if isinstance(exp, SweepExperiment):
        run_sweep(exp, args, extra, parser)
        return
    if extra:
        parser.error(f"arguments after -- are for BMI sweeps; '{exp.name}' is a single experiment")
    if args.makeMovie is not None and args.makeMovie < 1:
        parser.error("--makeMovie must be at least 1")
    if args.drop_dcn is not None:
        n_per_group = model_v02.cerebellum.nDCN // 2
        if not 0 <= args.drop_dcn < n_per_group:
            parser.error(f"--drop_dcn must be between 0 and {n_per_group - 1} (each group has {n_per_group} DCNs)")
        exp = dataclasses.replace(exp, bmi=DroppedDCNBMI(args.drop_dcn))
    if sum(a is not None for a in (args.drop_dcn, args.drop_spike, args.cluster_dcn)) > 1:
        parser.error("use only one of --drop_dcn, --drop_spike and --cluster_dcn")
    if args.cluster_dcn is not None:
        n_per_group = model_v02.cerebellum.nDCN // 2
        if not 1 <= args.cluster_dcn <= n_per_group:
            parser.error(f"--cluster_dcn must be between 1 and {n_per_group} (each group has {n_per_group} DCNs)")
        exp = dataclasses.replace(exp, bmi=ClusterBMI(args.cluster_dcn))
    if args.drop_spike is not None:
        if not 0 <= args.drop_spike < 1:
            parser.error("--drop_spike must be at least 0 and below 1")
        exp = dataclasses.replace(exp, bmi=SpikeDropBMI(args.drop_spike))
    if not check_runnable(exp):
        sys.exit(1)

    if args.dry_run:
        print(f"[dry run] '{exp.name}' looks valid:")
        print(f"  trajectory: {exp.trajectory}")
        print(f"  results:      {exp.results}")
        print(f"  initWts:      {exp.initialWts}")
        print(f"  nDof:      {exp.nDof}")
        print(f"  qMins:      {exp.qMins}")
        print(f"  qMaxs:      {exp.qMaxs}")
        print(f"  brain_timestep:      {exp.brain_timestep}")
        print(f"  bmi:      {exp.bmi}")
        if args.makeMovie:
            print(f"  video:      {exp.videos} (every {args.makeMovie} trials)")
        print()
        return

    if not args.view:
        model_v02.simulate(exp, args.save, not args.hide_output, args.makeMovie, reset=not args.noReset)
    else:
        try:
            armPlot   = plt.imread(exp.graphs.with_name(exp.graphs.stem + "_commands" + exp.graphs.suffix))
            brainPlot = plt.imread(exp.graphs.with_name(exp.graphs.stem + "_MAE" + exp.graphs.suffix))
            maePlot = plt.imread(exp.graphs.with_name(exp.graphs.stem + "_pf_pc_wts" + exp.graphs.suffix))
            plt.imshow(armPlot)
            plt.axis('off')
            plt.show()
            plt.axis('off')
            plt.imshow(brainPlot)
            plt.show()
            plt.axis('off')
            plt.imshow(maePlot)
            plt.show()
            # results saved before the raster existed don't have one
            # results saved before these plots existed don't have them; _dcn_roles only exists for BMIs that misread DCNs
            for extra in ("_dcn_raster", "_dcn_roles"):
                extraFile = exp.graphs.with_name(exp.graphs.stem + extra + exp.graphs.suffix)
                if extraFile.exists():
                    plt.axis('off')
                    plt.imshow(plt.imread(extraFile))
                    plt.show()
            play_arm_path.play_traj(exp)
        except FileNotFoundError:
            print("Cannot find results to play. Running the sim and saving the results, then playing traj...")
            model_v02.simulate(exp, True, True, args.makeMovie, reset=not args.noReset)


if __name__ == "__main__":
    main()