"""
Experiment runner

Usage:
    python experiment_runner.py --list
    python experiment_runner.py --experiment baseline_v1
    python experiment_runner.py --experiment baseline_v1 --dry-run

Add a new experiment by adding one Experiment(...) entry to EXPERIMENTS below.
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from registry import EXPERIMENTS
import model_v01
import play_arm_path
import matplotlib.pyplot as plt

def print_list() -> None:
    width = max(len(name) for name in EXPERIMENTS) + 2
    print("Available experiments:\n")
    for exp in EXPERIMENTS.values():
        print(f"  {exp.name:<{width}} {exp.description}")
    print(f"\nRun one with: python {Path(__file__).name} --experiment <name>")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
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

    args = parser.parse_args()

    if args.list or not args.experiment:
        print_list()
        sys.exit(0 if args.list else 1)

    exp = EXPERIMENTS[args.experiment]
    problems = exp.validate()
    if problems:
        print(f"Cannot run '{exp.name}':")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    if args.dry_run:
        print(f"[dry run] '{exp.name}' looks valid:")
        print(f"  trajectory: {exp.trajectory}")
        print(f"  actualArm:      {exp.actualArm}")
        print(f"  illusoryArm:      {exp.illusoryArm}")
        print(f"  results:      {exp.results}")
        print(f"  initWts:      {exp.initialWts}")
        print(f"  nDof:      {exp.nDof}\n")
        return

    if not args.view:
        model_v01.simulate(exp, args.save, not args.hide_output, not args.no_grav)
    else:
        try:
            armPlot   = plt.imread(exp.graphs.with_name(exp.graphs.stem + "_arm" + exp.graphs.suffix))
            brainPlot = plt.imread(exp.graphs.with_name(exp.graphs.stem + "_brain" + exp.graphs.suffix))
            plt.imshow(armPlot)
            plt.axis('off')
            plt.show()
            plt.axis('off')
            plt.imshow(brainPlot)
            plt.show()
            play_arm_path.play_traj(exp)
        except FileNotFoundError:
            print("Cannot find results to play. Running the sim and saving the results, then playing traj...")
            model_v01.simulate(exp, True, True, True)


if __name__ == "__main__":
    main()