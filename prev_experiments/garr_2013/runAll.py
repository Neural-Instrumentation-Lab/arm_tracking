"""
Run every registered experiment back to back to (re)generate all data.

Usage:
    python run_all.py                    # run everything, saving results
    python run_all.py --no-save          # run everything but don't save (dry pass)
    python run_all.py --show_output      # don't suppress each sim's normal output
    python run_all.py --only baseline_v1 high_noise   # run a subset by name

By default this runs with save=True and hide_output=True for every experiment,
since the point of a batch run is usually "regenerate all the data quietly."
Both are overridable per the flags above.

DEPENDENCY ORDERING: if experiment B's initialWts points at experiment
A's results path (and that file doesn't already exist on disk), B is treated
as depending on A, and A is automatically run first. This is inferred from
the registry, not something you need to declare separately. Circular
dependencies, and dependencies excluded from --only, are reported as errors
before anything runs.

A failure in one experiment does not stop the rest — everything runs, and a
pass/fail summary prints at the end.
"""

import argparse
import sys
import time

from experiment_assets import Experiment, build_dependencies
from registry import EXPERIMENTS
from model_v01 import simulate

def topological_order(to_run: list[Experiment], deps: dict[str, set[str]]) -> list[Experiment]:
    """Order to_run so every experiment comes after everything it depends on.
    Raises ValueError on a circular dependency or a dependency that isn't
    part of this run and doesn't already exist on disk."""
    by_name = {exp.name: exp for exp in to_run}
    in_this_run = set(by_name)

    visited: set[str] = set()
    in_progress: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in in_progress:
            raise ValueError(f"Circular dependency detected involving '{name}'")
        in_progress.add(name)
        for dep in sorted(deps.get(name, ())):
            if dep not in in_this_run:
                raise ValueError(
                    f"'{name}' needs '{dep}' to run first (via initialWts), "
                    f"but '{dep}' is not included in this run. Add it to --only, "
                    f"run without --only, or make sure its results file already exists."
                )
            visit(dep)
        in_progress.discard(name)
        visited.add(name)
        order.append(name)

    for name in sorted(by_name):
        visit(name)

    return [by_name[n] for n in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="Only run these experiment names instead of the full registry.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results (default is to save).",
    )
    parser.add_argument(
        "--show_output", action="store_true",
        help="Don't hide each sim's normal output (default is to hide it during batch runs).",
    )
    parser.add_argument(
        "--makeMovie", action="store_true",
        help="Makes videos for all the experiments (MORE THAN DOUBLES RUNTIME).",
    )

    args = parser.parse_args()

    if args.only:
        unknown = [n for n in args.only if n not in EXPERIMENTS]
        if unknown:
            print(f"Unknown experiment name(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(sorted(EXPERIMENTS))}")
            sys.exit(1)
        to_run = [EXPERIMENTS[n] for n in args.only]
    else:
        to_run = list(EXPERIMENTS.values())

    save = not args.no_save
    hide_output = not args.show_output

    # Resolve dependency order (may raise on cycles / missing dependencies).
    try:
        deps = build_dependencies(to_run, EXPERIMENTS)
        to_run = topological_order(to_run, deps)
    except ValueError as e:
        print(f"Dependency error: {e}")
        sys.exit(1)

    if any(deps[exp.name] for exp in to_run):
        print("Run order (adjusted for dependencies):")
        for exp in to_run:
            dep_note = f"  (needs: {', '.join(sorted(deps[exp.name]))})" if deps[exp.name] else ""
            print(f"  {exp.name}{dep_note}")
        print()

    # Validate everything up front so a typo'd path fails fast, before any
    # experiment has partially run. Results this batch will itself produce
    # are treated as OK even though they don't exist on disk yet.
    print(f"Validating {len(to_run)} experiment(s)...")
    results_this_run = frozenset(exp.finalWts for exp in to_run if exp.finalWts is not None)
    all_problems = {}
    for exp in to_run:
        problems = exp.validate(assume_produced=results_this_run)
        if problems:
            all_problems[exp.name] = problems

    if all_problems:
        print("\nValidation failed for:")
        for name, problems in all_problems.items():
            print(f"  {name}:")
            for p in problems:
                print(f"    - {p}")
        print("\nFix the above before running. Nothing was run.")
        sys.exit(1)

    print("All experiments valid. Running...\n")

    results = []  # (name, ok, elapsed_seconds, error_or_None)
    for i, exp in enumerate(to_run, 1):
        print(f"[{i}/{len(to_run)}] {exp.name} ... ", end="", flush=True)
        start = time.time()
        try:
            simulate(exp, save=save, showOutput= not hide_output, grav=True, makeMovie=args.makeMovie)
            elapsed = time.time() - start
            results.append((exp.name, True, elapsed, None))
            print(f"ok ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            results.append((exp.name, False, elapsed, str(e)))
            print(f"FAILED ({elapsed:.1f}s): {e}")

    n_ok = sum(1 for _, ok, _, _ in results if ok)
    n_failed = len(results) - n_ok
    total_time = sum(elapsed for _, _, elapsed, _ in results)

    print(f"\n{'=' * 50}")
    print(f"Done: {n_ok}/{len(results)} succeeded, {n_failed} failed, {total_time:.1f}s total")
    if n_failed:
        print("\nFailed:")
        for name, ok, _, error in results:
            if not ok:
                print(f"  {name}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()