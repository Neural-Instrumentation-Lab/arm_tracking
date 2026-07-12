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

A failure in one experiment does not stop the rest — everything runs, and a
pass/fail summary prints at the end.
"""

import argparse
import sys
import time

from registry import EXPERIMENTS
import model_v01


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

    # Validate everything up front so a typo'd path fails fast, before any
    # experiment has partially run.
    print(f"Validating {len(to_run)} experiment(s)...")
    all_problems = {}
    for exp in to_run:
        problems = exp.validate()
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
            model_v01.simulate(exp, save, args.show_output, True, False)
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