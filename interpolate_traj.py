'''
Resamples a trajectory onto a finer time grid by linear interpolation, so the arm's
physics can be integrated with a smaller timestep (the sim steps once per trajectory sample).

usage: python interpolate_traj.py --factor 8
       python interpolate_traj.py --input trajectories/circleTrajPos.csv --factor 8 --output trajectories/circleTrajInterp8.csv
'''
import argparse
from pathlib import Path
import numpy as np
from experiment_assets import load_trajectory
from create_trajectories import save_data


def interpolate(traj, factor):
    '''
    Args:
        traj: Trajectory with a uniform time grid
        factor: how many samples per original sample interval (2 halves the timestep)
    Returns:
        time, {field: resampled array} for every field the trajectory has
    '''
    t = traj.time
    fine_t = np.linspace(t[0], t[-1], (len(t) - 1) * factor + 1)
    fields = {}
    for name in ("position", "velocity", "acceleration", "joint_position", "joint_velocity", "joint_acceleration"):
        data = getattr(traj, name)
        if data is not None:
            fields[name] = np.column_stack([np.interp(fine_t, t, col) for col in data.T])
    return fine_t, fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='trajectories/circleTrajPos.csv', help='trajectory csv to resample')
    parser.add_argument('--factor', type=int, default=2, help='samples per original sample interval (timestep / factor)')
    parser.add_argument('--output', default=None, help='output csv (default: <input>Interp<factor>.csv)')
    args = parser.parse_args()
    if args.factor < 1:
        parser.error('--factor must be at least 1')

    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_name(f"{src.stem.replace('Pos', '')}Interp{args.factor}.csv")
    traj = load_trajectory(src)
    time, fields = interpolate(traj, args.factor)
    # keep the sidecar (encoding / supervisor ranges etc.) with the resampled trajectory
    save_data(fname=out, time=time, meta=traj.meta or None, **fields)
    print(f"{src} ({len(traj.time)} samples, dt {np.diff(traj.time).mean()*1e3:.3f} ms) -> "
          f"{out} ({len(time)} samples, dt {np.diff(time).mean()*1e3:.3f} ms)")


if __name__ == "__main__":
    main()
