'''
CREATE TRAJECTORIES

A group of functions for creating arm virtual arm trajectories
for the simulated 2D arm to attempt to track.

@Author: Iyad Obeid
@Date: Fall 2023

'''

## IMPORTS #####################################################################
from __future__ import annotations
import numpy as np
import sympy as sp
import argparse
from arm_assets_v02 import dynamic_3dof_arm
from pathlib import Path
import yaml

TRAJ = Path("trajectories")
## HELPER FUNCTIONS ############################################################

###################################
def linspace(a,b,dx):
###################################
    '''linspace(a,b,dx)
        Creates an array of numbers from a to b with spacing dx
    Args:
        a: starting point
        b: ending point
        dx: spacing between points
    Returns:
        array of values
    '''
    nPts = int( (b-a)/dx + 1 )
    x = [a + dx*i for i in range(nPts)]
    return x

###################################
def build_array(t,x,y, z=np.array([])):
###################################
    '''build_array(t,x,y)
        Builds a 2D array of three columns, suitable for saving into a csv file
        or a 3D array of four columns
    Args:
    Returns:'''
    if z.size == 0:
        return np.array([t,x,y]).T
    else:
        return np.array([t,x,y,z]).T

###################################
def get_trajectory_list():
###################################
    '''Get the list of trajectory functions and what they do
    
    Args:
        none
    Returns:
        string  trajectory table-of-contents
    '''
    str = '\n'
    str += '0   : single point located at (7.07, 12.07)\n'
    str += '1   : horizontal line at y = 12\n'
    return str

## TRAJECTORY WRITING FUNCTIONS ################################################
###################################
def create_traj_003():
###################################
    ''' Trajectory 03- a line for the 3dof arm 
    '''
    f = 2
    fs = 500 
    t = np.arange(0, 1, 1/fs)
    x = np.linspace(0.3, 0.1, len(t)) 
    y = np.linspace(0.4, -0.2, len(t)) 
    z = np.linspace(0.8, 0.5, len(t)) 
    pos = np.column_stack([x, y, z])
    fname = TRAJ / "traj_003.csv"
    save_data(fname,time=t, position=pos)

###################################
def create_traj_004():
###################################
    ''' Trajectory 04 - a sinusoid 
        for dynamic 2dof arm (static link .31 high)
    '''
    f = 1
    fs = 500
    t = np.arange(0, 10, 1/fs)
    x = 0.4*np.cos(2*np.pi*f*t)
    y = [0 for _ in x]
    z = np.array([0.6 for _ in range(len(x))])
    pos = np.column_stack([x, y, z])
    fname = TRAJ / "traj_004.csv"
    save_data(fname,time=t, position=pos)

###################################
def create_traj_005():
###################################
    ''' Trajectory 05 - a sinusoid in x and z
        for dynamic 2dof arm (static link .31 high)
    '''
    f = 2
    f2 = 2
    fs = 500
    t = np.arange(0, 6, 1/fs)
    x = 0.5*np.cos(2*np.pi*f*t)
    y = [0 for _ in x]
    z = np.array(0.2*np.cos(2*np.pi*f2*t) + 0.6)
    pos = np.column_stack([x, y, z])
    fname = TRAJ / "traj_005.csv"
    save_data(fname,time=t, position=pos)

def create_traj_006():
    ''' trajectories in the garrido matlab
    from cin_inv_och3joints.m 
    it is a 8-figure trajectory but not 
    shown off in their paper 
    '''
    fs = 1500
    t = np.arange(0, 1, 1/fs)
    y = 0.1 * np.sin(2*np.pi * t)+0.21502
    z = 0.1 * np.sin(4 * np.pi * t)+0.18502
    x = np.ones(len(t)) * 0.5
    dx = np.zeros_like(t)
    dy = 0.1*2*np.pi*np.cos(2*np.pi*t)
    dz = 0.1*4*np.pi*np.cos(4*np.pi*t)
    ddx = np.zeros_like(t)
    ddy = -0.1*4*np.pi*np.pi*np.sin(2*np.pi*t)
    ddz = -0.1*16*np.pi*np.pi*np.sin(4*np.pi*t)
    pos = np.column_stack([x, y, z])
    vel = np.column_stack([dx, dy, dz])
    acel = np.column_stack([ddx, ddy, ddz])
    fname = TRAJ / "traj_006.csv"
    save_data(fname, time=t, position=pos, velocity=vel, acceleration=acel)

def create_traj_007():
    '''
    traj 6 but with lower fs 
    '''
    fs = 500
    t = np.arange(0, 1, 1/fs)
    y = 0.1 * np.sin(2*np.pi * t)+0.21502
    z = 0.1 * np.sin(4 * np.pi * t)+0.18502
    x = np.ones(len(t)) * 0.5
    dx = np.zeros_like(t)
    dy = 0.1*2*np.pi*np.cos(2*np.pi*t)
    dz = 0.1*4*np.pi*np.cos(4*np.pi*t)
    ddx = np.zeros_like(t)
    ddy = -0.1*4*np.pi*np.pi*np.sin(2*np.pi*t)
    ddz = -0.1*16*np.pi*np.pi*np.sin(4*np.pi*t)
    pos = np.column_stack([x, y, z])
    vel = np.column_stack([dx, dy, dz])
    acel = np.column_stack([ddx, ddy, ddz])
    fname = TRAJ / "traj_007.csv"
    save_data(fname, time=t, position=pos, velocity=vel, acceleration=acel)

def create_traj_008():
    '''
    this is a similar 8-shaped trajectory as used
    in the garrido paper, in fact it is the traj in their fig 5C, 
    but the actual traj they used for their results was traj 9,
    not this trajectory.
    '''
    fs = 1000
    t = np.arange(0, 1, 1/fs)
    q3 = 0.1*np.sin(2*np.pi*t) + np.pi/2
    q2 = 0.1*np.sin(2*np.pi*t + np.pi/4)
    q1 = 0.1*np.sin(2*np.pi*t + np.pi/2) 
    positions = np.concat([q1.reshape(-1, 1), q2.reshape(-1, 1), q3.reshape(-1, 1)], axis=1)
    arm = dynamic_3dof_arm("models/arm_3dof.urdf", disp=False)
    coords = np.zeros_like(positions)
    for i, pos in enumerate(positions):
        coords[i] = arm.getEEFromJoint(pos)
    fname = TRAJ / "traj_008.csv"
    save_data(fname, time=t, position=coords, joint_position=positions)

def rescale_to_corr(coord, lengths):
    '''
    rescales out of bounds positions back to reachability
    sometimes coords get right out reach due to floating point
    computation so this is necessary
    '''
    x, y, z = coord
    l1, l2, l3 = lengths
    D = np.linalg.norm([x, y, z-l1])
    scaleF = (l2+l3) / D
    return 0.9999 * coord * scaleF

def create_traj_009():
    '''
    this is the correct taco-shaped
    trajectory used in the garrido paper
    both cartesian and joint positions / derivatives
    are calculated analytically to reduce error
    '''
    fs = 1000
    # --- symbols ---
    t = sp.symbols('t', real=True)
    q1, q2, q3, l1, l2, l3 = sp.symbols('q1 q2 q3 l1 l2 l3')
    # --- DH-style transforms (same structure as the MATLAB code) ---
    A01 = sp.Matrix([
        [sp.cos(q1), 0,  sp.sin(q1), 0],
        [sp.sin(q1), 0, -sp.cos(q1), 0],
        [0,          1,  0,          l1],
        [0,          0,  0,          1]
    ])
    A12 = sp.Matrix([
        [sp.cos(q2), -sp.sin(q2), 0, l2*sp.cos(q2)],
        [sp.sin(q2),  sp.cos(q2), 0, l2*sp.sin(q2)],
        [0,           0,          1, 0],
        [0,           0,          0, 1]
    ])
    A23 = sp.Matrix([
        [sp.cos(q3), -sp.sin(q3), 0, l3*sp.cos(q3)],
        [sp.sin(q3),  sp.cos(q3), 0, l3*sp.sin(q3)],
        [0,           0,          1, 0],
        [0,           0,          0, 1]
    ])
    A03 = sp.simplify(A01 * A12 * A23)
    x_expr = A03[0, 3]
    y_expr = A03[1, 3]
    z_expr = A03[2, 3]
    # --- link lengths (substitute numeric values now, keep q1,q2,q3 symbolic) ---
    lr1, lr2, lr3 = 0.310, 0.4, 0.390
    subs_links = {l1: lr1, l2: lr2, l3: lr3}
    x_expr = x_expr.subs(subs_links)
    y_expr = y_expr.subs(subs_links)
    z_expr = z_expr.subs(subs_links)
    # --- joint trajectories as functions of t (symbolic, matches your Python q1/q2/q3) ---
    A = sp.Rational(1, 10)  # 0.1
    # The Trajectory from figure 5B
    # q1_t = A*sp.sin(2*sp.pi*t + sp.pi/2)
    # q2_t = A*sp.sin(2*sp.pi*t + sp.pi/4)
    # q3_t = A*sp.sin(2*sp.pi*t) + sp.pi/2
    q1_t =A*sp.sin(2*sp.pi*t)
    q2_t =A*sp.sin(2*sp.pi*t+sp.pi/4)
    q3_t =A*sp.sin(2*sp.pi*t+sp.pi/2)
    subs_q = {q1: q1_t, q2: q2_t, q3: q3_t}
    x_t = x_expr.subs(subs_q)
    y_t = y_expr.subs(subs_q)
    z_t = z_expr.subs(subs_q)
    # --- differentiate symbolically for velocity and acceleration ---
    vx_t, vy_t, vz_t = [sp.diff(e, t) for e in (x_t, y_t, z_t)]
    ax_t, ay_t, az_t = [sp.diff(e, t, 2) for e in (x_t, y_t, z_t)]
    # simplify (trig simplify can be slow but cleans things up)
    x_t, y_t, z_t = [sp.trigsimp(e) for e in (x_t, y_t, z_t)]
    vx_t, vy_t, vz_t = [sp.trigsimp(e) for e in (vx_t, vy_t, vz_t)]
    ax_t, ay_t, az_t = [sp.trigsimp(e) for e in (ax_t, ay_t, az_t)]
    # --- lambdify to numpy for fast numeric evaluation ---
    funcs = {
        'x': sp.lambdify(t, x_t, 'numpy'),
        'y': sp.lambdify(t, y_t, 'numpy'),
        'z': sp.lambdify(t, z_t, 'numpy'),
        'vx': sp.lambdify(t, vx_t, 'numpy'),
        'vy': sp.lambdify(t, vy_t, 'numpy'),
        'vz': sp.lambdify(t, vz_t, 'numpy'),
        'ax': sp.lambdify(t, ax_t, 'numpy'),
        'ay': sp.lambdify(t, ay_t, 'numpy'),
        'az': sp.lambdify(t, az_t, 'numpy'),
    }
    t_arr = np.arange(0, 1, 1/fs)
    pos = np.stack([funcs['x'](t_arr), funcs['y'](t_arr), funcs['z'](t_arr)], axis=1)
    arm = dynamic_3dof_arm("models/arm_3dof.urdf", disp=False)
    for i, coord in enumerate(pos):
        if not arm.is_valid_location(np.round(coord, 5)):
            fixed = rescale_to_corr(np.round(coord,5), [lr1, lr2, lr3])
            pos[i] = fixed
    vel = np.stack([funcs['vx'](t_arr), funcs['vy'](t_arr), funcs['vz'](t_arr)], axis=1)
    acc = np.stack([funcs['ax'](t_arr), funcs['ay'](t_arr), funcs['az'](t_arr)], axis=1)
    q1_r   = 0.1*np.sin(2*np.pi*t_arr)
    q2_r   = 0.1*np.sin(2*np.pi*t_arr+np.pi/4)
    q3_r   = 0.1*np.sin(2*np.pi*t_arr+np.pi/2)
    qd1_r  = 0.1*2*np.pi*np.cos(2*np.pi*t_arr)
    qd2_r  = 0.1*2*np.pi*np.cos(2*np.pi*t_arr+np.pi/4)
    qd3_r  = 0.1*2*np.pi*np.cos(2*np.pi*t_arr+np.pi/2)
    qdd1_r = -0.1*4*np.pi*np.pi*np.sin(2*np.pi*t_arr)
    qdd2_r = -0.1*4*np.pi*np.pi*np.sin(2*np.pi*t_arr+np.pi/4)
    qdd3_r = -0.1*4*np.pi*np.pi*np.sin(2*np.pi*t_arr+np.pi/2)
    q = np.column_stack([q1_r, q2_r, q3_r])
    qd = np.column_stack([qd1_r, qd2_r, qd3_r])
    qdd = np.column_stack([qdd1_r, qdd2_r, qdd3_r])
    fname = TRAJ / "traj_009.csv"
    save_data(fname,time=t_arr, position=pos, velocity=vel, acceleration=acc, joint_position=q, joint_velocity=qd, joint_acceleration=qdd) 

## MAIN ###################################################################

###################################
def parse_args():
###################################
    '''
    Parse arguements and determine whether to list existing trajectory functions
    or run one of them
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_id' , help='trajectory id list' )
    parser.add_argument('--list'          , help='list all trajectories' , action="store_true")

    args = parser.parse_args()    

    # make sure at least one arguement is specified
    if not args.list and not args.trajectory_id:
        raise ValueError("Must specify either --list or --trajectory_id [ID]\n")

    return args

###################################
def main():
###################################

    # retrieve user preferences
    args = parse_args()

    # print list of trajectories, if user requests it
    if args.list:
        print( get_trajectory_list() )
        return

    # create the specified trajectory
    elif args.trajectory_id == '3': create_traj_003()
    elif args.trajectory_id == '4': create_traj_004()
    elif args.trajectory_id == '5': create_traj_005()
    elif args.trajectory_id == '6': create_traj_006()
    elif args.trajectory_id == '7': create_traj_007()
    elif args.trajectory_id == '8': create_traj_008()
    elif args.trajectory_id == '9': create_traj_009()

    # complain if user requests an unimplemented trajectory
    else: raise ValueError(f"Trajectory {args.trajectory_id} not found\n")


_AXES = ["x", "y", "z"]
 
 
def _axis_columns(prefix: str, array: np.ndarray) -> tuple[list[str], np.ndarray]:
    """array: shape (N, k) with k in 1..3. Returns (['pos_x', 'pos_y', ...], array)."""
    array = np.atleast_2d(array)
    if array.shape[0] == 1 and array.shape[1] != 1:
        # allow passing a flat (N,) array for a single axis without forcing reshape upstream
        array = array.T
    n_axes = array.shape[1]
    if n_axes > 3:
        raise ValueError(f"{prefix}: expected at most 3 columns (x,y,z), got {n_axes}")
    cols = [f"{prefix}_{axis}" for axis in _AXES[:n_axes]]
    return cols, array
 
 
def _joint_columns(prefix: str, array: np.ndarray) -> tuple[list[str], np.ndarray]:
    """array: shape (N, n_dof). Returns (['joint_pos_1', 'joint_pos_2', ...], array)."""
    array = np.atleast_2d(array)
    n_dof = array.shape[1]
    cols = [f"{prefix}_{i}" for i in range(1, n_dof + 1)]
    return cols, array
 
 
def save_data(
    fname: str | Path,
    time: np.ndarray,
    *,
    position: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    acceleration: np.ndarray | None = None,
    joint_position: np.ndarray | None = None,
    joint_velocity: np.ndarray | None = None,
    joint_acceleration: np.ndarray | None = None,
    meta: dict | None = None,
    fmt: str = "%0.5f",
) -> None:
    """Save trajectory data to a headered CSV, with an optional YAML metadata sidecar.
 
    Args:
        fname: output path, e.g. "trajectories/traj_010.csv"
        time: shape (N,), required
        position, velocity, acceleration: shape (N, k) with k in 1..3 (x[,y[,z]]),
            or None to omit that group entirely
        joint_position, joint_velocity: shape (N, n_dof), or None to omit
        meta: optional dict written to a sidecar fname.yaml (units, coordinate_frame,
            sample_rate_hz, notes, etc). Anything that can't be inferred from column
            names goes here.
        fmt: numeric format string, passed to np.savetxt
 
    Returns:
        none
    """
    fname = Path(fname)
    time = np.asarray(time).reshape(-1, 1)
    n = time.shape[0]
 
    headers = ["time"]
    columns = [time]
 
    for prefix, array in [("pos", position), ("vel", velocity), ("acc", acceleration)]:
        if array is None:
            continue
        cols, arr = _axis_columns(prefix, array)
        if arr.shape[0] != n:
            raise ValueError(f"{prefix}: expected {n} rows to match time, got {arr.shape[0]}")
        headers.extend(cols)
        columns.append(arr)
 
    for prefix, array in [("joint_pos", joint_position), ("joint_vel", joint_velocity), ("joint_acc", joint_acceleration)]:
        if array is None:
            continue
        cols, arr = _joint_columns(prefix, array)
        if arr.shape[0] != n:
            raise ValueError(f"{prefix}: expected {n} rows to match time, got {arr.shape[0]}")
        headers.extend(cols)
        columns.append(arr)
 
    data = np.hstack(columns)
    fname.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(fname, data, delimiter=",", fmt=fmt, header=",".join(headers), comments="")
 
    if meta:
        meta_path = fname.with_suffix(".yaml")
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

if __name__ == "__main__":
    main()
