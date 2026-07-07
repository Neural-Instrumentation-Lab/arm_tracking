'''
CREATE TRAJECTORIES

A group of functions for creating arm virtual arm trajectories
for the simulated 2D arm to attempt to track.

@Author: Iyad Obeid
@Date: Fall 2023

'''

## IMPORTS #####################################################################
import numpy as np
import argparse
from arm_assets_v01 import dynamic_3dof_arm
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
def save_data(fname,array):
###################################
    '''save comma-separated data to text file

    Args:
        fname: filename for saving data
        array: data to be saved to file, column formatted
    Returns:
        none
    '''
    np.savetxt(fname , array , delimiter=',' , fmt="%0.5f")

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
def create_traj_000():
###################################
    ''' Trajectory 00 - a simple point'''
    t = [0]
    x = [7.07]
    y = [12.07]
    array = build_array(t,x,y)
    fname = 'trajectories/traj_000.csv'
    save_data(fname,array)

###################################
def create_traj_001():
###################################
    ''' Trajectory 01 - a horizontal line
        from (-8,12) to (8,12) at spacing 0.01
    '''
    dx = 0.01
    x = linspace(-8,8,dx)
    y = [12 for _ in range(len(x))]

    dt = 0.01
    t = [0+dt*i for i in range(len(x))]
    array = build_array(t,x,y)
    fname = 'trajectories/traj_001.csv'
    save_data(fname,array)

###################################
def create_traj_002():
###################################
    ''' Trajectory 02 - a sinusoid 
        from (-8,12) to (8,12) at spacing 0.01
    '''
    f = 1
    fs = 1000
    t = np.arange(0, 10, 1/fs)
    x = np.cos(2*np.pi*f*t)
    y = [12 for _ in range(len(x))]
    array = build_array(t,x,y)
    fname = 'trajectories/traj_002.csv'
    save_data(fname,array)

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
    array = build_array(t,x,y, z)
    fname = 'trajectories/traj_003.csv'
    save_data(fname,array)

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
    y = [0 for _ in x] # this arm has no dof in the y-axis
    z = np.array([0.6 for _ in range(len(x))])
    array = build_array(t,x,y,z)
    fname = 'trajectories/traj_004.csv'
    save_data(fname,array)

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
    y = [0 for _ in x] # this arm has no dof in the y-axis
    z = np.array(0.2*np.cos(2*np.pi*f2*t) + 0.6) # this arm has no dof in the y-axis
    array = build_array(t,x,y,z)
    fname = 'trajectories/traj_005.csv'
    save_data(fname,array)


def clip_to_reachable(pos, lengths):
    '''
    helper for create_traj_006
    only works for a 3dof arm
    '''
    L1, L2, L3 = lengths
    rel = np.array([pos[0], pos[1], pos[2] - L1])
    D = np.linalg.norm(rel)

    r_min, r_max = abs(L2 - L3), L2 + L3

    if D > r_max:
        rel = rel * (r_max / D)
    elif D < r_min:
        rel = rel * (r_min / D) 

    rel = np.array([rel[0], rel[1], rel[2] + L1])
    return rel 

def create_traj_006():
    ''' one(?) of the trajectories in the garrido matlab
    from cin_inv_och3joints.m 
    it is a 8-figure trajectory but not the one in 
    figure 5A in their 2013 paper
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
    array = np.array([t,x,y,z,dx,dy,dz,ddx,ddy,ddz]).T
    fname = 'trajectories/traj_006.csv'
    save_data(fname,array)

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
    array = np.array([t,x,y,z,dx,dy,dz,ddx,ddy,ddz]).T
    fname = 'trajectories/traj_007.csv'
    save_data(fname,array)

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
    if   args.trajectory_id == '0': create_traj_000()
    elif args.trajectory_id == '1': create_traj_001()
    elif args.trajectory_id == '2': create_traj_002()
    elif args.trajectory_id == '3': create_traj_003()
    elif args.trajectory_id == '4': create_traj_004()
    elif args.trajectory_id == '5': create_traj_005()
    elif args.trajectory_id == '6': create_traj_006()
    elif args.trajectory_id == '7': create_traj_007()

    # complain if user requests an unimplemented trajectory
    else: raise ValueError(f"Trajectory {args.trajectory_id} not found\n")


if __name__ == "__main__":
    main()