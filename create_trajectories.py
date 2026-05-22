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
    f = 1
    fs = 60 
    t = np.arange(0, 10, 1/fs)
    x = np.linspace(0, 0.5, len(t)) 
    y = np.linspace(0, -0.2, len(t)) 
    z = np.linspace(1.1, 0.3, len(t)) 
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
    fs = 10000
    t = np.arange(0, 3, 1/fs)
    x = 0.4*np.cos(2*np.pi*f*t)
    y = [0 for _ in x] # this arm has no dof in the y-axis
    z = np.array([0.6 for _ in range(len(x))])
    array = build_array(t,x,y,z)
    fname = 'trajectories/traj_004.csv'
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

    # complain if user requests an unimplemented trajectory
    else: raise ValueError(f"Trajectory {args.trajectory_id} not found\n")


if __name__ == "__main__":
    main()