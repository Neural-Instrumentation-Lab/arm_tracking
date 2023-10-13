import numpy as np
import matplotlib.pyplot as plt
import argparse

from atrack_assets import simplest_2dof_limb, simplest_2dof_controller, cerebellum

def get_trajectory(fname:str):
    '''
    load desired end effector trajectory from csv file
    
    Params:
        fname:  file name
    Returns:
        traj:   n_samples x n_dims trajectory data
        t:      n_samples time data
        n_dim:  number of dimensions in trajectory
    '''

    # append .csv to filename if not specified
    if fname[-3:] != 'csv':
        fname += '.csv'

    # prepend direcotry name
    fname = 'trajectories/' + fname

    # load data - time always in column 0
    data = np.loadtxt(fname,delimiter=',')

    # parse columns
    t    = data[:,0]
    traj = data[:,1:]
    n_dim = traj.shape[1]

    # exit gracefully
    return traj,t,n_dim

def parse_args():
    '''
    Parses commandline arguments

    Returns:
        fname(string)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("trj_file" , help='trajectory file name')
    args = parser.parse_args()
    fname = args.trj_file
    return fname

def plot_results(desired_position , actual_limb_location):
    '''
    plots desired and actual end effector positions
    '''
    
    plt.plot(desired_position[:,0]     , desired_position[:,1])
    plt.plot(actual_limb_location[:,0] , actual_limb_location[:,1])
    plt.xlim([-10,10])
    plt.ylim([10,15])
    plt.show()


def main():

    # load user preferences from command line
    fname = parse_args()

    # load end effector trajectory
    desired_position, t, n_dimensions = get_trajectory(fname)   

    # init variables
    actual_limb_location = np.zeros_like(desired_position)
    correction = np.zeros(n_dimensions)

    # instantiate limb, motor control unit, brain    
    limb       = simplest_2dof_limb()
    motor_ctrl = simplest_2dof_controller(L1=9.8 , L2=5.2 )
    brain      = cerebellum()

    # iterate control / learning algorithm over time
    for i,waypoint in enumerate(desired_position):

        joint_angles = motor_ctrl.get_joint_angles(waypoint + correction)
        limb_location = limb.move(joint_angles)
        movement_error = waypoint - limb_location

        brain.update(movement_error , joint_angles)
        correction = brain.compute_correction(joint_angles)

        # store outcomes
        actual_limb_location[i,:] = limb_location

    # plot outcomes
    plot_results(desired_position , actual_limb_location)

main()