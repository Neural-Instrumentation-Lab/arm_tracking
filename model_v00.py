'''
MODEL V00

First implementation of 2D arm movement with cerebellar
error correction

@Author: Iyad Obeid
@Date: Fall 2023

'''

## IMPORTS #####################################################################
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from atrack_assets import simplest_2dof_limb, simplest_2dof_controller, cerebellum_marr_albus

matplotlib.use('TkAgg')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S"
)


###################################
def get_trajectory(fname:str):
###################################
    ''' load desired end effector trajectory from csv file
    
    Parse filename, prepend directory name, load from csv, and parse into columns
    
    Args:
        fname:  file name
    Returns:
        traj:   n_samples x n_dims trajectory data
        t:      n_samples time data
        n_dim:  number of dimensions in trajectory
    '''

    # append .csv to filename if not specified
    if fname[-3:] != 'csv':
        fname += '.csv'

    # prepend directory name
    fname = 'trajectories/' + fname

    # load data - time should always be in column 0
    data = np.loadtxt(fname,delimiter=',')

    # parse columns
    t     = data[:,0]
    traj  = data[:,1:]
    n_dim = traj.shape[1]

    # exit gracefully
    return traj,t,n_dim

###################################
def parse_args():
###################################
    '''
    Parses commandline arguments

    Collects trajectory file name from commandline

    Returns:
        fname(string)
    '''
    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_file" , nargs='?', default='traj_001.csv', help='trajectory file name')

    # parse args and extract filename
    args  = parser.parse_args()
    fname = args.traj_file

    return fname

###################################
def plot_results(desired_position , actual_limb_location):
###################################    
    '''
    plots desired and actual end effector positions

    Args:
        desired_position:       trajectory that end effector was trying to achieve
        actual_limb_position:   trajectory that end effector actually acheived
    '''
    
    plt.plot(desired_position[:,0]     , desired_position[:,1], label='Desired Position')
    plt.plot(actual_limb_location[:,0] , actual_limb_location[:,1], label='Actual Position')
    plt.xlim([-10,10])
    plt.ylim([10,15])
    plt.legend()
    plt.show()

###################################
def main():
###################################

    # load user preferences from command line
    fname = parse_args()

    # load end effector trajectory
    desired_position, t, n_dimensions = get_trajectory(fname)   

    # init variables
    actual_limb_location = np.zeros_like(desired_position)
    correction = np.zeros(n_dimensions)

    # instantiate limb, motor control unit, brain    
    # FIXME simulation only works when L1 and L2 are exactly 10 & 5
    # compare against matlab code and see if we lost a minus sign or something
    limb       = simplest_2dof_limb()
    motor_ctrl = simplest_2dof_controller(L1=10.01 , L2=5 )
    brain      = cerebellum_marr_albus()

    # iterate control / learning algorithm over time
    for i,waypoint in enumerate(desired_position):

        joint_angles   = motor_ctrl.get_joint_angles(waypoint + correction)
        limb_location  = limb.move(joint_angles)
        movement_error = waypoint - limb_location

        brain.update(movement_error)
        correction = brain.compute_correction(joint_angles)

        # store outcomes
        actual_limb_location[i,:] = limb_location

    # plot outcomes
    plot_results(desired_position , actual_limb_location)

main()
