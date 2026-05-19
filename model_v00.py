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
from atrack_assets import simplest_2dof_limb, simplest_2dof_controller, cerebellum_marr_albus, dynamic_3dof_arm
from pinocchio.visualize import MeshcatVisualizer

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
    data = np.loadtxt(fname,delimiter=',',ndmin=2)

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
    parser.add_argument("traj_file" , nargs='?', default='traj_003.csv', help='trajectory file name')

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

def makeAngleData(arm, trajectory, t):
    joints = np.zeros((len(trajectory), 3))
    velocity = np.zeros_like(joints)
    acceleration = np.zeros_like(joints)
    prevCord = np.array([0, 0, 0])
    for i, coord in enumerate(trajectory):
        j = arm.getJointPosFromEE(coord, prevCord)
        joints[i, :] = j
        if i > 0:
            dt = (t[i] - t[i-1])
            velocity[i, :] = (coord - prevCord) / dt
            acceleration[i, :] = (velocity[i,:] - velocity[i-1,:]) / dt
        prevCord = coord
    return joints, velocity, acceleration

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
    limb       = simplest_2dof_limb()
    motor_ctrl = simplest_2dof_controller(L1=10.05 , L2=5.05)
    brain      = cerebellum_marr_albus()

    # testing arm with dynamics
    coolarm    = dynamic_3dof_arm("models/arm_3dof.urdf") 
    # this should be ran once for every trajectory because it takes a long time
    # and will be the same every time
    j, v, a = makeAngleData(coolarm, desired_position, t)

    # inverse kinematics - need to introduce error somewhere here
    #for i, (joint, vel, accel) in enumerate(zip(j, v, a)):
    #    torques[i, :] = coolarm.inverse(joint, vel, accel)
    #print(torques)

    # makes the sim in browser. Make sure looking at http://127.0.0.1:7000/static/ NOT http://127.0.0.1:7000
    viz = MeshcatVisualizer(coolarm.model, coolarm.collModel, coolarm.visualModel)
    viz.initViewer(open=False)
    viz.loadViewerModel()
    # loop the sim forever
    while True:
        viz.play(j, 1/60)

    # iterate control / learning algorithm over time
    # for i,waypoint in enumerate(desired_position):

    #     joint_angles   = motor_ctrl.get_joint_angles(waypoint) + correction
    #     limb_location  = limb.move(joint_angles)
    #     movement_error = waypoint - limb_location

    #     brain.update(movement_error)
    #     correction = brain.compute_correction(limb_location)

    #     # store outcomes
    #     actual_limb_location[i,:] = limb_location

    # # plot outcomes
    # #plot_results(desired_position , actual_limb_location)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(t, desired_position[:, 0])
    # ax[0].plot(t, actual_limb_location[:, 0])
    # ax[0].set_title("x position")
    # ax[1].plot(t, desired_position[:, 1])
    # ax[1].plot(t, actual_limb_location[:, 1])
    # ax[1].set_title("y position")
    # plt.show()

    # to keep visualizer running
    # there are better ways to do this

main()
