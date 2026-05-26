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
from atrack_assets import simplest_2dof_limb, simplest_2dof_controller, cerebellum_marr_albus, dynamic_3dof_arm, dynamic_2dof_arm
import pinocchio as pin
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
    parser.add_argument("traj_file" , nargs='?', default='traj_004.csv', help='trajectory file name')

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

def angle_diff(a, b):
    """
    Smallest signed difference between angles a and b.
    Result is in [-pi, pi].
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def makeJointData(arm, trajectory, t):
    '''
    constructs joint positions, velocities, and accelerations
    for a given trajectory of the end-effector.

    Args:
        arm: arm class.
        trajectory: positions (x,y,z) in 3d space of the end effector.
        t: time vector corresponding to the trajectory.
    Returns:
        joints: joint positions 
        velocity: joint angular velocities
        acceleration: joint angular accelerations 
    '''
    joints = np.zeros((len(trajectory), arm.njoints))
    velocity = np.zeros_like(joints)
    acceleration = np.zeros_like(joints)
    prevCord = np.array([0, 0, 0])
    for i, coord in enumerate(trajectory):
        j = arm.getJointPosFromEE(coord)
        joints[i, :] = j
        if i > 0:
            dt = (t[i] - t[i-1])
            velocity[i, :] = angle_diff(j, joints[i-1,:]) / dt
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

    # instantiate limb, motor control unit, brain    
    coolarm     = dynamic_2dof_arm("models/arm_2dof.urdf") 
    illusoryArm = dynamic_2dof_arm("models/arm_2dofIllusoryLengths.urdf", lengths=[.31, .401, .391]) 
    # turn gravity off
    #coolarm.model.gravity = pin.Motion.Zero()
    #illusoryArm.model.gravity = pin.Motion.Zero()
    brain       = cerebellum_marr_albus()

    # IK computing the necessary torques along a trajectory
    # PD controller because otherwise it will diverge
    # Computed for an arm with different lengths to introduce error
    positions, velocities, accels = makeJointData(illusoryArm, desired_position, t)
    torques   = np.zeros_like(positions)
    torquesPD = np.zeros_like(torques)
    kp        = np.array([200, 200]) # found empirically, these seem ok
    kd        = 2*np.sqrt(kp) # this is the best the ratio for a reason
    pos       = positions[0]  
    vel       = np.zeros_like(pos)
    acc       = np.zeros_like(pos)
    for i, (posCorr, velCorr, accCorr) in enumerate(zip(positions, velocities, accels)):
        torques[i, :]   = illusoryArm.inverse(posCorr, velCorr, accCorr) # ideal torque
        torquesPD[i, :] = torques[i, :] + kp*(posCorr - pos) + kd*(velCorr - vel)
        acc             = illusoryArm.forward(pos, vel, torquesPD[i, :])
        if i > 0:
            dt  = t[i] - t[i-1]
            vel = vel + acc * dt 
            pos = pin.integrate(illusoryArm.model, pos, vel*dt)

    # FK applying those computed torques to the actual arm
    # with control loop from the cerebellum
    correction = np.zeros(2)
    pos        = coolarm.getJointPosFromEE(desired_position[0]) 
    vel        = np.zeros_like(pos)
    acc        = np.zeros_like(pos)
    actualPos  = np.zeros_like(positions)
    for i, torque in enumerate(torquesPD):
        corrTorque = torque + correction
        acc = coolarm.forward(pos, vel, corrTorque)
        # move the arm
        if i > 0:
            dt = t[i] - t[i-1]
            vel = vel + acc * dt 
            pos = pin.integrate(coolarm.model, pos, vel*dt)
        coolarm.move(pos, vel, acc)

        # calculate error & correction
        eePos = coolarm.getPos()
        error = desired_position[i] - eePos 

        brain.update([error[0], error[2]])
        correction = brain.compute_correction([eePos[0], eePos[2]])

        actualPos[i,:] = pos
    # makes the sim in browser. Make sure looking at http://127.0.0.1:7000/static/ NOT http://127.0.0.1:7000
    viz = MeshcatVisualizer(coolarm.model, coolarm.collModel, coolarm.visualModel)
    viz.initViewer(open=False)
    viz.loadViewerModel()
    while True:
        viz.play(actualPos, 1/10000)

main()