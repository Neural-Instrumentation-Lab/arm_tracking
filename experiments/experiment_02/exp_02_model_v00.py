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
from exp_02_atrack_assets import simplest_2dof_limb, simplest_2dof_controller, cerebellum_marr_albus, dynamic_3dof_arm, dynamic_2dof_arm, armTraj, angle_diff
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
    parser.add_argument("illusory_arm_file" , nargs='?', default='arm_2dofBigIllusion.urdf', help='arm file with wrong lengths name')

    # parse args and extract filename
    args  = parser.parse_args()
    fname = args.traj_file
    armFile = args.illusory_arm_file

    return fname, armFile

###################################
def plot_results(traj_no_error, traj_w_error, final_traj, time, errTorqNoBrain, errTorqBrain, errDistNoBrain, errDistBrain):
###################################    
    '''
    plots desired and actual end effector positions

    Args:
        traj_no_error:          trajectory that end effector was trying to achieve
        traj_w_error:           trajectory that end effector would have achieved without brain 
        final_traj:             trajectory that end effector actually reached
        time:                   time vector
        errTorqNoBrain:         torque errors without brain
        errTorqBrain:           torque errors with brain
        errDistNoBrain:         distance errors without brain
        errDistBrain:           distance errors with brain
    '''
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(time, traj_no_error.torq[:,0], linewidth=2)
    axs[0].plot(time, traj_w_error.torq[:,0], linewidth=2)
    axs[0].plot(time, final_traj.torq[:,0], linewidth=2)
    axs[0].set_title("Joint 1 Torque")
    axs[0].set_ylabel("Torque")
    axs[0].legend(["Ideal", "No Brain", "Brain"])
    axs[0].grid(True)

    axs[1].plot(time, traj_no_error.torq[:, 1], linewidth=2)
    axs[1].plot(time, traj_w_error.torq[:, 1], linewidth=2)
    axs[1].plot(time, final_traj.torq[:, 1], linewidth=2)
    axs[1].set_title("Joint 2 Torque")
    axs[1].set_ylabel("Torque")
    axs[1].legend(["Ideal", "No Brain", "Brain"])
    axs[1].grid(True)

    axs[2].plot(time, errTorqNoBrain, linewidth=2)
    axs[2].plot(time, errTorqBrain, linewidth=2)
    axs[2].set_title("Torque Error")
    axs[2].set_ylabel("||τ_des - τ_actual||")
    axs[2].legend(["No Brain Error", "Brain Error"])
    axs[2].grid(True)

    axs[3].plot(time, errDistNoBrain, linewidth=2)
    axs[3].plot(time, errDistBrain, linewidth=2)
    axs[3].set_title("Distance Error")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("ee distance error")
    axs[3].legend(["No Brain Error", "Brain Error"])
    axs[3].grid(True)
    plt.show()


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
    for i, coord in enumerate(trajectory):
        if i > 0:
            j = arm.getJointPosFromEE(coord, joints[i-1,:])
        else:
            j = arm.getJointPosFromEE(coord)
        joints[i, :] = j
        if i > 0:
            dt = (t[i] - t[i-1])
            velocity[i, :] = angle_diff(j, joints[i-1,:]) / dt
            acceleration[i, :] = (velocity[i,:] - velocity[i-1,:]) / dt

    armData = armTraj(joints, velocity, acceleration)
    return armData 

def initViz(arm):
    '''
    creates a video playing through meshcat in your browser

    Args:
        arm: which arm following to use
        traj: trajectory to follow. Only needs to have the traj.pos to be filled
        timestep: how long each position frame lasts in the viewer
    '''
    viz = MeshcatVisualizer(arm.model, arm.collModel, arm.visualModel)
    viz.initViewer(open=False)
    viz.loadViewerModel()
    return viz

def makeEEData(pos, t):

    velocity = np.zeros_like(pos)
    acceleration = np.zeros_like(pos)
    for i, coord in enumerate(pos):
        if i > 0:
            dt = (t[i] - t[i-1])
            velocity[i, :] = (coord - pos[i-1,:]) / dt
            acceleration[i, :] = (velocity[i,:] - velocity[i-1,:]) / dt

    armData = armTraj(pos, velocity, acceleration)
    return armData 


###################################
def main():
###################################

    # load user preferences from command line
    fname, armFile = parse_args()

    # load end effector trajectory
    desired_ee_pos, time, n_dim = get_trajectory(fname)   

    # instantiate limb, motor control unit, brain    
    arm         = dynamic_2dof_arm("models/arm_2dof.urdf") 
    illusoryArm = dynamic_2dof_arm("models/"+armFile) 
    brain       = cerebellum_marr_albus()

    # turn gravity off
    #arm.model.gravity = pin.Motion.Zero()
    #illusoryArm.model.gravity = pin.Motion.Zero()

    # IK computing the necessary torques along a trajectory
    # PD controller because otherwise it will diverge
    # Computed for an arm with different lengths to introduce error
    traj_w_error      = makeJointData(illusoryArm, desired_ee_pos, time)
    traj_w_error.torq, traj_w_error.eePos = illusoryArm.inverseDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.acel, time)

    # FD applying those computed torques to the actual arm
    # with control loop from the cerebellum
    currPos       = arm.getJointPosFromEE(desired_ee_pos[0]) 
    currVel       = np.zeros_like(currPos)
    currAcc       = np.zeros_like(currPos)
    final_traj    = armTraj(pos=np.zeros_like(traj_w_error.pos), 
                            torq=np.zeros_like(traj_w_error.torq),
                            eePos=np.zeros_like(desired_ee_pos),
                            vel = np.zeros_like(traj_w_error.vel),
                            accel=np.zeros_like(traj_w_error.acel))
    desired_ee_traj = makeEEData(desired_ee_pos, time)
    eeVel = np.zeros_like(desired_ee_pos[0])
    corrTorque = np.zeros_like(currPos)
    torrPd = np.zeros_like(currPos)

    kp = np.array([20, 20])
    kd = 2*np.sqrt(kp)
    for i, torque in enumerate(traj_w_error.torq):
        corrTorque = brain.compute_correction(currPos, currVel)
        torrPd = kp*(traj_w_error.pos[i] - currPos) + kd*(traj_w_error.vel[i] - currVel)

        # move the arm
        if i > 0:
            dt      = time[i] - time[i-1]
            currVel = (currVel + currAcc *dt)
            currPos = pin.integrate(arm.model, currPos, currVel * dt)
        currAcc = arm.forward(currPos, currVel, torque + corrTorque + torrPd)
        arm.move(currPos, currVel, currAcc)
        final_traj.torq[i] = torque + corrTorque + torrPd # stores for plot

        # calculate end-effector velocity 
        pin.computeJointJacobians(arm.model, arm.data, currPos)
        J = pin.getFrameJacobian(arm.model, arm.data, arm.eeId, pin.LOCAL_WORLD_ALIGNED)
        eeVel = (J[:3,:]).dot(currVel)

        # PD control

        # compute error 
        eePosErr = arm.getPos() - desired_ee_traj.pos[i]
        eeVelErr = eeVel - desired_ee_traj.vel[i]

        # brain 
        brain.update(np.array([eePosErr[0], eePosErr[2]]), np.array([eeVelErr[0], eeVelErr[2]]))

        # store results
        final_traj.pos[i,:]   = currPos
        final_traj.vel[i,:]   = currVel
        final_traj.acel[i,:]  = currAcc
        final_traj.eePos[i,:] = arm.getPos()

    # forward dynamics on the no-brain case for a control 
    cntrl_traj = arm.forwardDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.torq, 
                                     desired_ee_pos[0], time)

    print((cntrl_traj.torq == final_traj.torq).all())

    # calculating error
    traj_no_error      = makeJointData(arm, desired_ee_pos, time)
    traj_no_error.torq, _ = arm.inverseDynamics(traj_no_error.pos, traj_no_error.vel, traj_no_error.acel, time)
    errTorqNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  cntrl_traj.torq)]
    errTorqBrain       = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  final_traj.torq)]
    errDistBrain       = [np.linalg.norm(des - act) for (des,act) in zip(desired_ee_traj.pos,  final_traj.eePos)]
    errDistNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(desired_ee_traj.pos,  cntrl_traj.eePos)]
    print(f"total distance error: {np.sum(errDistBrain)}")

    # plotting 
    plot_results(traj_no_error, cntrl_traj, final_traj, time, errTorqNoBrain, errTorqBrain, 
                 errDistNoBrain, errDistBrain)

    # makes the sim in browser. Make sure looking at http://127.0.0.1:7000/static/ NOT http://127.0.0.1:7000
    viz = initViz(arm)
    while True:
        trajectories = {1 : traj_no_error,
                        2 : traj_w_error,
                        3 : final_traj}
        userTraj = input("select the trajectory to watch:\n1: desired trajectory"
                         "\n2: trajectory with no brain\n3: trajectory with brain\nq: exit\n")
        if userTraj == "q":
            break

        try:
            userTraj = int(userTraj)
        except:
            print("not valid input")
            continue
        
        if userTraj > 3 or userTraj < 1:
            print("not valid input")
            continue

        viz.play(trajectories[userTraj].pos, 1/100)

main()