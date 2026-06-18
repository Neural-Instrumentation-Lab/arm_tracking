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
import sys
from exp_02_09_atrack_assets import simplest_2dof_limb, simplest_2dof_controller, cerebellum_marr_albus, dynamic_3dof_arm, dynamic_2dof_arm, armTraj, angle_diff
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
    parser.add_argument("experiment" , nargs='?', default='2', help='select experiment. See experiment.txt')

    # parse args and extract filename
    args  = parser.parse_args()
    exp = args.experiment
    try:
        exp = int(exp)
    except:
        print("invalid argument")
        sys.exit()
    if exp > 9 or exp < 2:
        print("experiment must be between 2-9")
        sys.exit()

    return exp

###################################
def plot_results(traj_no_error, traj_w_error, final_traj, time, errTorqNoBrain, errTorqBrain, errDistNoBrain, errDistBrain, nDof):
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
    
    if nDof == 3:
        fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    else:
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    pltN = 0
    axs[pltN].plot(time, traj_no_error.torq[:,0], linewidth=2)
    axs[pltN].plot(time, traj_w_error.torq[:,0], linewidth=2)
    axs[pltN].plot(time, final_traj.torq[:,0], linewidth=2)
    axs[pltN].set_title("Joint 1 Torque")
    axs[pltN].set_ylabel("Torque")
    axs[pltN].legend(["Ideal", "No Brain", "Brain"])
    axs[pltN].grid(True)

    pltN += 1
    axs[pltN].plot(time, traj_no_error.torq[:, 1], linewidth=2)
    axs[pltN].plot(time, traj_w_error.torq[:, 1], linewidth=2)
    axs[pltN].plot(time, final_traj.torq[:, 1], linewidth=2)
    axs[pltN].set_title("Joint 2 Torque")
    axs[pltN].set_ylabel("Torque")
    axs[pltN].legend(["Ideal", "No Brain", "Brain"])
    axs[pltN].grid(True)

    if nDof == 3:
        pltN += 1
        axs[pltN].plot(time, traj_no_error.torq[:, 2], linewidth=2)
        axs[pltN].plot(time, traj_w_error.torq[:, 2], linewidth=2)
        axs[pltN].plot(time, final_traj.torq[:, 2], linewidth=2)
        axs[pltN].set_title("Joint 3 Torque")
        axs[pltN].set_ylabel("Torque")
        axs[pltN].legend(["Ideal", "No Brain", "Brain"])
        axs[pltN].grid(True)

    pltN += 1
    axs[pltN].plot(time, errTorqNoBrain, linewidth=2)
    axs[pltN].plot(time, errTorqBrain, linewidth=2)
    axs[pltN].set_title("Torque Error")
    axs[pltN].set_ylabel("||τ_des - τ_actual||")
    axs[pltN].legend(["No Brain Error", "Brain Error"])
    axs[pltN].grid(True)

    pltN += 1
    axs[pltN].plot(time, errDistNoBrain, linewidth=2)
    axs[pltN].plot(time, errDistBrain, linewidth=2)
    axs[pltN].set_title("Distance Error")
    axs[pltN].set_xlabel("Time")
    axs[pltN].set_ylabel("ee distance error")
    axs[pltN].legend(["No Brain Error", "Brain Error"])
    axs[pltN].grid(True)
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
    traj_files      = {2 : "traj_004.csv",
                        3 : "traj_004.csv",
                        4 : "traj_004.csv",
                        5 : "traj_005.csv",
                        6 : "traj_003.csv",
                        7 : "traj_004.csv",
                        8 : "traj_004.csv",
                        9 : "traj_004.csv"}
    arm_files       = {2 : "arm_2dof.urdf",
                        3 : "arm_2dof.urdf",
                        4 : "arm_3dof.urdf",
                        5 : "arm_2dof.urdf",
                        6 : "arm_3dof.urdf",
                        7 : "arm_2dof.urdf",
                        8 : "arm_2dof.urdf",
                        9 : "arm_3dof.urdf"}
    illusion_files  = {2 : "arm_2dofBigIllusion.urdf",
                        3 : "arm_2dofIllusoryLengths.urdf",
                        4 : "arm_3dofIllusion.urdf",
                        5 : "arm_2dofBigIllusion.urdf",
                        6 : "arm_3dofIllusion.urdf",
                        7 : "arm_2dofIllusoryMass.urdf",
                        8 : "arm_2dofBigIllusoryMass.urdf",
                        9 : "arm_3dofIllusoryMass.urdf"}
    dof = {2 : 2,
           3: 2,
           4: 3,
           5: 2,
           6 : 3,
           7 : 2,
           8 : 2,
           9 : 3}

    experiment = parse_args()

    fname      = traj_files[experiment]
    armFile    = arm_files[experiment]
    illFile    = illusion_files[experiment]
    n_dof      = dof[experiment]

    # load end effector trajectory
    desired_ee_pos, time, n_dim = get_trajectory(fname)   

    # instantiate limb, motor control unit, brain    
    if n_dof == 2:
        arm         = dynamic_2dof_arm("models/"+armFile) 
        illusoryArm = dynamic_2dof_arm("models/"+illFile) 
    else:
        arm         = dynamic_3dof_arm("models/"+armFile) 
        illusoryArm = dynamic_3dof_arm("models/"+illFile) 

    brain           = cerebellum_marr_albus(n_dims=arm.njoints)

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

    kp = np.ones(arm.njoints)*20 
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
        if n_dof == 3:
            brain.update(eePosErr, eeVelErr)
        else:
            brain.update(np.array([eePosErr[0], eePosErr[2]]), np.array([eeVelErr[0], eeVelErr[2]]))

        # store results
        final_traj.pos[i,:]   = currPos
        final_traj.vel[i,:]   = currVel
        final_traj.acel[i,:]  = currAcc
        final_traj.eePos[i,:] = arm.getPos()

    # forward dynamics on the no-brain case for a control 
    cntrl_traj = arm.forwardDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.torq, 
                                     desired_ee_pos[0], time)


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
                 errDistNoBrain, errDistBrain, n_dof)

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