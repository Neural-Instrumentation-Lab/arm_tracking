'''
MODEL V01

Implementation of cerebellar error correction
with dynamic 3dof arm, the cerebellar model is 
from the paper "Distributed cerebellar plasticity implements adaptable gain control in a 
manipulation task: a closed-loop robotic simulation" by Garrido et All

@Author: James Soley
@Date: Summer 2026
'''

## IMPORTS #####################################################################
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from arm_assets_v01 import simplest_2dof_limb, simplest_2dof_controller, dynamic_3dof_arm, dynamic_2dof_arm, armTraj, angle_diff
from garrido_brain_v00 import cerebellum
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import pickle

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

    return traj,t,n_dim

###################################
def parse_args():
###################################
    '''
    Parses commandline arguments

    Collects experiment number from command line 
    and whether the user wants to save the traj data
    CURRENTLY UNUSED

    Returns:
        experiment (int)
        save (bool)
    '''
    # set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment" , nargs='?', default='2', help='select experiment. See experiment.txt')
    parser.add_argument("--save", help='saves the outputted trajectories.', action='store_true')

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

    return exp, args.save

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
    
    # choose to plot the 3rd joint plot based on ndof
    if nDof == 3:
        fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    else:
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # plot joint 1 torque
    pltN = 0
    axs[pltN].plot(time, traj_no_error.torq[:,0] - traj_w_error.torq[:,0], linewidth=2, linestyle='--')
    axs[pltN].plot(time, final_traj.torq[:,0] - traj_w_error.torq[:,0], linewidth=2)
    axs[pltN].set_title("Joint 1 Corrective Torque")
    axs[pltN].set_ylabel("Torque")
    axs[pltN].legend(["Ideal", "Brain"])
    axs[pltN].grid(True)

    # plot joint 2 torque
    pltN += 1
    axs[pltN].plot(time, traj_no_error.torq[:,1] - traj_w_error.torq[:,1], linewidth=2, linestyle='--')
    axs[pltN].plot(time, final_traj.torq[:,1] - traj_w_error.torq[:,1], linewidth=2)
    axs[pltN].set_title("Joint 2 Corrective Torque")
    axs[pltN].set_ylabel("Torque")
    axs[pltN].legend(["Ideal", "Brain"])
    axs[pltN].grid(True)

    # plot joint 3 torque
    if nDof == 3:
        pltN += 1
        axs[pltN].plot(time, traj_no_error.torq[:,2] - traj_w_error.torq[:,2], linewidth=2, linestyle='--')
        axs[pltN].plot(time, final_traj.torq[:,2] - traj_w_error.torq[:,2], linewidth=2)
        axs[pltN].set_title("Joint 3 Torque")
        axs[pltN].set_ylabel("Torque")
        axs[pltN].legend(["Ideal", "No Brain", "Brain"])
        axs[pltN].grid(True)

    # plot total absolute mean torque error
    pltN += 1
    axs[pltN].plot(time, errTorqNoBrain, linewidth=2)
    axs[pltN].plot(time, errTorqBrain, linewidth=2)
    axs[pltN].set_title("Torque Error")
    axs[pltN].set_ylabel("||τ_des - τ_actual||")
    axs[pltN].legend(["No Brain Error", "Brain Error"])
    axs[pltN].grid(True)

    # plot total absolute mean distance error
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
    calculates angular vel and acceleration thru joint Jacobian
    instead of derivative to reduce error thru derivation

    Args:
        arm: arm class.
        trajectory: positions (x,y,z) in 3d space of the end effector.
        t: time vector corresponding to the trajectory.
    Returns:
        joints: joint positions 
        velocity: joint angular velocities
        acceleration: joint angular accelerations 
    '''
    joints = np.zeros((len(trajectory.pos), arm.njoints))
    vels = np.zeros_like(joints)
    accs = np.zeros_like(joints)
    v = np.zeros(3)
    a = np.zeros(3)
    for i, coord in enumerate(trajectory.pos):
        if i > 0:
            j = arm.getJointPosFromEE(coord, joints[i-1,:])
        else:
            j = arm.getJointPosFromEE(coord)
        joints[i, :] = j

        pin.forwardKinematics(arm.model, arm.data, j, v)
        pin.computeJointJacobians(arm.model, arm.data, j)
        pin.computeJointJacobiansTimeVariation(arm.model, arm.data, j, v)
        J    = pin.getFrameJacobian(arm.model, arm.data, arm.eeId, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jdot = pin.getFrameJacobianTimeVariation(arm.model, arm.data, arm.eeId, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        v = np.linalg.pinv(J) @ trajectory.vel[i, :] 
        a = np.linalg.pinv(J) @ (trajectory.acel[i,:] - Jdot @ trajectory.vel[i,:])
        vels[i, :] = v
        accs[i, :] = a

    #velocity = np.gradient(np.unwrap(joints, axis=0), t, axis=0)
    #acceleration = np.gradient(velocity, t, axis=0)

    armData = armTraj(joints, vels, accs)
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
    '''
    creates the velocities and accelerations of 
    the end-effector given positions (cartesian)

    Args:
        pos: end-effector coordinates
        t: corresponding time vector
    Returns:
        armData: trajectory object containing the end effector
                 data
    '''
    velocity = np.gradient(pos, t, axis=0)
    acceleration = np.gradient(velocity, t, axis=0)

    armData = armTraj(pos, velocity, acceleration)
    return armData 

def save_trajectories(trajectories, arm_ids, dt, filename="path_exp.pkl"):
    '''
    saves the travelled trajectories for further viewing
    without having to rerun the simulation
    trajectories can be played with the experiments/play_arm_path.py
    file.
    
    Args:
        trajectories: dictionary corresponding the traj id to the actual trajectory
        arm_ids: dictionary corresponding the traj id to the arm file name
        dt: timestep (constant throughout the playback)
        filename: output filename
    '''
    data = {
        key: {"arm": arm_ids[key], "pos": traj.pos, "dt":dt}
        for key, traj in trajectories.items()
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def runSimulation(arm, traj_w_error, desired_ee_traj, time, brain):
    '''
    Controls the arm with the cerebellar feedback loop

    Args:
    Returns:
        final_traj: Trajectory object containing results from the final trajectory
    '''
    # arm starts with no velocity or acceleration and the first position along the traj
    currPos       = traj_w_error.pos[0,:] 
    currVel       = np.zeros_like(currPos)
    currAcc       = np.zeros_like(currPos)
    arm.move(currPos, currVel, currAcc)
    final_traj    = armTraj(pos=np.zeros_like(traj_w_error.pos), 
                            torq=np.zeros_like(traj_w_error.torq),
                            eePos=np.zeros_like(desired_ee_traj.pos),
                            vel = np.zeros_like(traj_w_error.vel),
                            accel=np.zeros_like(traj_w_error.acel))
    corrTorque = np.zeros_like(currPos)
    torrPd = np.zeros_like(currPos)
    nTrials = 1500
    errorTot = np.zeros(nTrials)
    # PD Constants
    kp = np.ones(arm.njoints)*20
    kd = 2*np.sqrt(kp)
    for trial in range(nTrials):
        for i, torque in enumerate(traj_w_error.torq):
            # compute error 
            qError  = angle_diff(traj_w_error.pos[i], currPos)
            qdError = traj_w_error.vel[i] - currVel
            # compute corrections
            corrTorque = brain.compute(qError, qdError)
            torrPd = kp*(qError) + kd*(qdError)

            # move the arm
            if i > 0:
                dt      = time[i] - time[i-1]
                currVel = (currVel + currAcc *dt)
                currPos = pin.integrate(arm.model, currPos, currVel * dt)
            currAcc = arm.forward(currPos, currVel, torque + corrTorque + torrPd)
            arm.move(currPos, currVel, currAcc)
            final_traj.torq[i] = torque + corrTorque + torrPd # stores for plot

            # store results
            final_traj.pos[i,:]   = currPos
            final_traj.vel[i,:]   = currVel
            final_traj.acel[i,:]  = currAcc
            final_traj.eePos[i,:] = arm.getPos()
        # reset between each trial
        errorTot[trial] = np.sum([np.linalg.norm(des - act) for (des,act) in zip(desired_ee_traj.pos,  final_traj.eePos)])
        currPos       = traj_w_error.pos[0,:] 
        currVel       = np.zeros_like(currPos)
        currAcc       = np.zeros_like(currPos)
        arm.move(currPos, currVel, currAcc)
    return final_traj, errorTot


###################################
def main():
###################################
    # load user preferences from command line (UNUSED CURRENTLY)
    experiment, save = parse_args()

    # hardcoded for easy changing CURRENTLY
    fname      = "traj_006.csv" 
    armFile    = "arm_3dofIllusoryMassBigger.urdf" 
    illFile    = "arm_3dofIllusion.urdf" 
    n_dof      = 3 

    # load end effector trajectory
    desired_ee_data, time, n_dim = get_trajectory(fname)   
    desired_ee_traj = armTraj(desired_ee_data[:, 0:3], desired_ee_data[:, 3:6], desired_ee_data[:, 6:9])
    # use if traj doesn't have derivative data
    #desired_ee_traj = makeEEData(desired_ee_data, time)

    # instantiate limb, motor control unit, brain    
    if n_dof == 2:
        arm         = dynamic_2dof_arm("models/"+armFile) 
        illusoryArm = dynamic_2dof_arm("models/"+illFile) 
    else:
        arm         = dynamic_3dof_arm("models/"+armFile) 
        illusoryArm = dynamic_3dof_arm("models/"+illFile) 
    brain           = cerebellum(n_dof=arm.njoints)

    # turn gravity off
    #arm.model.gravity = pin.Motion.Zero()
    #illusoryArm.model.gravity = pin.Motion.Zero()

    # Inverse Dynamics: computing the torques along a trajectory (with error)
    traj_w_error      = makeJointData(illusoryArm, desired_ee_traj, time)
    traj_w_error.torq, traj_w_error.eePos = illusoryArm.inverseDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.acel, time)

    # Forward Dynamics: applying those computed torques to the actual arm
    # with a control feedback from the cerebellar model 
    final_traj, errorTot = runSimulation(arm, traj_w_error, desired_ee_traj, time, brain)

    # forward dynamics on the no-brain case for a control 
    cntrl_traj = arm.forwardDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.torq, 
                                     desired_ee_traj.pos[0], time)

    # calculating error for plotting
    traj_no_error      = makeJointData(arm, desired_ee_traj, time)
    traj_no_error.torq, _ = arm.inverseDynamics(traj_no_error.pos, traj_no_error.vel, traj_no_error.acel, time)
    errTorqNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  cntrl_traj.torq)]
    errTorqBrain       = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  final_traj.torq)]
    errDistBrain       = [np.linalg.norm(des - act) for (des,act) in zip(desired_ee_traj.pos,  final_traj.eePos)]
    errDistNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(desired_ee_traj.pos,  cntrl_traj.eePos)]
    print(f"total distance error: {np.sum(errDistBrain)}")

    # plotting arm data
    plot_results(traj_no_error, cntrl_traj, final_traj, time, errTorqNoBrain, errTorqBrain, 
                 errDistNoBrain, errDistBrain, n_dof)
    # plotting brain data
    plt.plot(errorTot)
    plt.xlabel("Trial")
    plt.ylabel("Total Mean Distance Error")
    plt.title("10kg Mass")
    plt.axhline(np.sum(errDistNoBrain), color='r', linestyle='--', linewidth=2)
    plt.legend(["Brain Erorr", "No Brain Error"])
    plt.show()

    # saving trajectories
    trajectories = {1: traj_no_error, 2: cntrl_traj, 3: final_traj, 4: traj_w_error}
    arm_ids = {1: armFile, 2: armFile, 3: armFile, 4: illFile}  # swap in your actual arm identifiers
    if save:
       save_trajectories(trajectories, arm_ids, time[1] - time[0])

    # makes the sim in browser. Make sure looking at http://127.0.0.1:7000/static/ NOT http://127.0.0.1:7000
    dt = time[1] - time[0]
    prevChoice = 0
    while True:
        userTraj = input("select the trajectory to watch:\n1: desired trajectory"
                          "\n2: trajectory with no brain\n3: trajectory with brain\n"
                          "4: trajectory ID thought it was following\nq: exit\n"
                          "r: play last played traj\n")
        if userTraj == "q":
            break
        if userTraj == "r":
            if prevChoice == 0:
                print("no trajectory to replay")
                continue
            else:
                viz.play(traj.pos, dt)
                continue
        else:
            try:
                userTraj = int(userTraj)
            except ValueError:
                print("not valid input")
                continue
            if userTraj not in trajectories:
                print("not valid input")
                continue
        traj = trajectories[userTraj]
        if prevChoice == 0 or arm_ids[userTraj] != arm_ids[prevChoice]:
            if arm_ids[userTraj] == armFile:
                playArm = arm
            else:
                playArm = illusoryArm
            viz = initViz(playArm)
        viz.play(traj.pos, dt)
        prevChoice = userTraj

main()