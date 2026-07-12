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
from experiment_assets import Experiment, Trajectory, load_trajectory

matplotlib.use('TkAgg')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S"
)

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
        axs[pltN].set_title("Joint 3 Corrective Torque")
        axs[pltN].set_ylabel("Torque")
        axs[pltN].legend(["Ideal", "Brain"])
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

def plot_brain_results(errorTot, errDistNoBrain, PCact, time, mf_dcn, pc_dcn):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    pltN = 0
    axs[pltN, 0].plot(errorTot/len(time))
    axs[pltN, 0].set_xlabel("Trial")
    axs[pltN, 0].set_ylabel("Mean Absolute Error")
    axs[pltN, 0].set_title("1.5kg Mass")
    axs[pltN, 0].axhline(np.sum(errDistNoBrain)/len(time), color='r', linestyle='--', linewidth=2)
    axs[pltN, 0].legend(["Brain Erorr", "No Brain Error"])

    pltN += 1
    axs[pltN, 0].plot(time, PCact[:, 1:3])
    axs[pltN, 0].set_xlabel("Time (s)")
    axs[pltN, 0].set_ylabel("PC Activation")
    axs[pltN, 0].set_title("Purkinje Cell Activation (Joint 2)")
    axs[pltN, 0].legend(["Joint 2 Agonist", "Joint 2 Antagonist"])

    pltN = 0
    axs[pltN, 1].plot(mf_dcn[:, 3])
    axs[pltN, 1].set_xlabel("Trial")
    axs[pltN, 1].set_ylabel("Weight")
    axs[pltN, 1].set_title("MF_DCN Weight Joint 2 Antagonist")

    pltN += 1
    axs[pltN, 1].plot(pc_dcn[:, 3])
    axs[pltN, 1].set_xlabel("Trial")
    axs[pltN, 1].set_ylabel("Weight")
    axs[pltN, 1].set_title("PC_DCN Weight Joint 2 Antagonist")
    plt.show()

def getDerivatives(t, x, dx=None):
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
    if dx is None:
        dx = np.gradient(x, t, axis=0)
    ddx = np.gradient(dx, t, axis=0)
    return dx, ddx 

def makeJointData(arm, positions, velocities, accelerations):
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
    joint_pos = np.zeros((len(positions), arm.njoints))
    joint_vels = np.zeros_like(joint_pos)
    joint_accs = np.zeros_like(joint_pos)
    q   = np.zeros(arm.njoints)
    qd  = np.zeros(arm.njoints)
    qdd = np.zeros(arm.njoints)
    for i, coord in enumerate(positions):
        if i > 0:
            q = arm.getJointPosFromEE(coord, joint_pos[i-1,:])
        else:
            q = arm.getJointPosFromEE(coord)
        joint_pos[i, :] = q
        pin.forwardKinematics(arm.model, arm.data, q, qd)
        pin.computeJointJacobians(arm.model, arm.data, q)
        pin.computeJointJacobiansTimeVariation(arm.model, arm.data, q, qd)
        J    = pin.getFrameJacobian(arm.model, arm.data, arm.eeId, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jdot = pin.getFrameJacobianTimeVariation(arm.model, arm.data, arm.eeId, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        qd = np.linalg.pinv(J) @ velocities[i, :] 
        qdd = np.linalg.pinv(J) @ (accelerations[i,:] - Jdot @ velocities[i,:])
        joint_vels[i, :] = qd
        joint_accs[i, :] = qdd
    return joint_pos, joint_vels, joint_accs 

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

def save_trajectories(trajectories, arm_ids, dt, filename="path_exp.pkl"):
    '''
    saves the travelled trajectories for further viewing
    without having to rerun the simulation
    trajectories can be played by adding the --view argument to runExperiment.py 
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

def runSimulation(arm, traj_w_error, desired_ee_traj, time, brain, nTrials):
    '''
    Controls the arm with the cerebellar feedback loop
    Args:
        arm: arm to move along the trajectory
        traj_w_error: trajectory to move along in joint space (pos, vel, and torques)
        desired_ee_traj: trajectory of EE in cartesian coords (pos)
        time: time vector that corresponds to both ee and error trajectories
        brain: brain to compute correction
        nTrials: Number of times to repeat the trajectory
    Returns:
        final_traj: Trajectory object containing results from the final trajectory
        errorTot: joint position MAE over nTrials
        PCact: evolution of Purkinje Cells over nTrials
        mf_dcn: evolution of mf-dcn weights over nTrials
        pc_dcn: evolution of pc-dcn weights over nTrials

    '''
    # initialize the trajectory
    currPos       = traj_w_error.pos[0,:] 
    currVel       = traj_w_error.vel[0,:] 
    currAcc       = traj_w_error.acel[0,:] 
    arm.move(currPos, currVel, currAcc)
    final_traj    = armTraj(pos=np.zeros_like(traj_w_error.pos), 
                            torq=np.zeros_like(traj_w_error.torq),
                            eePos=np.zeros_like(desired_ee_traj.pos),
                            vel = np.zeros_like(traj_w_error.vel),
                            accel=np.zeros_like(traj_w_error.acel))
    corrTorque = np.zeros_like(currPos)
    torrPd = np.zeros_like(currPos)
    errorTot = np.zeros(nTrials)
    mf_dcn = np.zeros((nTrials, arm.njoints*2))
    pc_dcn = np.zeros_like(mf_dcn)
    PCact = np.zeros((len(time), arm.njoints*2))
    state = 0
    step = np.floor(len(time) / brain.nPFs)
    # PD Constants
    kp = np.ones(arm.njoints)*20
    kd = 2*np.sqrt(kp)
    for trial in range(nTrials):
        for i, torque in enumerate(traj_w_error.torq):
            # compute error 
            qError  = angle_diff(traj_w_error.pos[i], currPos)
            qdError = traj_w_error.vel[i] - currVel
            # compute corrections, but brain only fires once for every PF
            if i % step ==0:
                corrTorque = brain.compute(qError, qdError, state)
                state += 1
            torrPd = kp*(qError) + kd*(qdError)
            # move the arm
            currAcc = arm.forward(currPos, currVel, torque + corrTorque + torrPd)
            if i > 0:
                dt      = time[i] - time[i-1]
                currVel = (currVel + currAcc *dt)
                currPos = pin.integrate(arm.model, currPos, currVel * dt)
            arm.move(currPos, currVel, currAcc)
            # store results for plotting
            final_traj.pos[i,:]   = currPos
            if trial == nTrials-1:
                final_traj.torq[i] = torque + corrTorque + torrPd
                final_traj.eePos[i,:] = arm.getPos()
                final_traj.vel[i,:]   = currVel
                final_traj.acel[i,:]  = currAcc
                PCact[i,:] = brain.getPC()
        # reset between each trial
        currPos          = traj_w_error.pos[0,:] 
        currVel          = traj_w_error.vel[0,:] 
        currAcc          = traj_w_error.acel[0,:] 
        arm.move(currPos, currVel, currAcc)
        # store for plotting
        errorTot[trial]  = np.sum([np.linalg.norm(des - act) for (des,act) in zip(traj_w_error.pos,  final_traj.pos)])
        mf_dcn[trial, :] = brain.getMF_DCN()
        pc_dcn[trial, :] = brain.getPC_DCN()
    return final_traj, errorTot, PCact, mf_dcn, pc_dcn

def fillTrajectory(traj, arm):
    """
    computes any missing derivates through getDerivatives()
    and computes missing joint positions thru IK makeJointData()
    """
    time = traj.time
    if traj.velocity is None or traj.acceleration is None:
        traj.velocity, traj.acceleration = getDerivatives(time, traj.position, dx=traj.velocity)
    if traj.joint_position is None:
        traj.joint_position, traj.joint_velocity, traj.joint_acceleration = makeJointData(arm, traj.position, traj.velocity, traj.acceleration)
    elif traj.joint_velocity is None or traj.joint_acceleration is None:
        traj.joint_velocity, traj.joint_acceleration = getDerivatives(time, traj.joint_position, dx=traj.joint_velocity)
    return traj

###################################
def simulate(exp, save, showOutput, grav):
###################################
    trajFile   = exp.trajectory 
    armFile    = exp.actualArm
    illFile    = exp.illusoryArm 
    n_dof      = exp.nDof 

    # instantiate limb, motor control unit, brain    
    if n_dof == 2:
        arm         = dynamic_2dof_arm(armFile, disp=showOutput) 
        illusoryArm = dynamic_2dof_arm(illFile, disp=showOutput) 
    elif n_dof == 3:
        arm         = dynamic_3dof_arm(armFile, disp=showOutput) 
        illusoryArm = dynamic_3dof_arm(illFile, disp=showOutput) 
    brain           = cerebellum(n_dof=arm.njoints)

    # load desired trajectory
    # for smoothest trajectories, the trajectory should have analytically determined 
    # cartesian pos, vel, accel and joint-space q, qd, qdd. If not they will be calculated, 
    # but this introduces noise in the double-differentation 
    traj_data = load_trajectory(trajFile)   
    traj_data = fillTrajectory(traj_data, arm)
    # splitting into desired joint (traj_w_error) and cartesian (desired_ee_traj) data
    time                = traj_data.time
    desired_ee_traj     = armTraj(pos=traj_data.position,       vel=traj_data.velocity,       accel=traj_data.acceleration)
    traj_w_error        = armTraj(pos=traj_data.joint_position, vel=traj_data.joint_velocity, accel=traj_data.joint_acceleration)
    traj_no_error       = armTraj(pos=traj_data.joint_position, vel=traj_data.joint_velocity, accel=traj_data.joint_acceleration)
    traj_no_error.eePos = desired_ee_traj.pos

    # turn gravity off
    if grav == False:
        arm.model.gravity         = pin.Motion.Zero()
        illusoryArm.model.gravity = pin.Motion.Zero()

    # Inverse Dynamics: computing the torques along a trajectory (with error) and recording where those torques make you end up
    traj_w_error.torq, traj_w_error.eePos = illusoryArm.inverseDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.acel, time)
    traj_no_error.torq, _                 = arm.inverseDynamics(traj_no_error.pos, traj_no_error.vel, traj_no_error.acel, time)

    # Forward Dynamics: applying those computed torques to the actual arm
    # with a control feedback from the cerebellar model 
    final_traj, errorTot, pc, mfdcn, pcdcn = runSimulation(arm, traj_w_error, desired_ee_traj, time, brain, nTrials=1500)
    # no-brain case for a control 
    cntrl_traj = arm.forwardDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.torq, time)
    
    # saving trajectories
    trajectories = {1: traj_no_error, 2: cntrl_traj, 3: final_traj, 4: traj_w_error}
    arm_ids = {1: armFile, 2: armFile, 3: armFile, 4: illFile}
    if save:
       save_trajectories(trajectories, arm_ids, time[1] - time[0], filename=exp.results)

    if not showOutput:
        print("Simulation Complete...")
        return

    # calculating error for plotting
    errTorqNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  cntrl_traj.torq)]
    errTorqBrain       = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  final_traj.torq)]
    errJointNoBrain    = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.pos,  cntrl_traj.pos)]
    errDistBrain       = [np.linalg.norm(des - act) for (des,act) in zip(desired_ee_traj.pos,  final_traj.eePos)]
    errDistNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(desired_ee_traj.pos,  cntrl_traj.eePos)]
    print(f"average distance error: {np.sum(errDistBrain)/len(time)}")

    # plotting arm data
    plot_results(traj_no_error, traj_w_error, final_traj, time, errTorqNoBrain, errTorqBrain, 
                 errDistNoBrain, errDistBrain, n_dof)
    # plotting brain data
    plot_brain_results(errorTot, errJointNoBrain, pc, time, mfdcn, pcdcn)

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