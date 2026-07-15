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
from arm_assets_v01 import dynamic_3dof_arm, dynamic_2dof_arm, armTraj, angle_diff
import garrido_brain_cpp as gc 
# from garrido_brain_v00 import cerebellum
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import pickle
from experiment_assets import load_trajectory, load_weights, save_weights, prepare_brain_weights

matplotlib.use('TkAgg')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S"
)

class brainData:
    def __init__(self, nTrials, trajLen, nDof, nPFs):
        self.pc      = np.zeros((nTrials, trajLen, nDof*2)) 
        self.dcn     = np.zeros((nTrials, trajLen, nDof*2)) 
        self.pf_pc   = np.zeros((nTrials, nPFs, nDof*2)) 
        self.mf_dcn  = np.zeros((nTrials, nDof*2)) 
        self.pc_dcn  = np.zeros((nTrials, nDof*2)) 

###################################
def plot_arm_results(traj_no_error, cntrl_traj, traj_w_error, final_traj, time, nDof, saveLoc=None, show=True, save=False):
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
    # calculating errors
    errDistBrain       = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.eePos,  final_traj.eePos)]
    errDistNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.eePos,  cntrl_traj.eePos)]
    errTorqNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  cntrl_traj.torq)]
    errTorqBrain       = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  final_traj.torq)]
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

    saveLoc.parent.mkdir(parents=True, exist_ok=True)
    if save:
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_arm" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()

def plot_brain_results(errorTot, errJointNoBrain, brainResults, time, saveLoc=None, show=True, save=False):
    '''
    Args:
        errorTot: Vector of joint MAE over nTrials
        errJointNoBrain: Joint MAE without a brain
        brainResults: stores the various weights over nTrials and a trajectory
        time: time vector of the trajectory
    '''
    # plotting evolution of MAE over nTrials
    fig, axs = plt.subplots(3, 2, figsize=(12, 12), gridspec_kw={'hspace':0.3})
    nTrials = brainResults.pc.shape[0]
    sampleTrialNums = [1, int(np.ceil(nTrials/15)), int(np.ceil(nTrials/5)), int(np.ceil(nTrials*2/3))]
    pltN = 0
    axs[pltN, 0].plot(errorTot/len(time))
    axs[pltN, 0].set_xlabel("Trial")
    axs[pltN, 0].set_ylabel("Mean Absolute Error")
    axs[pltN, 0].set_title("Evolution of MAE")
    axs[pltN, 0].axhline(np.sum(errJointNoBrain)/len(time), color='r', linestyle='--', linewidth=2)
    axs[pltN, 0].legend(["Brain Error", "No Brain Error"])
    # plotting PC activity over last trial of Joint 2
    pltN += 1
    axs[pltN, 0].plot(time, brainResults.pc[sampleTrialNums[0]-1, :, 2])
    axs[pltN, 0].plot(time, brainResults.pc[sampleTrialNums[1]-1, :, 2])
    axs[pltN, 0].plot(time, brainResults.pc[sampleTrialNums[2]-1, :, 2])
    axs[pltN, 0].plot(time, brainResults.pc[sampleTrialNums[3]-1, :, 2])
    axs[pltN, 0].set_xlabel("Time (s)")
    axs[pltN, 0].set_ylabel("PC Activation")
    axs[pltN, 0].set_title("Purkinje Cell Activation (Joint 2 Agonist)")
    axs[pltN, 0].legend([f"Trial {sampleTrialNums[0]}", f"Trial {sampleTrialNums[1]}", f"Trial {sampleTrialNums[2]}", f"Trial {sampleTrialNums[3]}"])
    # plotting some PF-PC synapses
    pltN += 1
    axs[pltN, 0].plot(brainResults.pf_pc[:, 50, 2]) 
    axs[pltN, 0].plot(brainResults.pf_pc[:, 175, 2]) 
    axs[pltN, 0].plot(brainResults.pf_pc[:, 300, 2]) 
    axs[pltN, 0].plot(brainResults.pf_pc[:, 400, 2])
    axs[pltN, 0].set_xlabel("Trial")
    axs[pltN, 0].set_ylabel("PF-PC Weight")
    axs[pltN, 0].set_title("PF-PC Weights for Joint 2 Agonist")
    axs[pltN, 0].legend(["100ms", "350ms", "600ms", "800ms"])
    # plotting MF_DCN weight of J2 Anti over nTrials
    pltN = 0
    axs[pltN, 1].plot(brainResults.mf_dcn[:, 2])
    axs[pltN, 1].set_xlabel("Trial")
    axs[pltN, 1].set_ylabel("Weight")
    axs[pltN, 1].set_title("MF_DCN Weight Joint 2 Agonist")
    # plotting PC_DCN weight of J2 Anti over nTrials
    pltN += 1
    axs[pltN, 1].plot(brainResults.pc_dcn[:, 2])
    axs[pltN, 1].set_xlabel("Trial")
    axs[pltN, 1].set_ylabel("Weight")
    axs[pltN, 1].set_title("PC_DCN Weight Joint 2 Agonist")
    # plotting DCN activity 
    pltN += 1
    axs[pltN, 1].set_xlabel("Time (s)")
    axs[pltN, 1].set_ylabel("Weight")
    axs[pltN, 1].set_title("DCN activity Joint 2 Agonist")
    axs[pltN, 1].plot(time, brainResults.dcn[sampleTrialNums[0]-1, :, 2])
    axs[pltN, 1].plot(time, brainResults.dcn[sampleTrialNums[1]-1, :, 2])
    axs[pltN, 1].plot(time, brainResults.dcn[sampleTrialNums[2]-1, :, 2])
    axs[pltN, 1].plot(time, brainResults.dcn[sampleTrialNums[3]-1, :, 2])
    axs[pltN, 1].legend([f"Trial {sampleTrialNums[0]}", f"Trial {sampleTrialNums[1]}", f"Trial {sampleTrialNums[2]}", f"Trial {sampleTrialNums[3]}"])

    saveLoc.parent.mkdir(parents=True, exist_ok=True)
    if save:
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_brain" + saveLoc.suffix), dpi=300)
    if show:
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
    filename.parent.mkdir(parents=True, exist_ok=True)
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
    brainResults = brainData(nTrials, len(time), arm.njoints, brain.getnPFs())
    state = 0
    step = np.floor(len(time) / brain.getnPFs())
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
            brainResults.pc[trial, i,:]  = brain.getPC()
            brainResults.dcn[trial, i,:] = brain.getDCN()
            if trial == nTrials-1:
                final_traj.torq[i] = torque + corrTorque + torrPd
                final_traj.eePos[i,:] = arm.getPos()
                final_traj.vel[i,:]   = currVel
                final_traj.acel[i,:]  = currAcc
        # reset between each trial
        currPos          = traj_w_error.pos[0,:] 
        currVel          = traj_w_error.vel[0,:] 
        currAcc          = traj_w_error.acel[0,:] 
        arm.move(currPos, currVel, currAcc)
        # store for plotting
        errorTot[trial]  = np.sum([np.linalg.norm(des - act) for (des,act) in zip(traj_w_error.pos,  final_traj.pos)])
        brainResults.mf_dcn[trial, :] = brain.getMF_DCN()
        brainResults.pc_dcn[trial, :] = brain.getPC_DCN()
        brainResults.pf_pc[trial, :]  = brain.getPF_PC()
    return final_traj, errorTot, brainResults

def fillTrajectory(traj, arm):
    """
    computes any missing derivates through getDerivatives()
    and computes missing joint positions thru IK makeJointData()
    """
    time = traj.time
    if traj.velocity is None or traj.acceleration is None:
        traj.velocity, traj.acceleration = getDerivatives(time, traj.position, dx=traj.velocity)
    if traj.joint_position is None:
        traj.joint_position, traj.joint_velocity, traj.joint_acceleration = arm.makeJointData(traj.position, traj.velocity, traj.acceleration)
    elif traj.joint_velocity is None or traj.joint_acceleration is None:
        traj.joint_velocity, traj.joint_acceleration = getDerivatives(time, traj.joint_position, dx=traj.joint_velocity)
    return traj

def playVideo(time, trajectories, arm_ids, arm, illusoryArm):
    """
    creates the visualizer through meshcat and prompts user
    to play the different trajectories 
    Args:
        time: time vector. Only works with constant dt
        trajectories: trajectories dict, selection # -> trajectory
        arm_ids: arm dict, selection # -> arm file path. Trajectories and Arms should go together
        arm: arm object
        illusoryArm: illusory arm object
    """
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
            playArm = dynamic_3dof_arm(arm_ids[userTraj], disp=False)
            viz = initViz(playArm)
        viz.play(traj.pos, dt)
        prevChoice = userTraj

###################################
def simulate(exp, save, showOutput, grav):
###################################
    """
    Args:
        exp: Experiment object, see experiment_assets.py
        save: whether to save the results or not
        showOutput: whether to show output or not
        grav: whether gravity exists in sim or not
    """
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
    brain           = gc.Cerebellum(arm.njoints) 
    # brain           = cerebellum(n_dof=arm.njoints)

    # load desired trajectory
    # for smoothest trajectories, the trajectory should have analytically determined 
    # cartesian pos, vel, accel and joint-space q, qd, qdd. If not they will be calculated, 
    # but this introduces noise in the double-differentation 
    traj_data           = load_trajectory(trajFile)   
    traj_data           = fillTrajectory(traj_data, arm)
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

    # loading initial brain weights and setting plasticity
    plastic, wts = prepare_brain_weights(exp)
    brain.setActiveSites(plastic["pf_pc"], plastic["mf_dcn"], plastic["pc_dcn"])
    if wts is not None:
        brain.loadWts(wts.pf_pc, wts.mf_dcn, wts.pc_dcn)

    # Forward Dynamics: applying those computed torques to the actual arm
    # with a control feedback from the cerebellar model 
    final_traj, errorTot, brainResults = runSimulation(arm, traj_w_error, desired_ee_traj, time, brain, nTrials=exp.nTrials)
    # no-brain case for a control 
    cntrl_traj = arm.forwardDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.torq, time)
    
    # saving trajectories
    trajectories = {1: traj_no_error, 2: cntrl_traj, 3: final_traj, 4: traj_w_error}
    arm_ids      = {1: armFile,       2: armFile,    3: armFile,    4: illFile}
    if save:
        save_trajectories(trajectories, arm_ids, time[1] - time[0], filename=exp.results)
        save_weights(exp.finalWts, brainResults.pf_pc[-1, :], brainResults.mf_dcn[-1,:], brainResults.pc_dcn[-1,:])

    # plotting 
    errJointNoBrain    = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.pos,  cntrl_traj.pos)]
    plot_arm_results(traj_no_error, cntrl_traj, traj_w_error, final_traj, time, n_dof, show=showOutput, saveLoc=exp.graphs, save=save)
    plot_brain_results(errorTot, errJointNoBrain, brainResults, time, show=showOutput, saveLoc=exp.graphs, save=save)

    if not showOutput:
        print("Simulation Complete...")
        return

    # makes the sim in browser. Make sure looking at http://127.0.0.1:7000/static/ NOT http://127.0.0.1:7000
    playVideo(time, trajectories, arm_ids, arm, illusoryArm)