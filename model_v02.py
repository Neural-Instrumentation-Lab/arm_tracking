'''
MODEL V01

Implementation of cerebellar error correction
with joystick control, the cerebellar model is 
from the paper "Distributed cerebellar plasticity implements adaptable gain control in a 
manipulation task: a closed-loop robotic simulation" by Garrido et All

@Author: James Soley
@Date: Fall 2027
'''

## IMPORTS #####################################################################
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import pickle
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import deque
from arm_assets_v02 import armTraj
from joystick_movement import JoystickDotSimulator
from garrido_brain_v01 import cerebellum
# import garrido_brain as gb
from experiment_assets import load_trajectory
from simulate import runSimulation
from plot_results import plot_arm_results, plot_brain_results

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

def save_trajectories(trajectories, dt, filename="path_exp.pkl"):
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
        key: {"pos": traj.pos, "dt":dt}
        for key, traj in trajectories.items()
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def fillTrajectory(traj, arm):
    """
    computes any missing derivates through getDerivatives()
    and computes missing joint positions thru IK makeJointData()
    """
    time = traj.time
    if traj.velocity is None:
        traj.velocity, _ = getDerivatives(time, traj.position)
    if traj.joint_position is None:
        traj.joint_position = arm.makeJointData(traj.position, time)
    elif traj.joint_velocity is None:
        traj.joint_velocity, _ = getDerivatives(time, traj.joint_position)
    return traj

def playVideo(time, trajectories, arm_ids, arm, illusoryArm, fps=30):
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
            playArm = baxter_reduced(arm_ids[userTraj], pkgDirs=["./"], disp=False)
            viz = initViz(playArm)
        viz.play(traj.pos[::int(len(time)/(fps*time[-1]))], 1/fps)
        prevChoice = userTraj

###################################
def simulate(exp, save, showOutput, grav, makeMovie):
###################################
    """
    Args:
        exp: Experiment object, see experiment_assets.py
        save: whether to save the results or not
        showOutput: whether to show output or not
        grav: whether gravity exists in sim or not
    """
    trajFile   = exp.trajectory
    n_dof      = exp.nDof
    joystick   = JoystickDotSimulator()
    brain      = cerebellum([-2, -1], [-1, -1], [2, 1], [1, 1], n_dof=2)

    # load desired trajectory
    # for smoothest trajectories, the trajectory should have analytically determined 
    # cartesian pos, vel, accel and joint-space q, qd, qdd. If not they will be calculated, 
    # but this introduces noise in the double-differentation
    traj_data           = load_trajectory(trajFile)
    traj_data           = fillTrajectory(traj_data, joystick)
    time                = traj_data.time
    desired_ee_traj     = armTraj(pos=traj_data.position, vel=traj_data.velocity)
    traj_w_error        = armTraj(pos=traj_data.position, vel=traj_data.velocity)
    traj_no_error       = armTraj(pos=traj_data.position, vel=traj_data.velocity)
    traj_no_error.eePos = desired_ee_traj.pos

    # Forward Dynamics: applying those computed torques to the actual arm
    # with a control feedback from the cerebellar model 
    final_traj, errorTot, offset = runSimulation(joystick, traj_w_error, desired_ee_traj, time, brain, nTrials=exp.nTrials)
    # the simulation has a random start point, so we unroll by offset
    aligned_desired_pos = np.roll(traj_w_error.pos, -offset, axis=0) # or desired_ee_traj.pos, depending on what animate_inputs_duo expects
    aligned_desired_vel = np.roll(traj_w_error.vel, -offset, axis=0) # or desired_ee_traj.pos, depending on what animate_inputs_duo expects
    joystick.animate_inputs_duo(
        final_traj.pos, 
        aligned_desired_pos, 
        time, 
        ["actual path", "actual pos", "desired path", "desired pos"]
    )
    # joystick.animate_inputs_duo(final_traj.pos, traj_no_error.pos, time, 
    #                             ["actual path", "actual pos", "desired path", "desired pos"])
    joystick.animate_joystick(final_traj.vel)
    # saving trajectories
    trajectories = {1: traj_no_error, 2: final_traj}
    if save:
        save_trajectories(trajectories, time[1] - time[0], filename=exp.results)
        # save_weights(exp.finalWts, brainResults.pf_pc[-1, :], brainResults.mf_dcn[-1, :], brainResults.pc_dcn[-1, :])

    # plotting 
    # errJointNoBrain    = [np.linalg.norm(des - act)/(n_dof-1) for (des,act) in zip(traj_no_error.pos[:6],  cntrl_traj.pos[:6])]
    # wts = brain.pf_pc_wts
    # plot_arm_results(traj_no_error, cntrl_traj, traj_w_error, final_traj, time, n_dof, show=showOutput, saveLoc=exp.graphs, save=save)
    # plot_brain_results(wts, errJointNoBrain, errorTot, time, n_dof, show=showOutput, saveLoc=exp.graphs, save=save)

    if not showOutput:
        print("Simulation Complete...")
        return

    fig, axs = plt.subplots()
    axs.plot(final_traj.vel)
    axs.plot(aligned_desired_vel)
    axs.legend(["x actual", "y actual", "x des", "y des"])
    plt.show()
    fig, axs = plt.subplots()
    axs.plot(errorTot/(len(time)*(n_dof-1)))
    axs.set_xlabel("Trial")
    axs.set_ylabel("Mean Absolute Error")
    axs.set_title("Evolution of MAE")
    plt.show()
    wts = brain.pf_pc_wts
    plt.imshow(wts, cmap='viridis', interpolation='nearest', aspect="auto")
    plt.xlabel("PCs")
    plt.ylabel("PFs")
    plt.axvline(49, color='red', linestyle='--')
    plt.axvline(99, color='red', linestyle='--')
    plt.axvline(149, color='red', linestyle='--')
    plt.colorbar(label='Conductance (S)')
    labels = [
        "Joint 1 Agonist",
        "Joint 1 Antagonist",
        "Joint 2 Agonist",
        "Joint 2 Antagonist"
    ]

    # Centers of the four regions
    centers = [25, 75, 125, 175]

    for x, label in zip(centers, labels):
        plt.text(
            x, 
            -0.05, 
            label,
            ha='center',
            va='top',
            transform=plt.gca().get_xaxis_transform()
        )
    plt.show()
    # makes the sim in browser. Make sure looking at http://127.0.0.1:7000/static/ NOT http://127.0.0.1:7000
    # playVideo(time, trajectories, arm_ids, arm, arm)