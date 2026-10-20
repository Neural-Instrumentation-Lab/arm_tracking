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
from arm_assets_v02 import armTraj, JoystickDotSimulator
# from garrido_brain_v02 import cerebellum
from garrido_brain import cerebellum   # C++ port of v02, build with build.sh
from experiment_assets import load_trajectory, save_weights, brainData
from simulate import runSimulation
from plot_results import plot_arm_results, plot_brain_results
from play_arm_path import animate_combined

matplotlib.use('TkAgg')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S"
)



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
        key: {"pos": traj.pos, "vel" : traj.vel, "dt":dt}
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

###################################
def simulate(exp, save, showOutput, makeMovie, reset=True):
###################################
    """
    Args:
        exp: Experiment object, see experiment_assets.py
        save: whether to save the results or not
        showOutput: whether to show output or not
        reset: whether the arm is reset to a random start point every trial
    """
    trajFile   = exp.trajectory
    n_dof      = exp.nDof
    joystick   = JoystickDotSimulator()
    # garrido_brain_small_dt.cerebellum also needs dt=exp.brain_timestep here
    # brain      = cerebellum(list(exp.qMins), list(exp.qMaxs), n_dof=n_dof)
    brain      = cerebellum(list(exp.qMins), list(exp.qMaxs), n_dof=n_dof, dt=exp.brain_timestep)

    # load desired trajectory
    # for smoothest trajectories, the trajectory should have analytically determined 
    # cartesian pos, vel, accel and joint-space q, qd, qdd. If not they will be calculated, 
    # but this introduces noise in the double-differentation
    traj_data           = load_trajectory(trajFile)
    traj_data           = fillTrajectory(traj_data, joystick)
    time                = traj_data.time
    desired_traj        = armTraj(pos=traj_data.position, vel=traj_data.velocity)

    # Forward Dynamics: applying those computed torques to the actual arm
    # with a control feedback from the cerebellar model 
    final_traj, errorTot, offset, brainResults = runSimulation(joystick, desired_traj, time, brain, nTrials=exp.nTrials, brain_dt=exp.brain_timestep, reset=reset)
    # the simulation has a random start point, so we unroll by offset
    # saving trajectories
    aligned_desired_pos = np.roll(desired_traj.pos, -offset, axis=0) 
    aligned_desired_vel = np.roll(desired_traj.vel, -offset, axis=0) 
    aligned_des = armTraj(pos= aligned_desired_pos, vel=aligned_desired_vel)

    trajectories = {1: aligned_des, 2: final_traj}
    if save:
        save_trajectories(trajectories, time[1] - time[0], filename=exp.results)
        save_weights(exp.finalWts, brainResults)

    # showing results 
    if showOutput:
        # have to align because of random start point
        animate_combined(final_traj, aligned_des, time)
    plot_arm_results(final_traj, aligned_des, show=showOutput, saveLoc=exp.graphs, save=save)
    plot_brain_results(errorTot, time, brain, show=showOutput, saveLoc=exp.graphs, save=save)
    print("Simulation Complete...")
    # playVideo(time, trajectories, arm_ids, arm, arm)