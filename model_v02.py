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
from arm_assets_v02 import dynamic_3dof_arm, dynamic_2dof_arm, armTraj, angle_diff, baxter_reduced
from garrido_brain_v01 import cerebellum
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import pickle
from experiment_assets import load_trajectory, load_weights, save_weights, prepare_brain_weights
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import deque

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
    # calculating errors
    errDistBrain       = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.eePos,  final_traj.eePos[-1,:])]
    errDistNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.eePos,  cntrl_traj.eePos)]
    errTorqNoBrain     = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  cntrl_traj.torq)]
    errTorqBrain       = [np.linalg.norm(des - act) for (des,act) in zip(traj_no_error.torq,  final_traj.torq[-1, :])]

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.5, 1])

    # Large plot on the left
    ax_pos = fig.add_subplot(gs[:, 0])

    # Three smaller plots on the right
    ax_t1 = fig.add_subplot(gs[0, 1])
    ax_t2 = fig.add_subplot(gs[1, 1])
    ax_err = fig.add_subplot(gs[2, 1])

    ax_t1.plot(time, traj_no_error.torq[:, 0], '--', lw=2)
    ax_t1.plot(time, final_traj.torq[-1, :, 0], lw=2)
    ax_t1.set_title("Joint 1 Corrective Torque")
    ax_t1.set_ylabel("Torque")
    ax_t1.legend(["Ideal", "Brain"])
    ax_t1.grid(True)
    ax_t1.tick_params(labelbottom=False)

    ax_t2.plot(time, traj_no_error.torq[:, 1], '--', lw=2)
    ax_t2.plot(time, final_traj.torq[-1, :, 1], lw=2)
    ax_t2.set_title("Joint 2 Corrective Torque")
    ax_t2.set_ylabel("Torque")
    ax_t2.legend(["Ideal", "Brain"])
    ax_t2.grid(True)
    ax_t2.tick_params(labelbottom=False)
    ax_t2.sharex(ax_t1)

    colors = plt.cm.tab10.colors   # or plt.rcParams['axes.prop_cycle'].by_key()['color']

    leg = []

    for j in range(nDof - 1):
        c = colors[j % len(colors)]

        ax_pos.plot(
            time,
            traj_no_error.pos[:, j],
            '--',
            color=c,
            lw=2,
        )

        ax_pos.plot(
            time,
            final_traj.pos[:, j],
            '-',
            color=c,
            lw=2,
        )

        leg.extend([
            f"Joint {j+1} Ideal",
            f"Joint {j+1} Actual"
        ])

    ax_pos.set_title("Joint Positions")
    ax_pos.set_ylabel("Joint Position (rad)")
    ax_pos.set_xlabel("Time")
    ax_pos.grid(True)
    ax_pos.legend(leg, ncol=2)
    ax_pos.legend(leg, loc="center left", bbox_to_anchor=(1.02, 0.5))

    ax_err.plot(time, errDistNoBrain, lw=2)
    ax_err.plot(time, errDistBrain, lw=2)
    ax_err.set_title("Distance Error")
    ax_err.set_xlabel("Time")
    ax_err.set_ylabel("EE Distance Error")
    ax_err.legend(["No Brain", "Brain"])
    ax_err.grid(True)
    ax_err.sharex(ax_t1)

    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_arm" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)

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

    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_brain" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)

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

def supervisor_torque(currPos, currVel, qMin, qMax, smoothing):
    """Mimics mechanical brakes: zero everywhere except near joint limits,
    where it pushes back proportionally to how far past the safe margin
    the joint has gone."""
    kp = np.array([5, 5, 5, 5, 1, 1, 1])
    kd = np.array([20, 20, 20, 20, 3, 3, 3])
    correction = np.zeros_like(currPos)
    near_upper = currPos > qMax
    near_lower = currPos < qMin
    correction[near_upper] = kp[near_upper]*(qMax[near_upper]-currPos[near_upper])  - smoothing[near_upper]*kd[near_upper]*currVel[near_upper]
    correction[near_lower] = kp[near_lower]*(qMin[near_lower]-currPos[near_lower])  - smoothing[near_lower]*kd[near_lower]*currVel[near_lower]
    smoothing[near_upper] += 0.02
    smoothing[near_lower] += 0.02
    smoothing[(currPos <= qMax) * (currPos >= qMin)] = 0
    smoothing = np.clip(smoothing, 0, 1)
    return correction, smoothing

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
    currVel       = np.zeros_like(traj_w_error.vel[0,:])
    currAcc       = np.zeros_like(traj_w_error.acel[0,:])
    arm.move(currPos, currVel, currAcc)
    final_traj    = armTraj(pos=np.zeros_like(traj_w_error.pos), 
                            torq=np.zeros((nTrials, len(time), arm.njoints)),
                            eePos=np.zeros((nTrials, len(time), 3)),
                            vel = np.zeros_like(traj_w_error.vel),
                            accel=np.zeros_like(traj_w_error.acel))
    corrTorque = np.zeros_like(currPos)
    torrPd = np.zeros_like(currPos)
    errorTot = np.zeros(nTrials)
    # brainResults = brainData(nTrials, len(time), arm.njoints, brain.getnPFs())
    step = int(round(len(time) / (np.floor(time[-1] / brain.timeStep) + 1), 0))
    # PD Constants
    kp = np.ones(arm.njoints)*0
    kd = [5, 5, 5, 5, 1, 1, 1]
    # kd = 2*np.sqrt(kp)
    # kp = np.array([5, 5, 5, 5, 1, 1, 1])
    # kd = np.array([20, 20, 20, 20, 3, 3, 3])
    qMin = arm.model.lowerPositionLimit
    qMax = arm.model.upperPositionLimit
    qdMax = arm.model.velocityLimit
    tauMax = arm.model.effortLimit
    # brain commands and inputs stored for delay
    delEff = int(round(50e-3 * (len(time) / time[-1])))
    delAff = int(round(50e-3 * (len(time) / time[-1])))
    errSig = (np.zeros_like(currPos), np.zeros_like(currVel))
    qSig = (np.zeros_like(currPos), np.zeros_like(currVel), np.zeros_like(currPos), np.zeros_like(currVel))
    torSig = np.zeros_like(corrTorque)
    prevErrors = deque()
    prevPos = deque()
    prevComm = deque()
    smoothing = np.zeros_like(corrTorque)
    for trial in range(nTrials):
        for i, torque in enumerate(traj_w_error.torq):
            # compute error 
            qError  = angle_diff(traj_w_error.pos[i], currPos)
            qdError = traj_w_error.vel[i] - currVel
            prevErrors.append((qError.copy(), qdError.copy()))
            prevPos.append((currPos.copy(), currVel.copy(), traj_w_error.pos[i].copy(), traj_w_error.vel[i].copy()))
            if trial != 0 or i >= delEff:
                errSig = prevErrors.popleft()
                qSig = prevPos.popleft()
            # compute corrections, but brain only fires once for every PF
            if i % step ==0:
                corr = brain.compute(qSig[0][:6], qSig[1][:6], qSig[2][:6], 
                                    qSig[3][:6], (errSig[0])[:6], (errSig[1])[:6]) 
                prevComm.append(np.concat((corr, [0])))
                if trial != 0 or i >= delAff:
                    corrTorque = prevComm.popleft()
                # corrTorque[6] = torque[6] # this is for the joint not controlled by the brain
            torrPd = kp*(qError) + kd*(qdError)
            torrSup, smoothing = supervisor_torque(currPos, currVel, arm.qSuppMin, arm.qSuppMax, smoothing)
            finalTorque = corrTorque + torrPd + torrSup
            np.clip(finalTorque, -tauMax, tauMax, out=finalTorque)
            # move the arm
            currAcc = arm.forward(currPos, currVel, finalTorque)
            if i > 0:
                dt      = time[i] - time[i-1]
                currVel = (currVel + currAcc *dt)
                # would clip vel here but their traj has vel need to be more than the model limit
                # np.clip(currVel, -qdMax, qdMax, out=currVel)
                currPos = pin.integrate(arm.model, currPos, currVel * dt)
            np.clip(currPos, qMin, qMax, out=currPos)
            arm.move(currPos, currVel, currAcc)
            # pinocchio does not automatically clamp joints to their limits
            # so we gotta do it ourselves
            # store results for plotting
            final_traj.pos[i,:]   = currPos
            final_traj.vel[i,:]   = currVel
            # brainResults.pc[trial, i,:]  = brain.getPC()
            # brainResults.dcn[trial, i,:] = brain.getDCN()
            final_traj.eePos[trial, i,:] = arm.getPos()
            final_traj.torq[trial, i] = corrTorque
            if trial == nTrials-1:
                final_traj.acel[i,:]  = currAcc
        n_at_ceiling = np.sum(brain.pf_pc_wts >= brain.W_MAX - 1e-15)
        n_at_floor = np.sum(brain.pf_pc_wts <= brain.W_MIN + 1e-15)
        print(f"n:{trial} ceil: {n_at_ceiling}, floor: {n_at_floor}")
        # print(f"q_min: {final_traj.pos.min(axis=0)} q_max: {final_traj.pos.max(axis=0)}")
        # print(f"qd_min: {final_traj.vel.min(axis=0)} qd_max: {final_traj.vel.max(axis=0)}")
        # reset between each trial
        # currPos          = traj_w_error.pos[0,:] 
        # currVel          = np.zeros_like(traj_w_error.vel[0,:])
        # currAcc          = np.zeros_like(traj_w_error.acel[0,:])
        # arm.move(currPos, currVel, currAcc)
        # prevErrors.clear()
        # prevPos.clear()
        # prevComm.clear()
        # store for plotting
        errorTot[trial]  = np.sum([np.linalg.norm(des[:6] - act[:6]) for (des,act) in zip(traj_w_error.pos,  final_traj.pos)])
        # brainResults.mf_dcn[trial, :] = brain.getMF_DCN()
        # brainResults.pc_dcn[trial, :] = brain.getPC_DCN()
        # brainResults.pf_pc[trial, :]  = brain.getPF_PC()
    touched = brain.pf_pc_wts[brain.pf_pc_wts != cerebellum.INIT_PF_PC_WT]
    print(f"Touched entries: {touched.size}")
    return final_traj, errorTot

def transform(traj, arm):
    '''
    for the baxter arm, ROS or gazebo or something the authors use
    has a different rotation / origin then pinocchio. This function
    takes points given in their world frame and transfroms them to 
    pinocchio's 
    Args:
        traj: trajectory with pos 
        arm: baxter arm
    '''
    traj.position = np.array([arm.applyTransform(p) for p in traj.position])
    if traj.velocity is not None:
        traj.velocity = np.array([arm.applyTransform(v, derivative=True) for v in traj.velocity])
    if traj.acceleration is not None:
        traj.acceleration = np.array([arm.applyTransform(a, derivative=True) for a in traj.acceleration])
    return traj


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

def movie(traj_no_error, final_traj, brainResults, time, nTrials, fname, joint=1):
    '''
    makes animated plots with matplot and saves them to location in fname.
    Takes a considerable amount of runtime. 
    '''
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[2, 1])  # left col wider than right

    timeskip = int(len(time) / 30)
    trialskip = int(nTrials / 30)

    # Big 3d plot of cartesian position of end effector 
    posax = fig.add_subplot(gs[:, 0], projection="3d")
    line = posax.plot([], [], [])[0]
    axes_buffer = 0.1
    posax.plot(traj_no_error.eePos[:,0], traj_no_error.eePos[:,1], traj_no_error.eePos[:,2])[0]
    posax.set_xlim(traj_no_error.eePos[:, 0].min()-axes_buffer, traj_no_error.eePos[:, 0].max()+axes_buffer)
    posax.set_ylim(traj_no_error.eePos[:, 1].min()-axes_buffer, traj_no_error.eePos[:, 1].max()+axes_buffer)
    posax.set_zlim(traj_no_error.eePos[:, 2].min()-axes_buffer, traj_no_error.eePos[:, 2].max()+axes_buffer)
    posax.set_xlabel('X [m]', fontsize=10)
    posax.set_ylabel('Y [m]', fontsize=10)
    posax.set_zlabel('Z [m]', fontsize=10)
    posax.legend(["actual", "ideal"])

    trial_array = range(nTrials)
    # joint to show in the plots
    jointSel = {1 : 0, 2 : 2, 3 : 4}
    jointIdx = jointSel[joint]

    # all these plots are on the right column
    # mf-dcn over the trials
    mfdcnax = fig.add_subplot(gs[0, 1])
    mfdcnagon = mfdcnax.plot([],[])[0] 
    mfdcnaagon = mfdcnax.plot([],[])[0]
    mfdcnax.set_xlim(1, nTrials)
    mfdcnax.set_ylim(0, max(brainResults.mf_dcn[:, jointIdx+1].max(), brainResults.mf_dcn[:,jointIdx].max()) + 2)
    mfdcnax.set_title(f"mf-dcn weight [Joint {joint}]")
    mfdcnax.legend(["agonist", "antagonist"])

    # pc-dcn over the trials
    pcdcnax = fig.add_subplot(gs[1,1])
    pcdcnag = pcdcnax.plot([],[])[0]
    pcdcnaag = pcdcnax.plot([],[])[0]
    pcdcnax.set_xlim(1 ,nTrials)
    pcdcnax.set_ylim(0, max(brainResults.pc_dcn[:,jointIdx].max(), brainResults.pc_dcn[:, jointIdx+1].max()) + 2) 
    pcdcnax.set_title(f"pc-dcn weight [Joint {joint}]")
    pcdcnax.legend(["agonist", "antagonist"])

    # pc activity over time
    pcax = fig.add_subplot(gs[2, 1])
    pcagon = pcax.plot([],[])[0]
    pcaagon = pcax.plot([],[])[0]
    pcax.set_xlim(0, time.max())
    pcax.set_ylim(0, 1)
    pcax.set_title(f"pc activity [Joint {joint}]")
    pcax.legend(["agonist", "antagonist"])

    # dcn activity over time
    dcnax = fig.add_subplot(gs[3, 1])
    dcnagon = dcnax.plot([],[])[0]
    dcnaagon = dcnax.plot([],[])[0]
    dcnax.set_xlim(0, time.max())
    dcnax.set_ylim(0, max(brainResults.dcn[:, :, jointIdx].max(), brainResults.dcn[:, :, jointIdx+1].max()))
    dcnax.set_title(f"dcn activity [Joint {joint}]")
    dcnax.legend(["agonist", "antagonist"])


    fig.tight_layout()
    tot_frames = int((len(time)/timeskip)*np.floor(nTrials/trialskip))
    def update(frame_idx):
        '''
        updates frame data in animation
        '''
        trial_idx = ((frame_idx*timeskip) // len(time))*trialskip
        step_idx = (timeskip*frame_idx) % len(time)
        currPos = final_traj.eePos[trial_idx, :step_idx+1, :]
        line.set_data_3d([currPos[:, 0], currPos[:, 1], currPos[:, 2]])
        posax.set_title(f"Trial: {trial_idx+1}")
        mfdcnagon.set_data(trial_array[:trial_idx], brainResults.mf_dcn[:trial_idx, jointIdx])
        mfdcnaagon.set_data(trial_array[:trial_idx], brainResults.mf_dcn[:trial_idx, jointIdx+1])
        pcagon.set_data(time[:step_idx+1], brainResults.pc[trial_idx, :step_idx+1, jointIdx])
        pcaagon.set_data(time[:step_idx+1], brainResults.pc[trial_idx, :step_idx+1, jointIdx+1])
        dcnagon.set_data(time[:step_idx+1], brainResults.dcn[trial_idx, :step_idx+1, jointIdx])
        dcnaagon.set_data(time[:step_idx+1], brainResults.dcn[trial_idx, :step_idx+1, jointIdx+1])
        pcdcnag.set_data(trial_array[:trial_idx], brainResults.pc_dcn[:trial_idx, jointIdx])
        pcdcnaag.set_data(trial_array[:trial_idx], brainResults.pc_dcn[:trial_idx, jointIdx+1])

    ani = animation.FuncAnimation(fig, update, tot_frames, interval=8, blit=False)
    writer = animation.FFMpegWriter(fps=30, bitrate=1800)
    saveLoc = Path(fname)
    saveLoc.parent.mkdir(parents=True, exist_ok=True)
    ani.save(saveLoc, writer, dpi=100)
    plt.close()

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
    armFile    = exp.actualArm
    illFile    = exp.illusoryArm 
    n_dof      = exp.nDof 
    # instantiate limb, motor control unit, brain    
    package_dirs = ["./"] 
    arm         = baxter_reduced("baxter_description/urdf/baxter_fixed.urdf", package_dirs)
    illusoryArm = baxter_reduced("baxter_description/urdf/baxter_fixed.urdf", package_dirs) # not needed in this version
    # brain           = gc.Cerebellum(arm.njoints) 
    # -1 joint because joint w2 is uncontrolled by the brain
    brain           = cerebellum(arm.qSuppMin[:6], arm.qdSuppMin[:6], arm.qSuppMax[:6], arm.qdSuppMax[:6], n_dof=arm.njoints - 1)

    # load desired trajectory
    # for smoothest trajectories, the trajectory should have analytically determined 
    # cartesian pos, vel, accel and joint-space q, qd, qdd. If not they will be calculated, 
    # but this introduces noise in the double-differentation 
    traj_data           = load_trajectory(trajFile)   
    traj_data           = transform(traj_data, arm)
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
    # plastic, wts = prepare_brain_weights(exp)
    # brain.setActiveSites(plastic["pf_pc"], plastic["mf_dcn"], plastic["pc_dcn"])
    # if wts is not None:
        # brain.loadWts(wts.pf_pc, wts.mf_dcn, wts.pc_dcn)

    # Forward Dynamics: applying those computed torques to the actual arm
    # with a control feedback from the cerebellar model 
    final_traj, errorTot = runSimulation(arm, traj_w_error, desired_ee_traj, time, brain, nTrials=exp.nTrials)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # y= np.arange(len(brain.pf_pc_wts))
    # x= np.arange(len(brain.pf_pc_wts[0]))
    # (x, y) = np.meshgrid(x, y)
    # surf = ax.plot_surface(x, y, brain.pf_pc_wts, cmap='viridis', edgecolor='none')

    # no-brain case for a control 
    cntrl_traj = arm.forwardDynamics(traj_w_error.pos, traj_w_error.vel, traj_w_error.torq, time)
    
    # saving trajectories
    trajectories = {1: traj_no_error, 2: cntrl_traj, 3: final_traj, 4: traj_w_error}
    arm_ids      = {1: armFile,       2: armFile,    3: armFile,    4: illFile}
    if save:
        save_trajectories(trajectories, arm_ids, time[1] - time[0], filename=exp.results)
        # save_weights(exp.finalWts, brainResults.pf_pc[-1, :], brainResults.mf_dcn[-1, :], brainResults.pc_dcn[-1, :])
    wts = brain.pf_pc_wts
    print(wts.min())
    print(wts.max())
    plt.imshow(wts, cmap='viridis', interpolation='nearest', aspect="auto")
    plt.colorbar()
    plt.show()

    # plotting 
    errJointNoBrain    = [np.linalg.norm(des - act)/(n_dof-1) for (des,act) in zip(traj_no_error.pos[:6],  cntrl_traj.pos[:6])]
    plot_arm_results(traj_no_error, cntrl_traj, traj_w_error, final_traj, time, n_dof, show=showOutput, saveLoc=exp.graphs, save=save)
    # plot_brain_results(errorTot, errJointNoBrain, brainResults, time, show=showOutput, saveLoc=exp.graphs, save=save)

    fig, axs = plt.subplots()
    pltN = 0
    axs.plot(errorTot/(len(time)*(n_dof-1)))
    axs.set_xlabel("Trial")
    axs.set_ylabel("Mean Absolute Error")
    axs.set_title("Evolution of MAE")
    axs.axhline(np.sum(errJointNoBrain)/len(time), color='r', linestyle='--', linewidth=2)
    axs.legend(["Brain Error", "No Brain Error"])
    plt.show()

    # if makeMovie:
    #     movie(traj_no_error, final_traj, brainResults, time, exp.nTrials, "results/videos/"+exp.name+".mp4")

    if not showOutput:
        print("Simulation Complete...")
        return


    # makes the sim in browser. Make sure looking at http://127.0.0.1:7000/static/ NOT http://127.0.0.1:7000
    playVideo(time, trajectories, arm_ids, arm, illusoryArm)