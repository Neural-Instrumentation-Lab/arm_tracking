import numpy as np
from arm_assets_v02 import armTraj
from experiment_assets import brainData
from collections import deque
DEL_AFF = 50e-3
DEL_EFF = 50e-3

def teachingLagSteps(time):
    '''How many trajectory samples the desired state lags in the teaching signal and the MAE (the efferent delay).'''
    return int(round(DEL_EFF * (len(time) / time[-1])))

def runSimulation(arm, desired_traj, time, brain, bmi, nTrials, brain_dt, reset=True, record_brain=True, verbose=True, snapshot_every=None):
    '''
    Controls the arm with the cerebellar feedback loop

    bmi: decodes the brain's DCN spikes into the command sent to the arm (see bmi.py).
         It is set up from the brain here, so it can be passed in fresh.

    reset: if True, every trial starts at a random point on the trajectory with the
           arm moved there and the delay queues cleared. If False, only the first trial
           does this and later trials continue from where the previous one ended
           (the trajectory is periodic).
    record_brain: store DCN spikes / PF-PC weights every trial (large: nTrials x steps x nDCN).
                  If False, the returned brainResults is None.
    verbose: print a line after every trial
    snapshot_every: keep a copy of the arm's path every N trials (and the last trial), for the
                    learning timelapse. Returned as a list of (trial, armTraj, start_idx); empty if None.
    '''
    # initialize parameters
    bmi.setup(brain)
    n_steps = len(time)
    brainResults = None
    if record_brain:
        brainResults = brainData(nTrials, n_steps, brain.nDCN*brain.nJoints, brain.nGC*brain.nJoints, brain.nPC*brain.nJoints)
    final_traj = armTraj(pos=np.zeros_like(desired_traj.pos),
                         vel=np.zeros_like(desired_traj.vel))
    errorTot = np.zeros(nTrials)
    snapshots = []
    
    step = int(round(n_steps / (np.floor(time[-1] / brain_dt) + 1), 0))
    
    # brain commands and inputs stored for delay
    delEff = teachingLagSteps(time)
    delAff = int(round(DEL_AFF * (n_steps / time[-1])))
    kp = [0.1, 0.1]
    
    for trial in range(nTrials):
        if reset or trial == 0:
            # 1. Pick a random starting index along the trajectory
            start_idx = np.random.randint(0, n_steps)
            
            # 2. Move arm to that random starting position
            currPos = desired_traj.pos[start_idx, :].copy()
            currVel = np.zeros_like(desired_traj.vel[0,:])
            arm.move(currPos)
            
            # Reset queues for the new trial
            errSig = (np.zeros_like(currPos), np.zeros_like(currVel))
            qSig = (np.zeros_like(currPos), np.zeros_like(currVel), np.zeros_like(currPos), np.zeros_like(currVel))
            prevErrors = deque()
            prevPos = deque()
            prevComm = deque()
            corrTorque = np.zeros_like(currPos)
            
            # steps since the last reset, used for delay and brain-rate timing
            elapsed = 0
        
        trial_actual_pos = []
        
        # 3. Iterate through one full trial cycle (n_steps)
        for step_idx in range(n_steps):
            # Wrap around the circular trajectory using modulo
            i = (start_idx + step_idx) % n_steps
            
            # teaching signal compares the dot with the desired state from 50 ms earlier (efferent delay),
            # as EDLUT_BAXTER's error node does (same as the arm branch); the trajectory is periodic, so wrap
            iLag    = (i - delEff) % n_steps
            qError  = desired_traj.pos[iLag].copy() - arm.getPos().copy()
            qdError = desired_traj.vel[iLag].copy() - corrTorque;
            prevErrors.append((qError.copy(), qdError.copy()))
            prevPos.append((arm.getPos().copy(), desired_traj.pos[i].copy(), corrTorque, desired_traj.vel[i].copy()))
            
            # Use steps since last reset for delay logic
            if elapsed >= delEff:
                errSig = prevErrors.popleft()
                qSig = prevPos.popleft()
                
            # compute corrections (using steps since last reset for frequency check)
            if elapsed % step == 0:
                brain.compute(qSig[0], qSig[1], errSig[0], dt=brain_dt)
                corr = bmi.decode(brain.dcnOut)
                prevComm.append(corr)
                if elapsed >= delAff:
                    corrTorque = prevComm.popleft()
                    
            # move the arm
            if elapsed > 0:
                dt = time[max(step_idx, 1)] - time[max(step_idx, 1)-1]
                arm.step(corrTorque[0], corrTorque[1], dt)
            elapsed += 1
                
            trial_actual_pos.append(arm.getPos().copy())   # the arm updates its position array in place
            final_traj.pos[step_idx, :] = arm.getPos()
            final_traj.vel[step_idx, :] = corrTorque
            if record_brain:
                brainResults.dcn[trial, step_idx, :] = brain.dcnOut
            
        if verbose:
            print(f"trial {trial} done")
        if record_brain:
            brainResults.pf_pc[trial, :] = brain.pf_pc_wts
        
        # MAE compares the same signals as the teaching signal: actual vs the desired state from 50 ms earlier
        trial_desired_pos = np.array([desired_traj.pos[(start_idx + s - delEff) % n_steps] for s in range(n_steps)])
        errorTot[trial] = np.sum([np.linalg.norm(des - act) for (des, act) in zip(trial_desired_pos, trial_actual_pos)])

        if snapshot_every and (trial % snapshot_every == 0 or trial == nTrials - 1):
            snapshots.append((trial, armTraj(pos=final_traj.pos.copy(), vel=final_traj.vel.copy()), start_idx))
        
    return final_traj, errorTot, start_idx, brainResults, snapshots
