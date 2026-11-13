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
    record_brain: store DCN spikes (every brain tick) / PF-PC weights (every trial). Large:
                  nTrials x ticks x nDCN. brainResults.tick_step gives the sim step of each tick.
                  If False, the returned brainResults is None.
    verbose: print a line after every trial
    snapshot_every: keep a copy of the arm's path every N trials (and the last trial), for the
                    learning timelapse. Returned as a list of (trial, armTraj, start_idx); empty if None.
    '''
    # initialize parameters
    bmi.setup(brain)
    n_steps = len(time)
    step = int(round(n_steps / (np.floor(time[-1] / brain_dt) + 1), 0))
    n_ticks = -(-n_steps // step)     # most brain ticks a trial can have
    brainResults = None
    if record_brain:
        brainResults = brainData(nTrials, n_ticks, brain.nDCN*brain.nJoints, brain.nGC*brain.nJoints, brain.nPC*brain.nJoints)
    final_traj = armTraj(pos=np.zeros_like(desired_traj.pos),
                         vel=np.zeros_like(desired_traj.vel))
    errorTot = np.zeros(nTrials)
    snapshots = []

    # brain commands and inputs stored for delay
    delEff = teachingLagSteps(time)
    delAff = int(round(DEL_AFF * (n_steps / time[-1])))

    # per-step work is on plain Python floats/lists: numpy calls on 2-element arrays cost
    # far more than the arithmetic, and this loop runs n_steps * nTrials times
    des_pos = desired_traj.pos.tolist()
    # dt used at sim step s (s = 0 reuses step 1's, as before; the arm doesn't move on the first step)
    step_dt = np.diff(time, prepend=time[0])
    if n_steps > 1:
        step_dt[0] = step_dt[1]
    step_dt = step_dt.tolist()

    for trial in range(nTrials):
        if reset or trial == 0:
            # 1. Pick a random starting index along the trajectory
            start_idx = np.random.randint(0, n_steps)

            # 2. Move arm to that random starting position
            arm.move(desired_traj.pos[start_idx, :].copy())

            # Reset queues for the new trial
            errSig = [0.0] * len(des_pos[0])                        # error the brain sees
            qSig = (errSig, errSig)                                  # (actual, desired) the brain sees
            prevErrors = deque()
            prevPos = deque()
            prevComm = deque()
            corrTorque = [0.0] * len(des_pos[0])

            # steps since the last reset, used for delay and brain-rate timing
            elapsed = 0

        pos = arm.getPos().tolist()
        tick = 0
        actual_pos = []      # arm position after every step
        commands = []        # command applied at every step
        # 3. Iterate through one full trial cycle (n_steps)
        for step_idx in range(n_steps):
            # Wrap around the circular trajectory using modulo
            i = (start_idx + step_idx) % n_steps

            # teaching signal compares the dot with the desired state from 50 ms earlier (efferent delay),
            # as EDLUT_BAXTER's error node does (same as the arm branch); the trajectory is periodic, so wrap
            lagged = des_pos[(i - delEff) % n_steps]
            prevErrors.append([d - a for d, a in zip(lagged, pos)])
            prevPos.append((pos, des_pos[i]))

            # Use steps since last reset for delay logic
            if elapsed >= delEff:
                errSig = prevErrors.popleft()
                qSig = prevPos.popleft()

            # compute corrections (using steps since last reset for frequency check)
            if elapsed % step == 0:
                brain.compute(qSig[0], qSig[1], errSig, dt=brain_dt)
                dcnOut = brain.dcnOut
                prevComm.append(bmi.decode(dcnOut).tolist())
                if elapsed >= delAff:
                    corrTorque = prevComm.popleft()
                if record_brain:
                    brainResults.dcn[trial, tick, :] = dcnOut
                    brainResults.tick_step[trial, tick] = step_idx
                tick += 1

            # move the arm
            if elapsed > 0:
                arm.step(corrTorque[0], corrTorque[1], step_dt[step_idx])
                pos = arm.getPos().tolist()
            elapsed += 1

            actual_pos.append(pos)
            commands.append(corrTorque)
        trial_actual_pos = np.array(actual_pos)
        final_traj.pos[:] = trial_actual_pos
        final_traj.vel[:] = commands

        if verbose:
            print(f"trial {trial} done")
        if record_brain:
            brainResults.pf_pc[trial, :] = brain.pf_pc_wts

        # MAE compares the same signals as the teaching signal: actual vs the desired state from 50 ms earlier
        trial_desired_pos = desired_traj.pos[(start_idx + np.arange(n_steps) - delEff) % n_steps]
        errorTot[trial] = np.sum(np.linalg.norm(trial_desired_pos - trial_actual_pos, axis=1))

        if snapshot_every and (trial % snapshot_every == 0 or trial == nTrials - 1):
            snapshots.append((trial, armTraj(pos=final_traj.pos.copy(), vel=final_traj.vel.copy()), start_idx))

    return final_traj, errorTot, start_idx, brainResults, snapshots
