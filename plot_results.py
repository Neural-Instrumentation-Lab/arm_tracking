import matplotlib.pyplot as plt
import numpy as np
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

def plot_brain_results(wts, errJointNoBrain, errorTot, time, n_dof, show=True, save=False, saveLoc=None):
    plt.imshow(wts, cmap='viridis', interpolation='nearest', aspect="auto")
    plt.colorbar()
    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_arm" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close()
    fig, axs = plt.subplots()
    pltN = 0
    axs.plot(errorTot/(len(time)*(n_dof-1)))
    axs.set_xlabel("Trial")
    axs.set_ylabel("Mean Absolute Error")
    axs.set_title("Evolution of MAE")
    axs.axhline(np.sum(errJointNoBrain)/len(time), color='r', linestyle='--', linewidth=2)
    axs.legend(["Brain Error", "No Brain Error"])
    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_arm" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)
