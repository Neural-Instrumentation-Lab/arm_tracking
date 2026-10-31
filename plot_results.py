import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from arm_assets_v02 import armTraj

def plot_arm_results(final_traj, desired_traj, show=True, save=False, saveLoc=None):
    # velocities of final trajectory
    fig, axs = plt.subplots()
    axs.plot(final_traj.vel)
    axs.plot(desired_traj.vel)
    axs.legend(["x actual", "y actual", "x des", "y des"])
    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_commands" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)

def plot_brain_results(errorTot, time, brain, show=True, save=False, saveLoc=None):
    # MAE evolution graph
    fig, axs = plt.subplots()
    axs.plot(errorTot/(len(time)*(brain.nJoints-1)))
    axs.set_xlabel("Trial")
    axs.set_ylabel("Mean Absolute Error")
    axs.set_title("Evolution of MAE")
    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_MAE" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # PF-PC Weights
    wts = brain.pf_pc_wts
    plt.imshow(wts, cmap='viridis', interpolation='nearest', aspect="auto")
    plt.xlabel("PCs")
    plt.ylabel("PFs")
    plt.axvline(49.5, color='red', linestyle='--')
    plt.axvline(99.5, color='red', linestyle='--')
    plt.axvline(149.5, color='red', linestyle='--')
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
    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_pf_pc_wts" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)

def plot_dcn_raster(brainResults, time, brain, brain_dt, reset=True, show=True, save=False, saveLoc=None):
    """
    Raster of every DCN's spikes over one whole trial, at the first, middle and final trial.

    brainResults.dcn is recorded every sim step but the brain only updates every brain_dt,
    so only the steps where the brain ticked are plotted (same timing as runSimulation).
    """
    dcn = brainResults.dcn
    n_trials, n_steps, n_dcn = dcn.shape
    step = int(round(n_steps / (np.floor(time[-1] / brain_dt) + 1), 0))
    trials = sorted({0, n_trials // 2, n_trials - 1})

    fig, axs = plt.subplots(len(trials), 1, sharex=True, sharey=True, squeeze=False,
                            figsize=(9, 2.6 * len(trials)), layout="constrained")
    n_groups = brain.nJoints * 2
    per_group = n_dcn // n_groups
    for ax, trial in zip(axs[:, 0], trials):
        # brain-tick steps of this trial: runSimulation counts steps since the last reset
        offset = 0 if reset else (-trial * n_steps) % step
        ticks = np.arange(offset, n_steps, step)
        t_idx, neuron = np.nonzero(dcn[trial, ticks])
        ax.scatter(time[ticks[t_idx]], neuron, s=2, marker="|", linewidths=0.6, color="0.15", rasterized=True)
        for g in range(1, n_groups):
            ax.axhline(g * per_group - 0.5, color="0.75", linewidth=0.8)
        rate = dcn[trial, ticks].sum() / (n_dcn * len(ticks) * brain_dt)
        ax.set_title(f"Trial {trial + 1} of {n_trials}   (mean rate {rate:.1f} Hz)", loc="left", fontsize=10)
        ax.set_ylabel("DCN")
        ax.set_ylim(-0.5, n_dcn - 0.5)
        ax.set_yticks([(g + 0.5) * per_group for g in range(n_groups)])
        ax.set_yticklabels([f"J{g // 2 + 1} {'agonist' if g % 2 == 0 else 'antagonist'}" for g in range(n_groups)],
                           fontsize=8)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axs[-1, 0].set_xlabel("Time in trial (s)")
    axs[-1, 0].set_xlim(time[0], time[-1])
    fig.suptitle("DCN spike raster")

    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_dcn_raster" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)
