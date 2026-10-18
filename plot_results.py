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