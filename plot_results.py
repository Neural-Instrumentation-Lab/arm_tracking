import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.colors import ListedColormap
from arm_assets_v02 import armTraj
from bmi import DCN_ROLES

# every MAE plot uses this y-axis so experiments can be compared by eye
MAE_YLIM = (0, 1.5)

def apply_mae_ylim(ax, mae):
    ax.set_ylim(*MAE_YLIM)
    if np.max(mae) > MAE_YLIM[1]:
        print(f"warning: MAE reaches {np.max(mae):.2f}, above the plot's fixed limit of {MAE_YLIM[1]}")

# colours for how the BMI reads each DCN (bmi.DCN_ROLES): its own muscle, another muscle, both, neither
ROLE_COLORS = {"right": "#1f5fbf", "wrong": "#d62728", "both": "#9467bd", "unread": "#b0b0b0"}
ROLE_LABELS = {"right": "read as own muscle", "wrong": "read as another muscle",
               "both": "read as own and another", "unread": "not read"}


def _present_roles(roles):
    """(index, name) of every role some DCN has."""
    return [(i, name) for i, name in enumerate(DCN_ROLES) if np.any(roles == i)]


def _trial_ticks(brainResults, trial):
    """(DCN spikes, sim step) of every brain tick of one trial, from runSimulation's brainResults."""
    steps = brainResults.tick_step[trial]
    used = steps >= 0
    return brainResults.dcn[trial, used], steps[used]


def plot_arm_results(final_traj, desired_traj, show=True, save=False, saveLoc=None):
    # velocities of final trajectory
    fig, axs = plt.subplots()
    axs.plot(final_traj.vel)
    axs.plot(desired_traj.vel)
    axs.legend(["x actual", "y actual", "x des (50 ms ago)", "y des (50 ms ago)"])
    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_commands" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)

def plot_brain_results(errorTot, time, brain, show=True, save=False, saveLoc=None, roles=None):
    """
    MAE over trials, and the final PF-PC weights. roles (from bmi.dcn_roles(), optional) adds a strip
    above the weights colouring each PC by how the BMI reads the DCN it inhibits.
    """
    # MAE evolution graph
    fig, axs = plt.subplots()
    mae = errorTot/(len(time)*(brain.nJoints-1))
    axs.plot(mae)
    apply_mae_ylim(axs, mae)
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
    n_pc = wts.shape[1]
    per_group = n_pc // (brain.nJoints * 2)
    if roles is None:
        fig, ax = plt.subplots()
    else:
        fig, (strip, ax) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5.2),
                                        gridspec_kw={"height_ratios": [1, 14], "hspace": 0.05})
        strip.imshow(np.asarray(roles)[None, :], cmap=ListedColormap([ROLE_COLORS[r] for r in DCN_ROLES]),
                     vmin=-0.5, vmax=len(DCN_ROLES) - 0.5, interpolation="nearest", aspect="auto")
        strip.set_yticks([])
        strip.set_title("PF-PC weights (strip: how the BMI reads each PC's DCN)", fontsize=9, loc="left")
        handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLORS[name]) for _, name in _present_roles(roles)]
        labels = [f"{ROLE_LABELS[name]}: mean {wts[:, roles == i].mean():.3g}"
                  for i, name in _present_roles(roles)]
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.45, -0.1))
    image = ax.imshow(wts, cmap='viridis', interpolation='nearest', aspect="auto")
    ax.set_xlabel("PCs", labelpad=16)
    ax.set_ylabel("PFs")
    for g in range(1, brain.nJoints * 2):
        ax.axvline(g * per_group - 0.5, color='red', linestyle='--')
    fig.colorbar(image, ax=[strip, ax] if roles is not None else ax, label='Conductance (S)')
    labels = [f"Joint {g // 2 + 1} {'Agonist' if g % 2 == 0 else 'Antagonist'}" for g in range(brain.nJoints * 2)]
    for g, label in enumerate(labels):
        ax.text((g + 0.5) * per_group, -0.075, label, ha='center', va='top', fontsize=8,
                transform=ax.get_xaxis_transform())
    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saveLoc.with_name(saveLoc.stem + "_pf_pc_wts" + saveLoc.suffix), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_dcn_raster(brainResults, time, brain, brain_dt, show=True, save=False, saveLoc=None, roles=None):
    """
    Raster of every DCN's spikes over one whole trial, at the first, middle and final trial.

    brainResults.dcn holds the spikes of every brain tick, and tick_step the sim step of each.
    roles (from bmi.dcn_roles(), optional) colours each DCN by how the BMI reads it.
    """
    n_trials, _, n_dcn = brainResults.dcn.shape
    trials = sorted({0, n_trials // 2, n_trials - 1})

    fig, axs = plt.subplots(len(trials), 1, sharex=True, sharey=True, squeeze=False,
                            figsize=(9, 2.6 * len(trials)), layout="constrained")
    n_groups = brain.nJoints * 2
    per_group = n_dcn // n_groups
    for ax, trial in zip(axs[:, 0], trials):
        spikes, ticks = _trial_ticks(brainResults, trial)
        t_idx, neuron = np.nonzero(spikes)
        rate = spikes.sum() / (n_dcn * len(ticks) * brain_dt)
        title = f"Trial {trial + 1} of {n_trials}   (mean rate {rate:.1f} Hz"
        if roles is None:
            ax.scatter(time[ticks[t_idx]], neuron, s=2, marker="|", linewidths=0.6, color="0.15", rasterized=True)
        else:
            for i, name in _present_roles(roles):
                mine = roles[neuron] == i
                ax.scatter(time[ticks[t_idx[mine]]], neuron[mine], s=2, marker="|", linewidths=0.6,
                           color=ROLE_COLORS[name], rasterized=True)
                title += f"; {name} {spikes[:, roles == i].mean() / brain_dt:.0f} Hz"
        ax.set_title(title + ")", loc="left", fontsize=9)
        for g in range(1, n_groups):
            ax.axhline(g * per_group - 0.5, color="0.75", linewidth=0.8)
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
    fig.suptitle("DCN spike raster" + ("" if roles is None else ", coloured by how the BMI reads each DCN"))
    if roles is not None:
        handles = [plt.Line2D([], [], color=ROLE_COLORS[name], marker="|", linestyle="", markersize=8)
                   for _, name in _present_roles(roles)]
        fig.legend(handles, [ROLE_LABELS[name] for _, name in _present_roles(roles)], loc="outside lower center",
                   ncol=4, fontsize=8, frameon=False)

    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(saveLoc.with_name(saveLoc.stem + "_dcn_raster" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_dcn_roles(brainResults, time, brain_dt, roles, show=True, save=False, saveLoc=None):
    """
    Whether the brain learns to quiet the DCNs the BMI misreads: per trial, the mean firing rate of
    the DCNs in each role (bmi.dcn_roles()), and the mean PF-PC weight onto the PCs that inhibit
    them (higher weight -> more PC firing -> more inhibition of that DCN).
    """
    n_trials = brainResults.dcn.shape[0]
    present = _present_roles(roles)
    rates = np.zeros((len(present), n_trials))
    for trial in range(n_trials):
        spikes, _ = _trial_ticks(brainResults, trial)
        for k, (i, _) in enumerate(present):
            rates[k, trial] = spikes[:, roles == i].mean() / brain_dt
    wts = np.array([[brainResults.pf_pc[:, :, roles == i].mean(axis=(1, 2))] for i, _ in present])[:, 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6), layout="constrained")
    trials = np.arange(1, n_trials + 1)
    for k, (i, name) in enumerate(present):
        label = f"{ROLE_LABELS[name]} ({np.sum(roles == i)} DCNs)"
        ax1.plot(trials, rates[k], color=ROLE_COLORS[name], label=label)
        ax2.plot(trials, wts[k], color=ROLE_COLORS[name], label=label)
    ax1.set_ylabel("Mean DCN rate (Hz)")
    ax1.set_title("DCN firing by how the BMI reads them", loc="left", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)
    ax2.set_ylabel("Mean PF-PC weight onto their PCs (S)")
    ax2.set_xlabel("Trial")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if save:
        saveLoc.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saveLoc.with_name(saveLoc.stem + "_dcn_roles" + saveLoc.suffix), dpi=300)
    if show:
        plt.show()
    plt.close(fig)
