import pickle
from arm_assets_v02 import dynamic_3dof_arm, dynamic_2dof_arm, baxter_reduced
from experiment_assets import Experiment
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


"""
HOW TO WATCH TRAJECTORIES:
1. RUN THIS FILE
2. FOLLOW CLI INSTRUCTIONS
3. Speed multiplier 1 is normal speed
4. trajectory will play on the given url
5. CTRL + L_CLICK to view in vs code

"""
class Trajectory:
    """Lightweight stand-in so traj.pos still works like before."""
    def __init__(self, pos, vel, dt):
        self.pos = pos
        self.vel = vel
        self.dt  = dt
 

def load_trajectories(filename="path_exp.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return {key: Trajectory(v["pos"], v["vel"], v["dt"]) for key, v in data.items()}
 
def play_traj(exp):
    trajectories = load_trajectories(filename=exp.results)
    animate_combined(trajectories[2], trajectories[1], np.arange(len(trajectories[1].pos[0,:]))*trajectories[1].dt)
 
def _make_joystick(ax, title, color):
    """Draw a static joystick background on `ax` and return the animated artists."""
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11)
 
    ax.add_patch(plt.Circle((0, 0), 1.0, color='#333333', fill=False, linewidth=3))
    ax.axhline(0, color='#cccccc', linestyle='--', linewidth=1)
    ax.axvline(0, color='#cccccc', linestyle='--', linewidth=1)
 
    shaft, = ax.plot([], [], color='#888888', linewidth=6, zorder=2)
    knob = plt.Circle((0, 0), 0.15, color=color, zorder=3)
    ax.add_patch(knob)
    readout = ax.text(0.03, 0.97, '', transform=ax.transAxes, fontsize=9,
                      family='monospace', verticalalignment='top')
    return shaft, knob, readout
 
 
def _update_joystick(shaft, knob, readout, vx, vy):
    """Clamp (vx, vy) to the unit circle and move the joystick artists."""
    magnitude = np.hypot(vx, vy)
    if magnitude > 1.0:
        vx, vy = vx / magnitude, vy / magnitude
    shaft.set_data([0, vx], [0, vy])
    knob.center = (vx, vy)
    readout.set_text(f"Vx: {vx:+.2f}\nVy: {vy:+.2f}")
 
 
def _build_figure(all_pos, names):
    """
    Lays out the dot-position panel and the two joysticks. Returns the figure, its title
    and a draw(pos1, pos2, vel1, vel2, frame, start) function that moves every artist,
    tracing the paths from sample `start`.
    """
    # Layout: big position plot on the left, two joysticks stacked on the right.
    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 2, width_ratios=[3, 1], figure=fig,
                  wspace=0.05, hspace=0.15)
    ax_pos = fig.add_subplot(gs[:, 0])
    ax_js1 = fig.add_subplot(gs[0, 1])
    ax_js2 = fig.add_subplot(gs[1, 1])
 
    # ---- Position panel ----
    ax_pos.set_title("Dot Position")
    ax_pos.set_xlabel("X Position")
    ax_pos.set_ylabel("Y Position")
    ax_pos.grid(True)
 
    margin = 1.0
    ax_pos.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax_pos.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    ax_pos.set_aspect('equal', adjustable='box')
 
    c1, c2 = '#d9534f', '#2e8b57'  # actual = red, desired = green
    path1, = ax_pos.plot([], [], '--', color=c1, alpha=0.5, label=f"{names[0]} path")
    dot1, = ax_pos.plot([], [], 'o', color=c1, markersize=8, label=names[0])
    path2, = ax_pos.plot([], [], '--', color=c2, alpha=0.5, label=f"{names[1]} path")
    dot2, = ax_pos.plot([], [], 'o', color=c2, markersize=8, label=names[1])
    ax_pos.legend(loc='best')
 
    # ---- Joysticks ----
    shaft1, knob1, read1 = _make_joystick(ax_js1, f"{names[0]} Input", c1)
    shaft2, knob2, read2 = _make_joystick(ax_js2, f"{names[1]} Input", c2)
 
    artists = (path1, dot1, path2, dot2,
               shaft1, knob1, read1, shaft2, knob2, read2)
 
    def draw(pos1, pos2, vel1, vel2, frame, start=0):
        path1.set_data(pos1[start:frame + 1, 0], pos1[start:frame + 1, 1])
        dot1.set_data([pos1[frame, 0]], [pos1[frame, 1]])
        path2.set_data(pos2[start:frame + 1, 0], pos2[start:frame + 1, 1])
        dot2.set_data([pos2[frame, 0]], [pos2[frame, 1]])
 
        _update_joystick(shaft1, knob1, read1, *vel1[frame])
        _update_joystick(shaft2, knob2, read2, *vel2[frame])
        return artists
 
    return fig, ax_pos, artists, draw
 
 
def animate_combined(actual_traj, desired_traj, time,
                     names=None, fps: int = 30, speed: float = 2,
                     save_path=None):
    """
    Animates the actual vs. desired dot trajectories alongside the actual and
    desired joystick inputs, all in a single synchronized figure.
 
    :param actual_pos:  (N, >=2) array of (x, y) positions for the actual dot.
    :param desired_traj.pos: (N, >=2) array of (x, y) positions for the desired dot.
    :param actual_traj.vel:  (N, >=2) array of (vx, vy) joystick inputs for the actual dot.
    :param desired_traj.vel: (N, >=2) array of (vx, vy) joystick inputs for the desired dot.
    :param time:        Array of timestamps (used to work out the sample rate).
    :param names:       Optional [actual_name, desired_name] for the legend/titles.
    :param fps:         Frames per second for the animation render.
    :param speed:       Playback speed multiplier.
    :param save_path:   Optional path (e.g. 'out.gif' / 'out.mp4') to export.
    :return: the FuncAnimation object (keep a reference so it isn't garbage collected).
    """
    names = list(names) if names else ["Actual", "Desired (50 ms ago)"]
 
    # Downsample everything with the SAME step so all four panels stay in sync.
    sample_rate = 1.0 / (time[1] - time[0])
    step = max(1, int(sample_rate / fps))
 
    pos1 = np.asarray(actual_traj.pos)[::step, :2]
    pos2 = np.asarray(desired_traj.pos)[::step, :2]
    vel1 = np.asarray(actual_traj.vel)[::step, :2]
    vel2 = np.asarray(desired_traj.vel)[::step, :2]
 
    num_frames = min(len(pos1), len(pos2), len(vel1), len(vel2))
    pos1, pos2 = pos1[:num_frames], pos2[:num_frames]
    vel1, vel2 = vel1[:num_frames], vel2[:num_frames]
 
    fig, _, artists, draw = _build_figure(np.vstack((pos1, pos2)), names)
 
    def init():
        return draw(pos1, pos2, vel1, vel2, 0)
 
    def update(frame):
        return draw(pos1, pos2, vel1, vel2, frame)
 
    interval = (1000 / speed) / fps
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                         interval=interval, blit=True, repeat=False)
 
    if save_path:
        anim.save(save_path, fps=fps * speed)
 
    plt.show()
    return anim
 
 
def make_timelapse(clips, time, save_path, names=None, fps: int = 30, speed: float = 5):
    """
    Saves an mp4 of the learning: plays back a handful of trials one after the other,
    the paths retracing from the start of each trial.
 
    :param clips:     list of (trial, mae, actual_traj, desired_traj), one per trial to show.
                      The desired trajectory should already be aligned to the actual one.
    :param time:      Array of timestamps of one trial (used to work out the sample rate).
    :param save_path: where to write the video (e.g. 'results/videos/joystick.mp4').
    :param speed:     Playback speed multiplier of each trial.
    """
    names = list(names) if names else ["Actual", "Desired (50 ms ago)"]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
 
    sample_rate = 1.0 / (time[1] - time[0])
    step = max(1, int(sample_rate * speed / fps))
    data = [(trial, mae,
             np.asarray(act.pos)[::step, :2], np.asarray(des.pos)[::step, :2],
             np.asarray(act.vel)[::step, :2], np.asarray(des.vel)[::step, :2])
            for trial, mae, act, des in clips]
    frames_per_clip = min(len(d[2]) for d in data)
 
    fig, ax_pos, artists, draw = _build_figure(np.vstack([np.vstack((d[2], d[3])) for d in data]), names)
    n_trials = clips[-1][0] + 1
 
    def update(frame):
        trial, mae, pos1, pos2, vel1, vel2 = data[frame // frames_per_clip]
        ax_pos.set_title(f"Trial {trial + 1}/{n_trials}    MAE {mae:.3f}")
        return draw(pos1, pos2, vel1, vel2, frame % frames_per_clip)
 
    anim = FuncAnimation(fig, update, frames=frames_per_clip * len(data), repeat=False)
    print(f"Writing timelapse of {len(data)} trials to {save_path}...")
    anim.save(save_path, fps=fps, writer="ffmpeg")
    plt.close(fig)
