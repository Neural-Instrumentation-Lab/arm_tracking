import pickle
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from arm_assets_v02 import baxter_reduced
from experiment_assets import Experiment

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
    def __init__(self, pos, arm, dt):
        self.pos = pos
        self.arm = arm
        self.dt  = dt
 
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


def load_trajectories(filename="path_exp.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return {key: Trajectory(v["pos"], v["arm"], v["dt"]) for key, v in data.items()}
 
 
def play_traj(exp, fps=30):
    """
    mirrors model_v02.playVideo, but reads the trajectories saved by
    save_trajectories() instead of taking them from a live simulation
    """
    trajectories = load_trajectories(filename=exp.results)
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
                viz.play(traj.pos, traj.dt)
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
        if prevChoice == 0 or traj.arm != trajectories[prevChoice].arm:
            playArm = baxter_reduced(traj.arm, pkgDirs=["./"], disp=False)
            viz = initViz(playArm)
        # time vector isn't saved, so rebuild time[-1] from dt (time starts at 0)
        duration = (len(traj.pos) - 1) * traj.dt
        step = max(1, int(len(traj.pos) / (fps * duration)))
        viz.play(traj.pos[::step], 1/fps)
        prevChoice = userTraj
