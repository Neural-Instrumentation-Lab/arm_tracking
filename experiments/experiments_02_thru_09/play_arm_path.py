import pickle
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from exp_02_09_atrack_assets import simplest_2dof_limb, simplest_2dof_controller, cerebellum_marr_albus, dynamic_3dof_arm, dynamic_2dof_arm, armTraj, angle_diff

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
 
 
if __name__ == "__main__":
    while True:
        experiment = input("select experiment (2-9) to watch:")
        try:
            experiment = int(experiment)
        except ValueError:
            print("not valid input")
            continue
        if experiment > 9 or experiment < 2:
            print("not valid input")
            continue
        break
    while True:
        speed = input("speed multiplier (smaller is faster):")
        try:
            speed = float(speed)
        except ValueError:
            print("not valid input")
            continue
        break

    trajectories = load_trajectories(filename="experiments/experiments_02_thru_09/results/arm_paths/path_exp_0"+str(experiment)+".pkl")
 
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
                viz.play(traj.pos, traj.dt*speed)
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
            if traj.pos.shape[1] == 2:
                arm         = dynamic_2dof_arm("models/"+traj.arm, disp=False) 
            if traj.pos.shape[1] == 3:
                arm         = dynamic_3dof_arm("models/"+traj.arm, disp=False) 
            viz = initViz(arm)
        viz.play(traj.pos, traj.dt*speed)
        prevChoice = userTraj
 
