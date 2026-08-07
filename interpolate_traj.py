# trying to get more points out of the trajectories provided before reverse engineering them
import numpy as np
from experiment_assets import load_trajectory
from create_trajectories import save_data

traj = load_trajectory("trajectories/circleTrajPos.csv")

int_pos = np.zeros((len(traj.position)*2, 3))
int_jpos = np.zeros((len(traj.joint_position)*2, 7))
int_jvel = np.zeros((len(traj.joint_velocity)*2, 7))
int_t    = np.zeros(len(traj.time)*2)

int_pos[0::2] = traj.position
int_jpos[0::2] = traj.joint_position
int_jvel[0::2] = traj.joint_velocity
int_t[0::2] = traj.time

shift = np.copy(traj.position)
shift[:-1] = traj.position[1:]
int_pos[1::2] = (traj.position + shift) / 2

shift = np.copy(traj.joint_position)
shift[:-1] = traj.joint_position[1:]
int_jpos[1::2] = (traj.joint_position + shift) / 2

shift = np.copy(traj.joint_velocity)
shift[:-1] = traj.joint_velocity[1:]
int_jvel[1::2] = (traj.joint_velocity + shift) / 2

shift = np.copy(traj.time)
shift[:-1] = traj.time[1:]
int_t[1::2] = (traj.time + shift) / 2

save_data(fname="trajectories/circleTrajInterp.csv", time=int_t, position=int_pos, joint_position=int_jpos, joint_velocity=int_jvel)