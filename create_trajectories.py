import numpy as np
import argparse

def linspace(a,b,dx):
    nPts = int( (b-a)/dx + 1 )
    x = [a + dx*i for i in range(nPts)]
    return x

def get_trajectory_list():
    str = '\n'
    str += '0   : single point located at (7.07, 12.07)\n'
    str += '1   : horizontal line at y = 12\n'
    return str

def build_array(t,x,y):
    return np.array([t,x,y]).T

def save_data(fname,array):
    np.savetxt(fname , array , delimiter=',' , fmt="%0.3f")

def create_traj_000():
    t = [0]
    x = [7.07]
    y = [12.07]
    array = build_array(t,x,y)
    fname = 'trajectories/traj_000.csv'
    save_data(fname,array)
 
def create_traj_001():
    dx = 0.01
    x = linspace(-8,8,dx)
    y = [12 for _ in range(len(x))]

    dt = 0.01
    t = [0+dt*i for i in range(len(x))]
    array = build_array(t,x,y)
    fname = 'trajectories/traj_001.csv'
    save_data(fname,array)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_id' , help='trajectory id list' )
    parser.add_argument('--list'          , help='list all trajectories' , action="store_true")
    args = parser.parse_args()
    
    if args.list:
        print( get_trajectory_list() )
    
    if args.trajectory_id == '0'  : create_traj_000()
    elif args.trajectory_id == '1': create_traj_001()


if __name__ == "__main__":
    main()