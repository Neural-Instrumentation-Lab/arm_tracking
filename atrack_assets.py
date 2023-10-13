import numpy as np
from numpy import sin, cos, arccos

class JointAngleError(Exception): pass

class simplest_2dof_controller:
    '''
    Control unit -> converts desired reach locations into joint angles
    '''
    def __init__(self,L1=None,L2=None):
        if not L1:
            self.len_1 , self.len_2 = 10 , 5
        else:
            self.len_1 , self.len_2 = L1 , L2

    def get_joint_angles(self,pos):
        x,y   = pos[0] , pos[1]
        L1,L2 = self.len_1 , self.len_2

        # there is only a solution possible if |L1-L2| < mag(x,y) < L1+L2
        if not np.abs(L1-L2) < np.sqrt(x**2 + y**2) < (L1+L2):
            raise JointAngleError("Trying to reach to an unreachable location")

        th2 = np.arccos( ( (x**2+y**2) - (L1**2+L2**2) ) / (2*L1*L2) )       
        th1 = arccos( ( (L1+L2*cos(th2))*x + L2*sin(th2)*y ) / (x**2+y**2) )

        return np.array([th1,th2])

class simplest_2dof_limb:
    '''
    Model of a physical 2DOF arm. No dynamics, just kinematics
    '''
    def __init__(self , L1=None , L2=None):
        if not L1:
            self.len_1 , self.len_2 = 10 , 5
        else:
            self.len_1 , self.len_2 = L1 , L2

    def move(self,joint_angles):
        x = self.len_1*np.cos(joint_angles[0]) + self.len_2*np.cos(joint_angles[0] + joint_angles[1])
        y = self.len_1*np.sin(joint_angles[0]) + self.len_2*np.sin(joint_angles[0] + joint_angles[1])

        return np.array([x,y])

class cerebellum:
    '''
    Marr-Albus cerebellum
    '''
    def __init__(self):
        self.n_dims = 2
        pass
    def update(self,movement_error , joint_angles):
        pass
    def compute_correction(self,joint_angles):
        return np.zeros(self.n_dims)
