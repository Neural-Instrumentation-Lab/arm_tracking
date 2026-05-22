'''
ATRACK (ARM TRACK) ASSETS
A group of classes for use in cerebellar arm-tracking experiments

@Author: Iyad Obeid
@Date: Fall 2023
'''

## imports #####################################################################
import numpy as np
from numpy import sin, cos, arccos, exp
from itertools import product as combine
import pinocchio as pin
from scipy.optimize import fmin_bfgs
## create a Joint Angle Error ##################################################
class JointAngleError(Exception): pass

## Simple 2 DoF Motor Controller ###############################################
class simplest_2dof_controller:
    '''
    Control unit - converts desired reach locations into joint angles
        Written as a class for implementation simplicty
    '''
    ###################################
    def __init__(self,L1=None,L2=None):
    ###################################
        '''Constructor - load in values for L1,L2 or use defaults'''
        self.len_1,self.len_2 = [10,5] if not L1 else [L1,L2]

    ###################################
    def is_valid_location(self,pos):
    ###################################
        '''Determines whether location <pos> can be reached by the arm
        
        Args:
            pos (array of 2 elements)   desired arm location in 2D
        Returns:
            valid (boolean)             True if <pos> can be reached by the arm
        '''

        # define local copies for notational simplicity
        x,y   = pos[0] , pos[1]
        L1,L2 = self.len_1 , self.len_2

        # return true if pos can be physically reached
        return np.abs(L1-L2) < np.sqrt(x**2 + y**2) < (L1+L2)

    ###################################
    def get_joint_angles(self,pos):
    ###################################
        '''Given a desired position in x-y space, return the corresponding arm angles.
            Note that desired position must be greater than L1-L2 and less than L1+L2

            Args:
                pos (array of 2 elements)   desired arm location in 2D
            Returns:
                [th1,th2]                   corresponding joint angles in radians
        '''

        # define local copies for notational simplicity
        x,y   = pos[0] , pos[1]
        L1,L2 = self.len_1 , self.len_2

        # if <pos> cannot be physically reached by the arm, raise a Joint Angle Error and exit
        if not self.is_valid_location(pos):
            raise JointAngleError("Trying to reach to an unreachable location")

        #  Compute the values for th1 and th2
        th2 = arccos( ( (x**2+y**2) - (L1**2+L2**2) ) / (2*L1*L2) )       
        th1 = arccos( ( (L1+L2*cos(th2))*x + L2*sin(th2)*y ) / (x**2+y**2) )

        #  return th1, th2
        return np.array([th1,th2])

## Simple 2 DoF Limb ###########################################################
class simplest_2dof_limb:
    '''
    Model of a physical 2DOF arm. No dynamics, just kinematics
        Basically just a single function but written as a class for simplicity of implementation
    '''

    ###################################
    def __init__(self,L1=None,L2=None):
    ###################################
        '''Constructor - load in values for L1,L2 or use defaults'''
        self.len_1,self.len_2 = [10,5] if not L1 else [L1,L2]

    ###################################
    def move(self,joint_angles):
    ###################################
        ''' Determine x-y locations for the arm given the joint angles
            
            Args:
                joint_nagles (array of 2 elements)  arm joint angles
            Returns:
                [x,y]                               arm location
        '''

        # compute x-y locations
        x = self.len_1*np.cos(joint_angles[0]) + self.len_2*np.cos(joint_angles[0] + joint_angles[1])
        y = self.len_1*np.sin(joint_angles[0]) + self.len_2*np.sin(joint_angles[0] + joint_angles[1])

        # return x-y locations
        return np.array([x,y])

class dynamic_3dof_arm:
    '''
    3dof arm with pinocchio; rigid body sim library
    data from arm from Garrdio 2013 (3 dof)
    Functions mostly taken from pinocchio docs: 
    '''
    def __init__(self, filename, disp=True):
        '''
        creates arm from file.
        this arm is constructed from the tables in the Garrido 2013 paper. (Appendix B)

        Args:
            filename: urdf filepath describing the arm config.
            disp (optional): if true prints out the neutral configuration of the arm
        '''
        # initializes from urdf file
        # https://gepetto.github.io/doc/pinocchio/doxygen-html/md_doc_b-examples_a-model.html
        self.model, self.collModel, self.visualModel = pin.buildModelsFromUrdf(filename)
        self.data = self.model.createData()
        self.njoints = self.model.njoints - 1 # end effector counts a joint
        self.lengths = [.31, .4, .39]
        
        # end effector data
        self.eeDes = np.array([0, 0, 0]) 
        self.eeId = self.model.getFrameId("ee_fixed_joint")
        
        # Motor inertias, urdf files don't support these natively for some reason
        self.model.armature = np.array([361.6*10**-6, 415.5*10**-6, 415.5*10**-6]) 
        
        if disp:
            # print neutral config
            q = pin.neutral(self.model)
            print(f"q: {q.T}")
            # Perform the forward kinematics over the kinematic tree
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            # Print out the placement of each joint of the kinematic tree
            for name, oMi in zip(self.model.names, self.data.oMi):
                print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

    def forward(self, position, velocity, torque):
        '''
        computes joint acceleration 

        Args:
            position: joint positions
            velocity: joint velocities
            torque: torque vector to apply to the joints
        
        Returns:
            accel: joint accelerations
        '''
        accel = pin.aba(self.model, self.data, position, velocity, torque)
        return accel

    def move(self, pos, vel, accel):
        '''
        updates the model joints and frames.
        
        Args:
            pos: joint positions
            vel: joint velocities
            accel: joint accelerations
        '''
        pin.forwardKinematics(self.model, self.data, pos, vel, accel)
        pin.updateFramePlacements(self.model, self.data)

    def inverse(self, position, velocity, accel):
        '''
        computes the torque of a motion
        
        Args:
            position: joint positions
            velocity: joint velocities
            acceleration: joint accelerations
        
        Returns:
            torque: joint torques
        '''
        torque = pin.rnea(self.model, self.data, position, velocity, accel)
        return torque

    def getPos(self):
        '''
        Returns:
            end-effector position in 3d space. (x, y, z)
        '''
        return self.data.oMf[self.eeId].translation
    
    def eeError(self, joint):
        '''
        Args:
            joint: joint configuration 

        Returns:
            distance between EE pos in input config and desired EE pos
        '''
        # returns the difference between the desired EE pos
        # and the EE pos in a joint config given by input arg
        pin.forwardKinematics(self.model, self.data, joint)
        pin.updateFramePlacements(self.model, self.data)
        eePos = self.data.oMf[self.eeId].translation
        return np.linalg.norm(eePos - self.eeDes) 

    def getRandomConf(self):
        '''
        retrieve random joint config
        '''
        return pin.randomConfiguration(self.model)

    def getEEFromJoint(self, pos):
        # returns the end effector position [x, y, z]
        # from a joint config vector
        pin.forwardKinematics(self.model, self.data, pos)
        pin.updateFramePlacements(self.model, self.data)
        return self.getPos()

    def getJointPosFromEE_not_iterative(self, pos):
        # doesn't work yet because of base height and other things
        # https://www.atlantis-press.com/article/126003627.pdf
        x, y, z = pos
        r = np.sqrt(x**2 + y**2 + z**2)
        L1, L2, L3 = self.lengths
        joint1 = np.atan(y / x)
        print((x**2 + y**2 + z**2 - (L1**2 * L2**2)) / (2*L1*L2))
        joint3 = -np.acos( (x**2 + y**2 + z**2 - (L1**2 * L2**2)) / (2*L1*L2) )
        joint2 = np.asin(z / r) + np.atan( (L2 * sin(joint3)) / (L1 + L2*np.cos(joint3)) )
        return np.array([joint1, joint2, joint3])


    def getJointPosFromEE(self, pos, prevPos = np.array([])):
        '''
        Args:
            pos: joint position
            prevPos (optional): previous joint position. Defaults to model neutral.

        Returns:
            a possible joint configuration, minimzing movement between pos and prevPos
        
        LOOK INTO DOING THIS NON-ITERATIVELY FOR A 3DOF ARM BECAUSE THATS POSSIBLE
        AND SAVES A BUNCH OF TIME (METHOD ABOVE)
        
        '''
        # https://gepetto.github.io/doc/pinocchio/doxygen-html/md_doc_b-examples_d-inverse-kinematics.html 
        
        if prevPos.size == 0:
            prevPos = pin.neutral(self.model)
        self.eeDes = pos.copy()

        # these samples were chosen randomly and without care, should be better
        # the idea is: find a possible spot from a number of samples
        # then find the configuration that is the closest to the previous
        # if you see the sim "jump" in the trajectory
        # there should be another sample near to where the sim jumped
        sample_spots = np.array([[0, 0, 0],
                                [2.18, -1.38, 1.67],
                                [-2.97, 1.76, -.05],
                                [0, -2.33, -1.71],
                                [1.1, -4.01, -1.59],
                                [-1.63, -3, 0.68],
                                [-.19, -.19, 0.8],
                                [.78, 0, 0]])
        possibConfigs= []
        for sample in sample_spots:
            qDes, fopt, _, _, _, _, _ = fmin_bfgs(self.eeError, sample, full_output=True, disp=False)
            if fopt < 0.01:
                possibConfigs.append(qDes)
        # if no config within a cm of the desired, position is unreachable.
        if not possibConfigs:
            raise JointAngleError("Point not reachable by arm")
        dists = list(map(lambda x: np.linalg.norm(x - prevPos), possibConfigs))
        return possibConfigs[dists.index(min(dists))]

class dynamic_2dof_arm(dynamic_3dof_arm):
    def is_valid_location(self, pos):
        x,z   = pos[0] , pos[2]
        L1,L2,L3 = self.lengths 

        # return true if pos can be physically reached
        return np.abs(L2-L3) < np.sqrt(x**2 + (z-L1)**2) < (L2+L3)


    def getJointPosFromEE(self, pos, prevPos = np.array([])):
        x,z   = pos[0] , pos[2]
        L1,L2,L3 = self.lengths

        # if <pos> cannot be physically reached by the arm, raise a Joint Angle Error and exit
        if not self.is_valid_location(pos):
            raise JointAngleError("Trying to reach to an unreachable location")

        #  Compute the values for th1 and th2
        th2 = arccos( ( (x**2+(z-L1)**2) - (L2**2+L3**2) ) / (2*L2*L3) )
        # have to use different derivation because quadrant info
        # is weird when converting to the arm angles
        th1 = np.atan2(z-L1, x) - np.atan2(L3*sin(th2), L2 + L3*cos(th2)) 

        # joints's zero are on the y-axis not x-axis
        return np.array([(np.pi/2) - th1, -th2])

    def __init__(self, filename, lengths=[.31, .4, .39], disp=True):
        '''
        creates arm from file.
        this arm is constructed from the tables in the Garrido 2013 paper. (Appendix B)

        Args:
            filename: urdf filepath describing the arm config.
            disp (optional): if true prints out the neutral configuration of the arm
        '''
        # initializes from urdf file
        # https://gepetto.github.io/doc/pinocchio/doxygen-html/md_doc_b-examples_a-model.html
        self.model, self.collModel, self.visualModel = pin.buildModelsFromUrdf(filename)
        self.data = self.model.createData()
        self.njoints = self.model.njoints - 1 # end effector counts a joint
        self.lengths = lengths 
        
        # end effector data
        self.eeDes = np.array([0, 0, 0]) 
        self.eeId = self.model.getFrameId("ee_fixed_joint")
        
        # Motor inertias, urdf files don't support these natively for some reason
        self.model.armature = np.array([415.5*10**-6, 415.5*10**-6]) 
        
        if disp:
            # print neutral config
            q = pin.neutral(self.model)
            print(f"q: {q.T}")
            # Perform the forward kinematics over the kinematic tree
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            # Print out the placement of each joint of the kinematic tree
            for name, oMi in zip(self.model.names, self.data.oMi):
                print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

## Radial Basis Function *######################################################
class rbf:
    '''
    Radial Basis Function
        Returns activation strength of a given RBF cell
        Currently a 2D RBF
    '''
    ###################################
    def __init__(self , c , sigma):
    ###################################
        ''' Constructor '''
        self.c , self.sigma = c,sigma
    
    ###################################
    # slow, should implement in c++
    def compute(self,x):
        term0 = (x[0]-self.c[0])**2
        term1 = (x[1]-self.c[1])**2
        term2 = -(term0 + term1)/(2*self.sigma**2)
        activation = exp(term2)
        return activation

## Marr Albus Cerebellum #######################################################
class cerebellum_marr_albus:
    '''
    Marr-Albus cerebellum
    Still needs to be written
    '''

    ###################################
    def __init__(self):
    ###################################
        self.n_dims = 2

        # set up an array of RBFs over the angle space
        self.rbfs          = []
        self.wts           = []
        self.beta          = 0.0005 # learning rate
        d_x = 1
        sigma              = 2.5*d_x # biggest_d_btw_ctrs / np.sqrt(2 * self.n_cerebellums) # spread parameter

        for ctr in combine( np.arange(-15,15,d_x) , np.arange(-15,15,d_x) ):
            self.rbfs.append( rbf(ctr,sigma) )
            self.wts.append([0,0])
        self.n_cerebellums = len(self.rbfs) 

        self.activations = [0 for _ in range(self.n_cerebellums)]

    ###################################
    # slow
    def update(self,movement_error):
    ###################################
        for j in range(self.n_cerebellums):
            self.wts[j][0] -= self.beta * movement_error[0]*self.activations[j]
            self.wts[j][1] -= self.beta * movement_error[1]*self.activations[j]

    ###################################
    # slow
    def compute_correction(self,movement_error):
    ###################################

        corr_0 , corr_1 = 0,0
        i = 0
        for w,rbf in zip(self.wts,self.rbfs):
            activation = rbf.compute(movement_error)
            corr_0 += w[0]*activation
            corr_1 += w[1]*activation
            self.activations[i] = activation
            i += 1

        return [corr_0,corr_1]
