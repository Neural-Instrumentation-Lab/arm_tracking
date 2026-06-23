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

# makes things look prettier
class armTraj:
    def __init__(self, pos=[], vel=[], accel=[], torq=[], time=[], eePos = []):
        self.pos   = pos
        self.vel   = vel
        self.acel  = accel
        self.torq  = torq
        self.time  = time
        self.eePos = eePos

def angle_diff(a, b):
    """
    Smallest signed difference between angles a and b.
    Result is in [-pi, pi].
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def costHelper(q, prevQ):
    d = np.array([
        angle_diff(q[0], prevQ[0]),
        angle_diff(q[1], prevQ[1]),
        angle_diff(q[2], prevQ[2]),
    ])
    return np.linalg.norm(d)

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
        self.eeDes = np.array([0, 0, 0]) 
        self.eeId = self.model.getFrameId("ee_fixed_joint")

        # calculate limb lengths
        pin.framesForwardKinematics(self.model, self.data, pin.neutral(self.model))
        l1 = np.linalg.norm(self.model.jointPlacements[2].translation)
        l2 = np.linalg.norm(self.model.jointPlacements[3].translation)
        l3 = np.linalg.norm(self.data.oMf[self.eeId].translation - self.data.oMi[3].translation)
        self.lengths = [l1, l2, l3] 
        # end effector data
        
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

    def getJointPosFromEE(self, pos, prevQ=np.array([0, 0, 0])):
        x, y, z = pos
        L1, L2, L3 = self.lengths

        D = np.linalg.norm([x, y, z-L1])
        if D > L1 + L2 or D < np.abs(L1-L2):
            raise JointAngleError(f"Trying to reach to an unreachable location: {pos}")
    
        rho = np.hypot(x, y)
        h = z - L1

        r2 = rho**2 + h**2

        c3 = (r2 - L2**2 - L3**2) / (2 * L2 * L3)
        c3 = np.clip(c3, -1.0, 1.0)

        q3a = np.arccos(c3)      # elbow-up
        q3b = -np.arccos(c3)   # elbow-down

        if rho < 1e-6:
            q1 = prevQ[0]
        else:
            q1 = np.atan2(y, x)

        q2a = np.atan2(h, rho) - np.atan2(
            L3 * np.sin(q3a),
            L2 + L3 * np.cos(q3a)
        )

        q2b = np.atan2(h, rho) - np.atan2(
            L3 * np.sin(q3b),
            L2 + L3 * np.cos(q3b)
        )
        sola = [q1, np.pi/2 - q2a, -q3a]
        solb = [q1, np.pi/2 - q2b, -q3b]
        solc = [q1+np.pi, -(np.pi/2 - q2b), q3b]
        sold = [q1-np.pi, -(np.pi/2 - q2a), q3a]

        sol = min([sola, solb, solc, sold], key=lambda q: costHelper(q, prevQ))

        return np.array(sol)

        # r = np.sqrt(x**2 + y**2 + (z-L1)**2)
        # joint1 = np.atan2(y,x)
        # joint3 = -np.acos( (x**2 + y**2 + (z-L1)**2 - L2**2 * L3**2) / (2*L2*L3))
        # joint2 = np.asin((z-L1) / r) + np.atan2(L2*np.sin(joint3), (L2 + L3*np.cos(joint3)))
        # return np.array([joint1, joint2, joint3])

    def forwardDynamics(self, positions, velocities, torques, starting_ee_pos, time):
        currPos       = self.getJointPosFromEE(starting_ee_pos) 
        currVel       = np.zeros_like(currPos)
        currAcc       = np.zeros_like(currPos)
        torrPd        = np.zeros_like(currPos)
        cntrl_traj    = armTraj(pos=np.zeros((len(torques), self.njoints)),
                                torq=np.zeros((len(torques), self.njoints)),
                                eePos=np.zeros((len(torques), 3)))
        kp = np.ones(self.njoints)*20 
        kd = 2*np.sqrt(kp)
        for i, torque in enumerate(torques):
            # move the arm
            torrPd = kp*(positions[i] - currPos) + kd*(velocities[i] - currVel)
            if i > 0:
                dt      = time[i] - time[i-1]
                currVel = (currVel + currAcc *dt)
                currPos = pin.integrate(self.model, currPos, currVel * dt)
            currAcc = self.forward(currPos, currVel, torque + torrPd)
            cntrl_traj.torq[i,:] = torque + torrPd 
            self.move(currPos, currVel, currAcc)
            # PD-control
            # store results
            cntrl_traj.eePos[i,:] = self.getPos()
            cntrl_traj.pos[i,:]   = currPos
        return cntrl_traj

    def inverseDynamics(self, positions, velocities, accels, time):
        '''
        creates the ideal torques along a given trajectory
        filtered through a PD controller to reduce numerical error
        through multiple integrations

        Args:
            positions: joint positions 
            velocities: joint velocities
            accels: joint accelerations
            time: time vector
        
        Returns:
            torques
        '''
        torques   = np.zeros_like(positions)
        torquesPD = np.zeros_like(torques)
        eePos     = np.zeros((len(positions), 3))
        kp        = np.ones(self.njoints) * 20 # found empirically, these seem ok
        kd        = 2*np.sqrt(kp) # this is the best the ratio for a reason
        pos       = positions[0]  
        vel       = np.zeros_like(pos)
        acc       = np.zeros_like(pos)
        for i, (posCorr, velCorr, accCorr) in enumerate(zip(positions, velocities, accels)):
            torques[i, :]   = self.inverse(posCorr, velCorr, accCorr) # ideal torque
            torquesPD[i, :] = torques[i, :] + kp*(posCorr - pos) + kd*(velCorr - vel)
            acc             = self.forward(pos, vel, torquesPD[i, :])
            if i > 0:
                dt  = time[i] - time[i-1]
                vel = vel + acc * dt 
                pos = pin.integrate(self.model, pos, vel*dt)
            self.move(pos, vel, acc)
            eePos[i,:] = self.getPos()
        return torquesPD, eePos

class dynamic_2dof_arm(dynamic_3dof_arm):
    def is_valid_location(self, pos):
        x,z   = pos[0] , pos[2]
        L1,L2,L3 = self.lengths 

        # return true if pos can be physically reached
        return np.abs(L2-L3) < np.sqrt(x**2 + (z-L1)**2) < (L2+L3)


    def getJointPosFromEE(self, pos, prevQ = np.array([])):
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
        
        self.eeDes = np.array([0, 0, 0]) 
        self.eeId = self.model.getFrameId("ee_fixed_joint")

        # compute joint lengths for IK
        pin.framesForwardKinematics(self.model, self.data, pin.neutral(self.model))
        l1 = np.linalg.norm(self.model.jointPlacements[1].translation)
        l2 = np.linalg.norm(self.model.jointPlacements[2].translation)
        l3 = np.linalg.norm(self.data.oMf[self.eeId].translation - self.data.oMi[2].translation)
        self.lengths = [l1, l2, l3] 
        # end effector data
        
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

class cerebellum_marr_albus:
    '''
    Marr-Albus cerebellum
    Still needs to be written
    '''

    ###################################
    def __init__(self, n_dims=2):
    ###################################
        self.n_dims = n_dims

        # set up an array of RBFs over the angle space
        self.wts           = []
        self.wtsVel        = []
        if n_dims == 3:
            self.beta = 0.01
        else:
            self.beta          = 0.05 # learning rate
        self.damp          = 0.010 # dampening factor on weight update
        self.epsilon       = 0.001 # lowest error to still update weights
        posMax             = 5
        velMax             = 50
        d_x                = posMax * 0.5
        d_v                = velMax * 0.5
        sigma              = 3 * np.linalg.norm([posMax*2 for _ in range(n_dims)] + [velMax*2 for _ in range(n_dims)]) / np.sqrt(2 * (2*velMax/d_v)**(n_dims) * ((2*posMax)/d_x)**(n_dims)) 

        gridPnts = []
        for _ in range(n_dims):
            gridPnts.append(np.arange(-posMax, posMax, d_x))

        for _ in range(n_dims):
            gridPnts.append(np.arange(-velMax, velMax, d_v))

        centers = []
        for ctr in combine(*gridPnts):
            centers.append(ctr)
            self.wts.append(np.zeros(self.n_dims))
        self.wts = np.array(self.wts)
        self.wts = self.wts.reshape(self.n_dims, -1)
        self.centers = np.array(centers)        # shape (n_cerebellums, 2*n_dims)
        self.n_cerebellums = self.centers.shape[0]
        self.sigma   = sigma
        self.activations = np.array([0 for _ in range(self.n_cerebellums)]).reshape(1, -1)

    ###################################
    def update(self,movement_error,velocity_error):
    ###################################
        lamb = np.ones(self.n_dims)*2 # sliding surface 
        s = (lamb*movement_error + velocity_error)
        if np.linalg.norm(s) < self.epsilon:
            return
        self.wts = self.wts - self.beta * np.outer(s, self.activations) - self.damp*self.wts*np.linalg.norm(s)

    ###################################
    def compute_correction(self,pos,vel):
    ###################################
        x = np.concatenate((pos, vel))
        diff = self.centers - x
        dist2 = np.einsum('ij,ij->i', diff, diff)
        self.activations = np.exp(-dist2 / (2 * self.sigma**2)).reshape(1, -1)
        corr = np.sum(self.activations * self.wts, axis=1)
        return corr