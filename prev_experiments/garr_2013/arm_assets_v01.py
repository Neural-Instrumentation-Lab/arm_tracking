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

class armTraj:
    '''
    stores a bunch of trajectory information to make handling
    the data easier
    '''
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
        '''
        returns the end effector position [x, y, z]
        from a joint config vector
        '''
        pin.forwardKinematics(self.model, self.data, pos)
        pin.updateFramePlacements(self.model, self.data)
        return self.getPos()

    def is_valid_location(self, pos):
        '''
        returns true if position pos (x, y, z)
        can be reached by the arm
        '''
        L1, L2, L3 = self.lengths
        x, y, z = pos
        D = np.linalg.norm([x, y, z-L1])
        return not (D > L2 + L3 or D < np.abs(L2-L3))

    def getJointPosFromEE(self, pos, prevQ=np.array([0, 0, 0])):
        '''
        Inverse Kinematics for a 3dof arm
        if arm is more complex then this needs to be iterative
        and not analytical
        '''
        x, y, z = pos
        L1, L2, L3 = self.lengths
        if not self.is_valid_location(pos):
            raise JointAngleError(f"Cannot reach this position: {pos}")
        D = np.linalg.norm([x, y, z-L1])
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
        # this arm has 4 solutions for any given position
        sola = [q1, np.pi/2 - q2a, -q3a]
        solb = [q1, np.pi/2 - q2b, -q3b]
        solc = [q1+np.pi, -(np.pi/2 - q2b), q3b]
        sold = [q1-np.pi, -(np.pi/2 - q2a), q3a]
        sol = min([sola, solb, solc, sold], key=lambda q: costHelper(q, prevQ))
        return np.array(sol)

    def forwardDynamics(self, positions, velocities, torques, time):
        '''
        computes the necessary accelerations along a given trajectory
       '''
        currPos = positions[0]     
        currVel = velocities[0] 
        cntrl_traj = armTraj(pos=np.zeros((len(torques), self.njoints)),
                            torq=np.zeros((len(torques), self.njoints)),
                            eePos=np.zeros((len(torques), 3)))
        currAcc = self.forward(currPos, currVel, torques[0])
        self.move(currPos, currVel, currAcc)
        torrPd        = np.zeros_like(currPos)
        kp = np.ones(self.njoints)*20
        kd = 2*np.sqrt(kp)
        for i, tau in enumerate(torques):
            torrPd = kp*angle_diff(positions[i], currPos) + kd*(velocities[i] - currVel)
            torque = tau + torrPd         
            if i > 0:
                dt     = time[i] - time[i-1]
                currAcc = self.forward(currPos, currVel, torque)
                currVel = currVel + currAcc * dt
                currPos = pin.integrate(self.model, currPos, currVel * dt)
            self.move(currPos, currVel, currAcc)
            cntrl_traj.pos[i, :]   = currPos
            cntrl_traj.torq[i, :]  = torque 
            cntrl_traj.eePos[i, :] = self.getPos()
        return cntrl_traj

    def inverseDynamics(self, positions, velocities, accels, time, n_substeps=100):
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
        kp        = np.ones(self.njoints) * 20
        kd        = 2*np.sqrt(kp) 
        pos       = positions[0]  
        vel       = velocities[0] 
        acc       = accels[0]
        for i, (posCorr, velCorr, accCorr) in enumerate(zip(positions, velocities, accels)):
            torques[i, :]   = self.inverse(posCorr, velCorr, accCorr) # ideal torque
            torquesPD[i, :] = torques[i, :] + kp*angle_diff(posCorr, pos) + kd*(velCorr - vel)
            if i > 0:
                dt  = time[i] - time[i-1]
                torque = torquesPD[i]       
                acc = self.forward(pos, vel, torque)
                vel = vel + acc * dt
                pos = pin.integrate(self.model, pos, vel * dt)
            self.move(pos, vel, acc)
            eePos[i,:] = self.getPos()
        return torquesPD, eePos

    def idealTorques(self, positions, velocities, accels):
        '''
        open loop inverseDynamics()
        '''
        torques   = np.zeros_like(positions)
        for i, (pos, vel, acc) in enumerate(zip(positions, velocities, accels)):
            torques[i, :] = self.inverse(pos, vel, acc)
        return torques

    def makeJointData(self, positions, velocities, accelerations):
        '''
        constructs joint positions, velocities, and accelerations
        for a given trajectory of the end-effector.
        calculates angular vel and acceleration thru joint Jacobian
        instead of derivative to reduce error thru derivation

        Args:
            arm: arm class.
            trajectory: positions (x,y,z) in 3d space of the end effector.
            t: time vector corresponding to the trajectory.
        Returns:
            joints: joint positions 
            velocity: joint angular velocities
            acceleration: joint angular accelerations 
        '''
        joint_pos = np.zeros((len(positions), self.njoints))
        joint_vels = np.zeros_like(joint_pos)
        joint_accs = np.zeros_like(joint_pos)
        q   = np.zeros(self.njoints)
        qd  = np.zeros(self.njoints)
        qdd = np.zeros(self.njoints)
        for i, coord in enumerate(positions):
            if i > 0:
                q = self.getJointPosFromEE(coord, joint_pos[i-1,:])
            else:
                q = self.getJointPosFromEE(coord)
            joint_pos[i, :] = q
            pin.forwardKinematics(self.model, self.data, q, qd)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.computeJointJacobiansTimeVariation(self.model, self.data, q, qd)
            J    = pin.getFrameJacobian(self.model, self.data, self.eeId, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            Jdot = pin.getFrameJacobianTimeVariation(self.model, self.data, self.eeId, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            qd = np.linalg.pinv(J) @ velocities[i, :] 
            qdd = np.linalg.pinv(J) @ (accelerations[i,:] - Jdot @ velocities[i,:])
            joint_vels[i, :] = qd
            joint_accs[i, :] = qdd
        return joint_pos, joint_vels, joint_accs 


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
