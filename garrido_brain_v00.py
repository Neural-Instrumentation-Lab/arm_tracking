import numpy as np
class BrainError(Exception): pass

class cerebellum:
    def __init__(self, n_dof=2):
        self.currPF = 0
        self.nPFs = 500
        self.nMuscles = n_dof*2 # one pair agonist-antagonist per joint
        self.purAct = np.zeros(self.nMuscles)
        self.pcLookup = np.zeros(self.nPFs) 
        self.pf_pc_weights = np.zeros((self.nPFs, self.nMuscles))
        self.mf_dcn_weights = np.zeros(self.nMuscles)
        self.pc_dcn_weights = np.zeros(self.nMuscles)
        self.dcnAct = np.zeros(self.nMuscles)
        # from garrido paper
        self.LTP_max = 0.01 # long-term potentiation
        self.LTD_max = 0.02 # long-term depression 
        self.LTP_max_dcn = 1e-3 # long-term potentiation
        self.LTD_max_dcn = 1e-4 # long-term depression 
        self.alpha = 1000 # LTP decay factor

    def granuleLayer(self):
        self.currPF += 1
        if self.currPF > self.nPFs:
            raise BrainError("More timesteps then parallel fibers")

    def updatePF_PC(self, error):
        self.pf_pc_weights[self.currPF-1,:] += (self.LTP_max / ((error+1)**self.alpha)) - self.LTD_max*error
        self.pf_pc_weights[self.currPF-1,:] = np.clip(self.pf_pc_weights[self.currPF-1,:], 0, 1)

    def purkinjeCompute(self, motorError):
        self.updatePF_PC(motorError)
        self.purAct = self.pf_pc_weights[self.currPF-1, :].copy() 
        self.purAct = np.clip(self.purAct, 0, 1)

    def updateMF_DCN(self):
        self.mf_dcn_weights += (self.LTP_max_dcn / ((self.purAct + 1)**self.alpha)) - self.LTD_max_dcn*self.purAct
        self.mf_dcn_weights = np.clip(self.mf_dcn_weights, 0, None) 

    def updatePC_DCN(self):
        dcn_clipped = np.clip(self.dcnAct, 0, 1)
        self.pc_dcn_weights += ((self.LTP_max_dcn * self.purAct**self.alpha) / ((dcn_clipped + 1)**self.alpha)) - self.LTD_max_dcn*(1-self.purAct)
        self.pc_dcn_weights = np.clip(self.pc_dcn_weights, 0, None) 

    def DCNCompute(self):
        self.updateMF_DCN()
        self.updatePC_DCN()
        self.dcnAct = self.mf_dcn_weights - self.purAct*self.pc_dcn_weights 

    def dcnToTorque(self):
        corr = self.dcnAct.copy()
        corr[1::2] *= -1
        corr = (corr).reshape(-1, 2).sum(axis=1) 
        return corr
    
    def compute(self, qError, qdError):
        # combine errors
        self.granuleLayer()
        s = [2, 2, 2]
        error = qError + s * qdError 
        agonist = np.maximum(error, 0)
        antagonist = np.maximum(-error, 0)
        error = np.stack([agonist, antagonist], axis=1).reshape(-1)  # (n_muscles,)
        error = np.tanh(error) # clips errors to 0 - 1 BUT I DONT LIKE IT
        # for agonist / antagonist pairs
        self.purkinjeCompute(error)
        self.DCNCompute()
        return self.dcnToTorque()
    
    def resetTraj(self):
        self.currPF = 0