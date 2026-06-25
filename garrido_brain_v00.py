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
        self.alpha = 1000 # LTP decay factor

    def granuleLayer(self):
        self.currPF += 1
        if self.currPF > self.nPFs:
            raise BrainError("More timesteps then parallel fibers")

    def updatePF_PC(self, error):
        self.pf_pc_weights[self.currPF,:] += (self.LTP_max / ((error+1)**self.alpha)) - self.LTD_max*error

    def purkinjeCompute(self, motorError):
        self.updatePF_PC(motorError)
        self.purAct = self.pf_pc_weights[self.currPF, :] 

    def updateMF_DCN(self):
        self.mf_dcn_weights += (self.LTP_max / ((self.purAct + 1)**self.alpha)) - self.LTD_max*self.purAct

    def updatePC_DCN(self):
        self.pc_dcn_weights += (self.LTP_max * self.purAct**self.alpha) / ((self.dcnAct + 1)**self.alpha) - self.LTD_max*(1-self.purAct)

    def DCNCompute(self):
        self.updateMF_DCN()
        self.updatePC_DCN()
        self.dcnAct = self.mf_dcn_weights - self.purAct*self.pc_dcn_weights 

    def compute(self, qError, qdError):
        # combine errors
        s = [1, 1, 1]
        error = qError + s * qdError 
        error = np.repeat(error)
        # for agonist / antagonist pairs
        error[1::2] *= -1
        self.granuleLayer()
        self.purkinjeCompute(error)
        self.DCNCompute()
        return self.dcnAct