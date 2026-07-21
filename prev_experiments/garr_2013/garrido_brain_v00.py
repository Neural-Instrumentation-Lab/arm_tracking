import numpy as np

class BrainError(Exception): pass

class cerebellum:
    def __init__(self, n_dof=2):
        self.currPF = 0
        self.nPFs = 500
        self.pfIdx = 0
        self.nMuscles = n_dof*2 # one pair agonist-antagonist per joint
        self.purAct = np.zeros(self.nMuscles)
        self.pcLookup = np.zeros(self.nPFs) 
        self.pf_pc_weights = np.zeros((self.nPFs, self.nMuscles))
        self.mf_dcn_weights = np.zeros(self.nMuscles)
        self.pc_dcn_weights = np.zeros(self.nMuscles)
        self.dcnAct = np.zeros(self.nMuscles)
        self.pf_pc_only = False 
        # from garrido paper
        self.LTP_max = 0.01 # long-term potentiation
        self.LTD_max = 0.02 # long-term depression 
        self.LTP_max_dcn = 1e-3 # long-term potentiation
        self.LTD_max_dcn = 1e-4 # long-term depression 
        self.alpha = 1000 # LTP decay factor
        self.active_pf_pc = True
        self.active_pc_dcn = True
        self.active_mf_dcn = True

    def granuleLayer(self, state):
        '''
        The granular layer translates the 
        inputs to the parallel fibers,
        which act as a state machine (discritizing the motion)
        '''
        self.pfIdx = (state - 1) % self.nPFs
        self.currPF = (state) % self.nPFs 
        #print(f"{self.pfIdx} {self.currPF}")
        # if state >= self.nPFs:
        #     raise BrainError("state is greater than the number of PFs")

    def updatePF_PC(self, error):
        '''
        updates the synaptic weights between the parallel fibers
        and the purkinje cell
        '''
        self.pf_pc_weights[self.pfIdx,:] += (self.LTP_max / ((error+1)**self.alpha)) - self.LTD_max*error
        self.pf_pc_weights[self.pfIdx,:] = np.clip(self.pf_pc_weights[self.pfIdx,:], 0, 1)

    def purkinjeCompute(self):
        '''
        computes the purkinje cell firing rate [0 - 1]
        '''
        self.purAct = self.pf_pc_weights[self.currPF, :].copy() 
        self.purAct = np.clip(self.purAct, 0, 1)

    def getPC(self):
        return self.purAct

    def getDCN(self):
        return self.dcnAct

    def updateMF_DCN(self):
        '''
        updates the synaptic weights between the mossy fibers and the 
        deep cerebellar nuclei
        '''
        self.mf_dcn_weights += (self.LTP_max_dcn / ((self.purAct + 1)**self.alpha)) - self.LTD_max_dcn*self.purAct
        self.mf_dcn_weights = np.clip(self.mf_dcn_weights, 0, None) 

    def getMF_DCN(self):
        return(self.mf_dcn_weights)

    def getPC_DCN(self):
        return(self.pc_dcn_weights)

    def getPF_PC(self):
        return(self.pf_pc_weights)

    def getnPFs(self):
        return(self.nPFs)

    def setActiveSites(self, pf_pc, mf_dcn, pc_dcn):
        self.active_mf_dcn = mf_dcn
        self.active_pc_dcn = pc_dcn
        self.active_pf_pc  = pf_pc

    def updatePC_DCN(self):
        '''
        updates the weights between the purkinje cell
        and the deep cerebellar nuclei
        '''
        dcn_clipped = np.clip(self.dcnAct, 0, 1)
        self.pc_dcn_weights += ((self.LTP_max_dcn * self.purAct**self.alpha) * (1 - 1/(dcn_clipped + 1)**self.alpha)) - self.LTD_max_dcn*(1-self.purAct)
        self.pc_dcn_weights = np.clip(self.pc_dcn_weights, 0, None) 

    def DCNCompute(self):
        '''
        computes the deep cerebellar nuclei's activation,
        which is the torque outputs (for each muscle)
        '''
        self.dcnAct = self.mf_dcn_weights - self.purAct*self.pc_dcn_weights 
        self.dcnAct = np.clip(self.dcnAct, 0, None)

    def dcnToTorque(self):
        '''
        takes the DCN's output and adds antagonist and agonist together
        to make joint torque commands
        '''
        corr = self.dcnAct.copy()
        corr[1::2] *= -1
        corr = (corr).reshape(-1, 2).sum(axis=1) 
        return corr
    
    def compute(self, qError, qdError, state):
        '''
        updates the entire brain given the error signals
        '''
        # combine errors
        if state == 0:
            # no learning, just computation on first state
            self.currPF = 0
            self.purkinjeCompute()
            self.DCNCompute()
            return self.dcnToTorque()
        self.granuleLayer(state)
        posCon = [1, 12, 6]
        velCon = [2, 10, 5]
        error = posCon*qError + velCon*qdError 
        # error = np.tanh(error)
        agonist = np.maximum(error, 0)
        antagonist = np.maximum(-error, 0)
        error = np.stack([agonist, antagonist], axis=1).reshape(-1)  # (n_muscles,)
        error = np.clip(error, 0, 1) 
        #error = np.tanh(error) # clips errors to 0 - 1 BUT I DONT LIKE IT
        # for agonist / antagonist pairs
        if self.active_pf_pc:
            self.updatePF_PC(error)
        self.purkinjeCompute()
        if self.active_mf_dcn:
            self.updateMF_DCN()
        if self.active_pc_dcn:
            self.updatePC_DCN()
        self.DCNCompute()
        return self.dcnToTorque()

    def loadWts(self, init_pf_pc, init_mf_dcn, init_pc_dcn):
        self.pf_pc_weights  = init_pf_pc
        self.mf_dcn_weights = init_mf_dcn
        self.pc_dcn_weights = init_pc_dcn