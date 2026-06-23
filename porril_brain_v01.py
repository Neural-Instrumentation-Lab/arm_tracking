import numpy as np
from numpy import sin, cos, arccos, exp
from itertools import product as combine
## Marr Albus Cerebellum #######################################################
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
        print(sigma)

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