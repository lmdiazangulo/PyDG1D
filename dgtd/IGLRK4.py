import numpy as np
from scipy.optimize import fsolve

from .spatialDiscretization import *
#RK4


# A = np.array([1/4,              1/4-np.sqrt(3)/6,   1/4+np.sqrt(3)/6,   1/4])
# b = np.array([1/2,              1/2])
# c = np.array([1/2-np.sqrt(3)/6, 1/2+np.sqrt(3)/6])

class IGLRK4:
    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.A = sp.buildEvolutionOperator()          

    def k_residual(self, k, f, dt, yo):
        #k = k.reshape(2, len(yo), order='F')
        k1 = k[:len(yo)]
        k2 = k[len(yo):] 
        return np.array([k1 - np.matmul(f, yo + dt/4 *k1 + (1/4 - np.sqrt(3)/6)*dt*k2), k2 - np.matmul(f, yo + (1/4 + np.sqrt(3)/6)*dt*k1 + 1/4*dt*k2) ])

    def step(self, fields, dt):
        yo = self.sp.convertToVector(fields)
        k1, k2 = fsolve(self.k_residual, np.ones((2,len(yo))), args = (self.A, dt, yo ))
        yp = yo + dt/2 * (k1 + k2)
        
        self.sp.copyVectorToFields(yp, fields)
        
        
