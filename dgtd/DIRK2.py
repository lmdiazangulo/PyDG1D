import numpy as np
from scipy.optimize import fsolve

from .spatialDiscretization import *
#RK4


# A = np.array([1/4,              1/4-np.sqrt(3)/6,   1/4+np.sqrt(3)/6,   1/4])
# b = np.array([1/2,              1/2])
# c = np.array([1/2-np.sqrt(3)/6, 1/2+np.sqrt(3)/6])

class DIRK2:
    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.A = sp.buildEvolutionOperator()          

    def k1_residual(self, k1, f, dt, yo):
        return k1 - np.matmul(f, yo + dt/4 *k1)
    
    def k2_residual(self, k2, f, dt, y1):
        return k2-np.matmul(f, y1 + dt/4*k2)

    def step(self, fields, dt):
        yo = self.sp.convertToVector(fields)
        k1 = np.ones(yo.shape)
        k2 = np.ones(yo.shape)
        k1 = fsolve(self.k1_residual, k1, args = (self.A, dt, yo ))
        y1 = yo + dt/2*k1
        k2 = fsolve(self.k2_residual, k2, args = (self.A, dt, y1 ))
        yp = yo + dt/2 * (k1 + k2)
        
        self.sp.copyVectorToFields(yp, fields)
        
        
