import numpy as np
from scipy.optimize import fsolve

from maxwell.spatialDiscretization import *
#Adams Moulton order 2 method

class AM2:
    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.A = sp.buildEvolutionOperator()          

    def adams_moulton_2_residual(self, yp, f, dt, yo1, yo2):
        return yp - yo1 - 1/12*dt*(5*np.matmul(f, yp)+8*np.matmul(f, yo1)-np.matmul(f, yo2))


    def step(self, fields, dt):
        yo1 = self.sp.convertToVector(fields)
        
        yo2 = self.sp.convertToVector(fields)
        
        yp = yo1 + dt/2 * (3*self.A.dot(yo1)-self.A.dot(yo2))
        yp = fsolve(self.adams_moulton_2_residual, yp, args = (self.A, dt, yo1, yo2 )) 

        self.sp.copyVectorToFields(yp, fields)
        
        
        
