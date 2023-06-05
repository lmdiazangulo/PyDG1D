import numpy as np
from scipy.optimize import fsolve

from .spatialDiscretization import *

class IBE:
    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.A = sp.buildEvolutionOperator()          

    def backward_euler_residual(self, yp, f, dt, yo):
        return yp - yo - dt * np.matmul(f, yp)


    def step(self, fields, dt):
        yo = self.sp.convertToVector(fields)
        
        yp = yo + dt * self.A.dot(yo)
        yp = fsolve(self.backward_euler_residual, yp, args = (self.A, dt, yo )) 

        self.sp.copyVectorToFields(yp, fields)
        
        
        
