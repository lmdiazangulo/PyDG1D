import numpy as np
from scipy.optimize import fsolve

from ..spatialDiscretization import *
#Backward Euler method
class IBE:
    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.A = sp.buildEvolutionOperator()          

    def backward_euler(self, yp, f, dt, yo):
        return yp - yo - dt * np.matmul(f, yp)


    def step(self, fields, dt):
        yo = self.sp.convertToVector(fields)
        
        yp = yo + dt/2 * self.A.dot(yo)
        yp = fsolve(self.backward_euler, yp, args = (self.A, dt, yo )) 

        self.time += dt
        
        self.sp.copyVectorToFields(yp, fields)
        
        
