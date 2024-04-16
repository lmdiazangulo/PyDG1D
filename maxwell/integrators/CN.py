import numpy as np
from scipy.optimize import fsolve

from maxwell.spatialDiscretization import *
#Crank Nicolson method

class CN:
    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.A = sp.buildEvolutionOperator()          

    def crank_nicolson_residual(self, yp, f, dt, yo):
        return yp - yo - 0.5*dt*(np.matmul(f, yp) + np.matmul(f, yo))


    def step(self, fields, dt):
        
        yo = self.sp.convertToVector(fields)
        
        yp = yo + dt * self.A.dot(yo)
        yp = fsolve(self.crank_nicolson_residual, yp, args = (self.A, dt, yo )) 
        
        self.time += dt

        self.sp.copyVectorToFields(yp, fields)
        
        
        
