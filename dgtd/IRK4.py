import numpy as np
from scipy.optimize import fsolve

from .spatialDiscretization import *
#RK4


A = np.array([1/4,              1/4-np.sqrt(3)/6,   1/4+np.sqrt(3)/6,   1/4])
b = np.array([1/2,              1/2])
c = np.array([1/2-np.sqrt(3)/6, 1/2+np.sqrt(3)/6])

class RK4:
    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.A = sp.buildEvolutionOperator()          

    def runge_kutta_4_residual(self, yp, f, dt, yo):
        
        return yp - yo - 0.5*dt*(np.matmul(f, yp)+np.matmul(f, yo))


    def step(self, fields, dt):
        yo = self.sp.convertToVector(fields)
        
        yp = yo + dt * self.A.dot(yo)
        yp = fsolve(self.crank_nicolson_residual, yp, args = (self.A, dt, yo )) 

        self.sp.copyVectorToFields(yp, fields)
        
        
        
