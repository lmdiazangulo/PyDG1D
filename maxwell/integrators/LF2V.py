import numpy as np

from ..spatialDiscretization import *

class LF2V:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

    def step(self, fields, dt):
        E = fields['E']
        H = fields['H']
        
        E += 0.5*dt*self.sp.computeRHSE(fields)
        H += dt*self.sp.computeRHSH(fields)
        E += 0.5*dt*self.sp.computeRHSE(fields)
        self.time += dt
        
        
        # #Verlet Algorithm
        # self.time += dt
        # E += dt*self.sp.computeRHSE(fields)
        # H += 0.5*dt*self.sp.computeRHSH(fields)
        # E_old = E

        # self.time += dt/2
        
        # H += 0.5*dt*self.sp.computeRHSH(fields)
        # E += dt*self.sp.computeRHSE(fields)
        # E_new = E
        
        # E += 2*E_new - E_old + dt**2*self.sp.computeRHSH(fields)
        
        
        


        
        