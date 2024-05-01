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
        
        
        


        
        