import numpy as np

from dgtd.spatialDiscretization import *

class LF2:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

    def step(self, fields, dt):
        E = fields['E']
        H = fields['H'] 
        
        if self.sp.dimension() == 1:
            self.time += dt/2
            E += dt * self.sp.computeRHSE(fields, self.time)
            self.time += dt/2
            H += dt * self.sp.computeRHSH(fields, self.time)
        elif self.sp.dimension() == 2:
            self.time += dt/2
            rhsE = self.sp.computeRHSE(fields)
            E['x'][:,:] += dt * rhsE['x'][:,:]
            E['y'][:,:] += dt * rhsE['y'][:,:]
            self.time += dt/2
            H += dt * self.sp.computeRHSH(fields)
        else:
            raise ValueError("Invalid dimension")        
        
        