import numpy as np

from .spatialDiscretization import *

class EULER:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

    def step(self, fields, dt):
        E = fields['E']
        H = fields['H']
        
        
        if self.time == 0.0:
            
            E += dt/2*self.sp.computeRHSE(fields)
            H += dt/2*self.sp.computeRHSH(fields)
        
        self.time += dt
        
        H += dt/2*self.sp.computeRHSH(fields)
        E += dt/2*self.sp.computeRHSE(fields)
   