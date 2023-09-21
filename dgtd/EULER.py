import numpy as np

from .spatialDiscretization import *

class EULER:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

    def step(self, fields, dt):
        E = fields['E']
        H = fields['H']
        
        
        # if self.time == 0.0:
        #     H += dt*self.sp.computeRHSH(fields)
        
        self.time += dt
        E += dt*self.sp.computeRHSE(fields)
        H += dt*self.sp.computeRHSH(fields)