import numpy as np

from .spatialDiscretization import *

class LF2_2D:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

    def step(self, fields, dt):
        E = fields['E']
        H = fields['H'] 
        
        self.time += dt/2
        E += dt*self.sp.computeRHSE(fields)
        self.time += dt/2
        H += dt*self.sp.computeRHSH(fields)
        
        