import numpy as np

from ..spatialDiscretization import *

class EULER:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

    def step(self, fields, dt):
        Eold = fields['E'].copy()
        Hold = fields['H'].copy()
        
        Eold += dt*self.sp.computeRHSE(fields)
        Hold += dt*self.sp.computeRHSH(fields)
        
        fields['E'] = Eold.copy()
        fields['H'] = Hold.copy()
        
        self.time += dt
   