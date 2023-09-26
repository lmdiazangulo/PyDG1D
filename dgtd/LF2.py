import numpy as np

from .spatialDiscretization import *

class LF2:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

    def step(self, fields, dt):
        E = fields['E']
        H = fields['H']
        E_aux = E
        H_aux = H
          
        if self.time == 0.0:
            E_initial = E
            H_initial = H
            time = time + dt
            E_new = self.sp.computeRHSE(fields)
            H_new = self.sp.computeRHSH(fields)
            time = time + dt
            
            E_old = E_new
            H_old = H_new
            
            E_new += dt*self.sp.computeRHSE(fields)
            H_new += dt*self.sp.computeRHSH(fields)
        
        self.time += dt
        E += dt*self.sp.computeRHSE(fields)
        self.time += dt
        H += dt*self.sp.computeRHSH(fields)
        
        