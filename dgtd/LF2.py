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
            time = time + dt/2
            E_aux = self.sp.computeRHSE(fields)
            H_aux = self.sp.computeRHSH(fields)
            time = time + dt/2
            E_new += + dt*E_aux
            H_new += dt/2*self.sp.computeRHSH(fields)
        
        self.time += dt/2
        E += dt*self.sp.computeRHSE(fields)
        self.time += dt/2
        H += dt*self.sp.computeRHSH(fields)
        
        