import numpy as np

from .spatialDiscretization import *
import LSERK4
from copy import deepcopy

class IRK4:
    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0
        self.initiator = LSERK4(sp, fields)

        fm3 = dict()
        for l, k in fields.items():
            fm3[l] = np.zeros(k.shape)
        fm2 = deepcopy(fm3)
        fm1 = deepcopy(fm3)
    
    def stepRK(self, fields, dt):
        self.initiator.step(fields, dt)

    def stepAM4(self, fields, dt):
        maxiter = 10
        tol = 1e-6
        
        i = 1
        dif = 1 
        x0 = y[k-1]
        while i < maxiter and dif > tol:
            [fx0, dfx0] = f(t[k], x0)
            g = x0 - y[k-1] - h/24*(f(t[k-3],y[k-3]) -5*f(t[k-2],y[k-2])+ 19*f(t[k-1],y[k-1]) +9*fx0)
            dg = 1-h/24*9*dfx0
            x1 = x0-g/dg
            dif=abs(x1-x0)
            i +=1
            x0=x1

    def step(self, fields, dt):
        currentStep = int(round(self.time / dt))

        if currentStep < 4:
            self.stepRK(fields, dt)
        else:
            self.stepAM4(fields, dt)

        for l, f in fields.items():
            self.fm3[l][:] = self.fm2[l][:]
            self.fm2[l][:] = self.fm1[l][:]
            self.fm1[l][:] = f[l][:]
        