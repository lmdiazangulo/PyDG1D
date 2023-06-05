import numpy as np
from scipy.optimize import fsolve

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

            
    def backward_euler_residual ( self, fields, dt, xo):
        value = fields - xo - dt * fields

    def stepAM4(self, fields, dt):
        maxiter = 30
        tol = 1e-5
        
        i = 1
        dif = 1 
        x0 = fields
        while i < maxiter and dif > tol:
        #     maxiter = 50
        # tol = 1e-03
        # for k in range(4, self.N_STAGES):
        #     i = 1
        #     dif = 1 
        #     x0 = fields
        #     while i < maxiter and dif > tol:
        #         self.sp.computeRHS(fields)-
        #         [fx0, dfx0] = f(t[k], x0)
        #         g = x0 - fields - dt/24*(f(t[k-3],y[k-3])[0] -5*f(t[k-2],y[k-2])[0]+ 19*f(t[k-1],y[k-1])[0] +9*fx0)
        #         dg = 1-h/24*9*dfx0
            fields = yo + dt * fields
            yp = fsolve(self.backward_euler_residual, yp, args = ( f, to, yo, tp ) )



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
        
        
