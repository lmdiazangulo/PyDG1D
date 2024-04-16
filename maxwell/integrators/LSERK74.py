import numpy as np

from maxwell.spatialDiscretization import *

class LSERK74:
    A = np.array([
        0,	                -0.647900745934,    -2.704760863204,     -0.460080550118,   -0.500581787785,    -1.906532255913,    -1.450000000000
    ])
    B = np.array([
        0.117322146869,	    0.503270262127,     0.233663281658,	    0.283419634625,	    0.540367414023,     0.371499414620,     0.136670099385
    ])


    N_STAGES = 7

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.fieldsRes = dict()
        for l, f in fields.items():
            self.fieldsRes[l] = np.zeros(f.shape)
    
    def step(self, fields, dt):
        for s in range(0, self.N_STAGES):
            fieldsRHS = self.sp.computeRHS(fields)
            for l, f in fieldsRHS.items():
                self.fieldsRes[l] = self.A[s]*self.fieldsRes[l] + dt*f
            
            for l, f in fields.items():
                fields[l] += self.B[s] * self.fieldsRes[l]

        self.time += dt