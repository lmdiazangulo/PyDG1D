import numpy as np

from .spatialDiscretization import *

class Leapfrog:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.fieldsRes = dict()
        for l, f in fields.items():
            self.fieldsRes[l] = np.zeros(f.shape)
    
    def step(self, fields, dt):
        fieldsRHS = self.sp.computeRHS(fields)
        for l, f in fieldsRHS.items():
            self.fieldsRes[l] = self.fieldsRes[l] + dt*f
            
        for l, f in fields.items():
            fields[l] += self.fieldsRes[l]

        self.time += dt