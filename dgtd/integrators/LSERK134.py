import numpy as np

from dgtd.spatialDiscretization import *

class LSERK134:
    A = np.array([
        0,	                    -0.6160178650170565,    -0.4449487060774118,    
        -1.0952033345276178,    -1.2256030785959187,    -0.2740182222332805,    
        -0.0411952089052647,    -0.1797084899153560,    -1.1771530652064288,    
        -0.4078831463120878,    -0.8295636426191777,    -4.7895970584252288,
        -0.6606671432964504 
    ])
    B = np.array([
        0.0271990297818803,     0.1772488819905108,     0.0378528418949694,
        0.6086431830142991,     0.2154313974316100,     0.2066152563885843,
        0.0415864076069797,     0.0219891884310925,     0.9893081222650993,
        0.0063199019859826,     0.3749640721105318,     1.6080235151003195,
        0.0961209123818189
    ])


    N_STAGES = 13

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