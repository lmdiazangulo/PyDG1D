import numpy as np

from maxwell1d import *

class LSERK4:
    A = np.array([
        0,	-0.417890474499852,	-1.19215169464268,	-1.69778469247153,	-1.51418344425716
    ])
    B = np.array([
        0.149659021999229,	0.379210312999627, 0.822955029386982,	0.699450455949122,	0.153057247968152
    ])
    C = np.array([
        0,	0.149659021999229,	0.370400957364205, 0.622255763134443,	0.958282130674690
    ])

    N_STAGES = 5

    def __init__(self, sp: Maxwell1D, fields):
        self.time = 0.0

        self.fieldsRes = list()
        for f in fields:
            self.fieldsRes.append(np.zeros(f.shape))
    
    def step(self, fields, dt):
        for s in range(0, self.N_STAGES):
            fieldsRHS = self.sp.computeRHS(fields)
            for f in range(len(fields)):
                self.fieldsRes[f] = self.A[s]*self.fieldsRes[f] + dt*fieldsRHS[f]
            
            for f in range(len(fields)):
                fields[f] += self.B[s] * self.fieldRes[f]

        self.time += dt