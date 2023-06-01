import numpy as np

from .spatialDiscretization import *

class IRK4:
    A = np.array([
        0,	-0.417890474499852,	-1.19215169464268,	-1.69778469247153,	-1.51418344425716
    ])
    B = np.array([
        0.149659021999229,	0.379210312999627, 0.822955029386982,	0.699450455949122,	0.153057247968152
    ])
    C = np.array([
        0,	0.149659021999229,	0.370400957364205, 0.622255763134443,	0.958282130674690
    ])

    N_STAGES = 2

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

        self.fieldsRes = dict()
        for l, f in fields.items():
            self.fieldsRes[l] = np.zeros(f.shape)
    
    def step(self, fields, dt):
  
        # for _ in range(0, 4):
        #     time += dt/2
        #     fieldsRHS = self.sp.computeRHS(fields)
        #     k1 = self.sp.computeRHS(fields)
        #     k2 = k1*dt/2*self.sp.computeRHS(fields)
        #     k3 = k2*dt/2*self.sp.computeRHS(fields)
        #     k4 = k3*dt*self.sp.computeRHS(fields)
        #     fields += dt*(k1+2*k2+2*k3+k4)
        
        maxiter = 50
        tol = 1e-03
        for k in range(4, self.N_STAGES):
            i = 1
            dif = 1 
            x0 = fields
            while i < maxiter and dif > tol:
                [fx0, dfx0] = f(t[k], x0)
                g = x0 - y[k-1] - h/24*(f(t[k-3],y[k-3])[0] -5*f(t[k-2],y[k-2])[0]+ 19*f(t[k-1],y[k-1])[0] +9*fx0)
                dg = 1-h/24*9*dfx0
                x1 = x0-g/dg
                dif=abs(x1-x0)
                i +=1
                x0=x1
            y.append( y[k-1] + h/24*(f(t[k-3],y[k-3])[0] -5*f(t[k-2],y[k-2])[0]+ 19*f(t[k-1],y[k-1])[0] +9*f(t[k],x0)[0])  )
        
        return([y])