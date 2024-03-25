import numpy as np


if __package__ == 'pydg1d.fdtd':
    from ..dgtd.spatialDiscretization import *
    from ..dgtd.mesh2d import Mesh2D
else:
    from dgtd.spatialDiscretization import *
    from dgtd.mesh2d import Mesh2D


class FDTD2D(): #TE mode
    def __init__(self, x_min, x_max, k_elem, boundary_label="PEC"):
        #SpatialDiscretization.__init__(self, mesh)
        
        self.boundary_label= boundary_label

        self.x= np.linspace(x_min, x_max, k_elem+1)
        self.dx= np.diff(self.x)

        self.xEx = (self.x[:-1] + self.x[1:]) / 2.0
        self.xEy = (self.x[:-1] + self.x[1:]) / 2.0

        self.dxEx = self.xEx[1:] - self.xEx[:-1]
        self.dxEy = self.xEy[1:] - self.xEy[:-1]

    def buildFields(self):

        H = np.zeros([self.x.shape[0], self.x.shape[0]])

        Ex = np.zeros([self.xEx.shape[0], self.xEy.shape[0]])
        Ey = np.zeros([self.xEx.shape[0], self.xEy.shape[0]])

        E = np.array([Ex,Ey])

        return {"E":E, "H": H}

#        return {"E":np.array([np.zeros(self.xEx.shape),np.zeros(self.xEy.shape)]),\
#                "H": np.zeros(self.x.shape)}
    
    def get_minimum_node_distance(self):
        return np.min(self.dx)

    def computeRHSE(self, fields):

        H = fields['H']

        rhsEx = np.zeros(fields['E'][0].shape)
        rhsEy = np.zeros(fields['E'][1].shape)

        rhsEx[1:-1, :] = - (1.0/self.dxEy) * (H[1:, :] - H[:-1, :])
        rhsEy[:, 1:-1] = - (1.0/self.dxEx) * (H[:, 1:] - H[:, :-1])
        
        if self.boundary_label == "PEC":
          
            rhsEx[0,:] = 0.0
            rhsEx[-1,:] = 0.0

            rhsEy[:,0] = 0.0
            rhsEy[:,-1] = 0.0

            rhsE = np.array([rhsEx, rhsEy])
    
        return rhsE
    
    def computeRHSH(self,fields):

        Ex = fields['E'][0]
        Ey = fields['E'][1]

        rhsH = np.zeros(fields['H'].shape)

        rhsH = - 1.0 * ((Ex[:, 1:] - Ex[:, :-1]) / self.dxEy - (Ey[1:, :] - Ey[:-1, :]) / self.dxEx)
        
        return rhsH

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}
    

#     r_min = sp.get_minimum_node_distance()
# >       if (sp.isStaggered()):
# E       AttributeError: 'FDTD2D' object has no attribute 'isStaggered'

# dgtd\maxwellDriver.py:26: AttributeError