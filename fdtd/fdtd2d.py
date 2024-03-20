import numpy as np


if __package__ == 'pydg1d.fdtd':
    from ..dgtd.spatialDiscretization import *
    from ..dgtd.mesh1d import Mesh1D
else:
    from dgtd.spatialDiscretization import *
    from dgtd.mesh1d import Mesh1D


class FDTD2D(SpatialDiscretization):
    def __init__(self, x_min, x_max, k_elem, boundary_label="PEC"):
        SpatialDiscretization.__init__(self, mesh)
        
        self.boundary_label= boundary_label

        self.x= np.linspace(x_min, x_max, k_elem+1)
        self.dx= np.diff(self.x)

        self.xEx = (self.x[:-1] + self.x[1:]) / 2.0
        self.xEy = (self.x[:-1] + self.x[1:]) / 2.0

        self.dxEx = self.xHx[1:] - self.xHx[:-1]
        self.dxEy = self.xHy[1:] - self.xHy[:-1]

    def buildFields(self):

        H = np.zeros(self.x.shape)

        Ex_grid = np.zeros(self.xEx.shape)
        Ey_grid = np.zeros(self.xEy.shape)

        Ex, Ey = np.meshgrid(Ex_grid, Ey_grid)

        return {"E":[Ex, Ey], "H": H}
    
    def get_minimum_node_distance(self):
        return np.min(self.dx)

    def computeRHSE(self, fields):

        H = fields['H']

        rhsEx = np.zeros(fields['E'[0].shape])
        rhsEy = np.zeros(fields['E'[1].shape])
        
        if self.boundary_label == "PEC":
          
            rhsEx[1:-1, :] += - (1.0/self.dxEx) * (H[1:, :] - H[:-1, :])
            rhsEy[:, 1:-1] -= - (1.0/self.dxEy) * (H[:, 1:] - H[:, :-1])
            
            rhsEx[0,:] = 0.0
            rhsEx[-1,:] = 0.0
            rhsEy[:,0] = 0.0
            rhsEy[:,-1] = 0.0

            rhsE = [rhsEx, rhsEy]
        
        return rhsE
    
    def computeRHSH(self,fields):

        Ex = fields['E'[0]]
        Ey = fields['E'[1]]

        rhsH = np.zeros(fields['H'].shape)

        if self.mesh.boundary_label == "PEC":
            rhsH = - (1.0/self.dx) * ((Ex[:, 1:] - Ex[:, :-1]) / self.dxEy - (Ey[1:, :] - Ey[:-1, :]) / self.dxEx)
        return rhsH
    
    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}
    
    
class FDTD2D_TE(FDTD2D):
    def __init__(self):
        FDTD2D.__init__(self, x_min, x_max, k_elem, boundary_label="PEC")


class FDTD2D_TM(FDTD2D):
    def __init__(self):
        FDTD2D.__init__(self, x_min, x_max, k_elem, boundary_label="PEC")

