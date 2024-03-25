import numpy as np


if __package__ == 'pydg1d.fdtd':
    from ..dgtd.spatialDiscretization import *
    from ..dgtd.mesh2d import Mesh2D
else:
    from dgtd.spatialDiscretization import *
    from dgtd.mesh2d import Mesh2D


class FDTD2D(): #TE mode
    def __init__(self, x_min, x_max, kx_elem, y_min=0.0, y_max=0.0, ky_elem=0, boundary_label="PEC"):
        if y_min == 0.0 and y_max == 0.0 and ky_elem == 0:
            y_min = x_min
            y_max = x_max
            ky_elem = kx_elem
        else:
            raise ValueError("Invalid values for y grid planes.")
        
        self.boundary_label= boundary_label

        self.x  = np.linspace(x_min, x_max, num=kx_elem+1)
        self.y  = np.linspace(y_min, y_max, num=ky_elem+1)
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)

        self.xH = (self.x[:-1] + self.x[1:]) / 2.0
        self.yH = (self.y[:-1] + self.y[1:]) / 2.0

    def buildFields(self):
        H  = np.zeros((len(self.dx), len(self.dy)))
        Ex = np.zeros((len(self.x),  len(self.y)))
        Ey = np.zeros((len(self.x),  len(self.dy)))

        E = np.array([Ex,Ey])

        return {"E":E, "H": H}
    
    def get_minimum_node_distance(self):
        return np.min(self.dx)

    def computeRHSE(self, fields):
        H = fields['H']
        Ex = fields['E'][0]
        Ey = fields['E'][1]

        rhsEx = np.zeros(Ex.shape)
        rhsEy = np.zeros(Ey.shape)

        rhsEx[1:-1, :] = - (1.0/self.dy) * (H[1:, :] - H[:-1, :])
        rhsEy[:, 1:-1] = - (1.0/self.dx) * (H[:, 1:] - H[:, :-1])
        
        if self.boundary_label == "PEC":
            rhsEx[ 0,  :] = 0.0
            rhsEx[-1,  :] = 0.0
            rhsEy[ :,  0] = 0.0
            rhsEy[ :, -1] = 0.0
        
        rhsE = np.array([rhsEx, rhsEy])
    
        return rhsE
    
    def computeRHSH(self,fields):
        H  = fields['H']
        Ex = fields['E'][0]
        Ey = fields['E'][1]

        rhsH = np.zeros(H.shape)

        rhsH = - 1.0 * ((Ex[:, 1:] - Ex[:, :-1]) / self.dy - (Ey[1:, :] - Ey[:-1, :]) / self.dx)
        
        return rhsH

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}
