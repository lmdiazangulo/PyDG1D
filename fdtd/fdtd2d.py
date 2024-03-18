import numpy as np


if __package__ == 'pydg1d.fdtd':
    from ..dgtd.spatialDiscretization import *
    from ..dgtd.mesh1d import Mesh1D
else:
    from dgtd.spatialDiscretization import *
    from dgtd.mesh1d import Mesh1D


class FDTD2D():
    def __init__(self, x_min, x_max, N, boundary_label="PEC"):

        self.x= np.linspace(x_min, x_max, N)
        self.xHx = (self.x[:-1] + self.x[1:]) / 2.0
        self.xHy = (self.x[:-1] + self.x[1:]) / 2.0

        self.dx= np.diff(self.x)
        self.dxHx = self.xHx[1:] - self.xHx[:-1]
        self.dxHy = self.xHy[1:] - self.xHy[:-1]

    def buildFields(self):
        E = np.zeros(self.x.shape)

        Hx_grid = np.zeros(self.xH_x.shape)
        Hy_grid = np.zeros(self.xH_y.shape)
        
        Hx, Hy = np.meshgrid(Hx, Hy)

        return {"E": E, "Hx": Hx, "Hy": Hy}

    def computeRHSE(self, fields):
        Hx = fields['Hx']
        Hy = fields['Hy']
        rhsE = np.zeros(fields['E'.shape])