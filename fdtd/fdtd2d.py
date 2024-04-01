import numpy as np


if __package__ == 'pydg1d.fdtd':
    from ..dgtd.spatialDiscretization import *
    from ..dgtd.mesh2d import Mesh2D
else:
    from dgtd.spatialDiscretization import *
    from dgtd.mesh2d import Mesh2D


class FDTD2D(SpatialDiscretization):  # TE mode
    def __init__(self, x_min, x_max, kx_elem, y_min=0.0, y_max=0.0, ky_elem=0, boundary_label="PEC"):
        if y_min == 0.0 and y_max == 0.0 and ky_elem == 0:
            y_min = x_min
            y_max = x_max
            ky_elem = kx_elem
        else:
            raise ValueError("Invalid values for y grid planes.")

        self.boundary_label = boundary_label

        self.x = np.linspace(x_min, x_max, num=kx_elem+1)
        self.y = np.linspace(y_min, y_max, num=ky_elem+1)
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)

        self.xH = (self.x[:-1] + self.x[1:]) / 2.0
        self.yH = (self.y[:-1] + self.y[1:]) / 2.0

        self.cEy = 1.0 / self.dy[0]
        self.cEx = 1.0 / self.dx[0]

    def buildFields(self):
        H = np.zeros((len(self.dy), len(self.dx)))
        Ex = np.zeros((len(self.y),  len(self.dx)))
        Ey = np.zeros((len(self.dy), len(self.x)))

        return {
            "E": {"x": Ex, "y": Ey},
            "H": H
        }

    def get_minimum_node_distance(self):
        return np.min(self.dx)

    def computeRHSE(self, fields):
        H = fields['H']
        Ex = fields['E']['x']
        Ey = fields['E']['y']

        rhsEx = np.zeros(Ex.shape)
        rhsEy = np.zeros(Ey.shape)

        rhsEx[1:-1, :] = - self.cEy * (H[:-1, :] - H[1:, :])
        rhsEy[:, 1:-1] = - self.cEx * (H[:, :-1] - H[:, 1:])

        if self.boundary_label == "PEC":
            rhsEx[0, :] = 0.0
            rhsEx[-1, :] = 0.0
            rhsEy[:,  0] = 0.0
            rhsEy[:, -1] = 0.0

        return {'x': rhsEx, 'y': rhsEy}

    def computeRHSH(self, fields):
        Ex = fields['E']['x']
        Ey = fields['E']['y']

        rhsH = - self.cEy*(Ex[1:, :] - Ex[:-1, :]) \
               + self.cEx*(Ey[:, 1:] - Ey[:, :-1])

        if self.boundary_label == "PMC": #[WIP]
            rhsH[0, :] = 0.0
            rhsH[-1, :] = 0.0
            rhsH[:,  0] = 0.0
            rhsH[:, -1] = 0.0

            #Para ello, hacemos que los campos magnéticos promediados sean cero 
            # en las posiciones enteras suponiendo que H es opuesto en las semi-enteras
            # próximas a la frotnera PMC

        return rhsH

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}

    def isStaggered(self):
        return True

    def dimension(self):
        return 2
