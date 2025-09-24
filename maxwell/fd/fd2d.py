import numpy as np

from ..spatialDiscretization import *

class FD2D(SpatialDiscretization):  # TE mode
    def __init__(self, x_min, x_max, kx_elem, y_min=0.0, y_max=0.0, ky_elem=0, boundary_labels="PEC"):
        
        if type(boundary_labels) == str:
            self.boundary_labels = dict()
            self.boundary_labels["XL"] = boundary_labels
            self.boundary_labels["XU"] = boundary_labels
            self.boundary_labels["YL"] = boundary_labels
            self.boundary_labels["YU"] = boundary_labels
        else:
            self.boundary_labels = boundary_labels

        if y_min == 0.0 and y_max == 0.0 and ky_elem == 0:
            y_min = x_min
            y_max = x_max
            ky_elem = kx_elem
        else:
            raise ValueError("Invalid values for y grid planes.")

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

        rhsEx[1:-1, :] =   self.cEy * ( H[1:, :] - H[:-1, :])
        rhsEy[:, 1:-1] = - self.cEx * ( H[:, 1:] - H[:, :-1])

        for bdr, label in self.boundary_labels.items():
            if bdr == "XL":
                if label == "PEC":
                    rhsEy[:,  0] = 0.0
                elif label == "PMC":
                    rhsEy[:, 0] =  - self.cEx * (2*H[:,0])
            elif bdr == "XU":
                if label == "PEC":
                    rhsEy[:, -1] = 0.0
                elif label == "PMC":
                    rhsEy[:, -1] =  - self.cEx * (-2*H[:,-1])
            elif bdr == "YL":
                if label == "PEC":
                    rhsEx[0, :] = 0.0
                elif label == "PMC":
                    rhsEx[0, :] =  self.cEy * (2*H[0,:])
            elif bdr == "YU":
                if label == "PEC":
                    rhsEx[-1, :] = 0.0
                elif label == "PMC":
                    rhsEx[-1, :] =  self.cEy * (-2*H[-1,:])
            else:
                raise ValueError("Invalid boundary tag.")       

        return {'x': rhsEx, 'y': rhsEy}

    def computeRHSH(self, fields):
        Ex = fields['E']['x']
        Ey = fields['E']['y']

        rhsH = + self.cEy*(Ex[1:, :] - Ex[:-1, :]) \
               - self.cEx*(Ey[:, 1:] - Ey[:, :-1])

        return rhsH

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}

    def isStaggered(self):
        return True

    def dimension(self):
        return 2
    
    def buildMagneticAlternateMatrixBasis(self):
        Ny = len(self.dy)
        Nx = len(self.dx)

        matrices = []
        minimum_separation = 3
        positions = [ (i, j) for i in [0, minimum_separation] for j in [0, minimum_separation]]
        shift = [0, 1, 2]

        for y_s in shift:
            for x_s in shift:
                M = np.zeros((Ny, Nx), dtype=int)

                for y0, x0 in positions:
                    y = y0 + y_s
                    x = x0 + x_s
                    if 0 <= y < Ny and 0 <= x < Nx:
                        M[y, x] = 1
                if np.any(M):
                    matrices.append(M)
        return matrices
    
    def buildElectricAlternateMatrixBasis(self):
        Ny = len(self.y)
        Nx = len(self.x)

        matrices = []
        minimum_separation = 2

        M1_Ex = np.zeros((Ny, Nx-1))
        M2_Ex = np.zeros((Ny, Nx-1))
        basis_Ex = np.ones(Nx-1)

        for i in range(Ny):
            if i%2 == 0:
                M1_Ex[i, :] = basis_Ex[:]
            else:
                M2_Ex[i, :] = basis_Ex[:]

        matrices.append(M1_Ex)
        matrices.append(M2_Ex)

        M1_Ey = np.zeros((Ny-1, Nx))
        M2_Ey = np.zeros((Ny-1, Nx))
        basis_Ey = np.ones(Ny-1)

        for i in range(Nx):
            if i%2 == 0:
                M1_Ey[:, i] = basis_Ey[:]
            else:
                M2_Ey[:, i] = basis_Ey[:]

        matrices.append(M1_Ey)
        matrices.append(M2_Ey)

        return matrices

    def buildAlternateBasis(self):
        matrices_E = self.buildElectricAlternateMatrixBasis()
        matrices_H = self.buildMagneticAlternateMatrixBasis()

        return matrices_E + matrices_H