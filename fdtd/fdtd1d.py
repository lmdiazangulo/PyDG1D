import numpy as np

from dgtd.spatialDiscretization import *
from dgtd.mesh1d import Mesh1D


class FDTD1D(SpatialDiscretization):
    def __init__(self, mesh: Mesh1D):
        SpatialDiscretization.__init__(self, mesh)

        self.x = mesh.vx
        self.xH = (self.x[:-1] + self.x[1:]) / 2.0
        self.dx = self.x[1:] - self.x[:-1]
        self.dxH = self.xH[1:] - self.xH[:-1]

        K = self.mesh.number_of_elements()

    def buildFields(self):
        E = np.zeros(self.x.shape)
        H = np.zeros(self.xH.shape)

        return {"E": E, "H": H}

    def get_minimum_node_distance(self):
        return np.min(self.dx)

    def computeRHSE(self, fields):
        H = fields['H']
        rhsE = np.zeros(fields['E'].shape)
        rhsE[1:-1] = - (1.0/self.dxH) * (H[1:] - H[:-1])

        if self.mesh.boundary_label == "PEC":
            rhsE[0] = 0.0
            rhsE[-1] = 0.0
        elif self.mesh.boundary_label  == "Periodic":
            rhsE[0] = - (1.0/self.dxH[0]) * (H[0] - H[-1])           
            rhsE[-1] = rhsE[0]
        else:
            raise ValueError("Invalid boundary label.")
        
        return rhsE

    def computeRHSH(self, fields):
        E = fields['E']
        return - (1.0/self.dx) * (E[1:] - E[:-1])

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}
    
    def isStaggered(self):
        return False

    def buildEvolutionOperator(self):
        NE = len(self.x) 
        N = NE + len(self.xH)
        A = np.zeros((N, N))
        for i in range(N):
            fields = self.buildFields()
            if i < NE:
                fields['E'][i] = 1.0
            else:
                fields['H'][i - NE] = 1.0
            fieldsRHS = self.computeRHS(fields)
            q0 = np.concatenate([ fieldsRHS['E'], fieldsRHS['H'] ])
            A[:, i] = q0[:]
        return A

    def getEnergy(self, field):
        raise ValueError("Not implemented")
        Np = self.number_of_nodes_per_element()
        K = self.mesh.number_of_elements()
        assert field.shape == (Np, K)
        energy = 0.0
        for k in range(K):
            energy += np.inner(
                field[:, k].dot(self.mass),
                field[:, k]*self.jacobian[:, k]
            )

        return energy
