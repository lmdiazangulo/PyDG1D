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

    def fieldsOnBoundaryConditions(self, E, H):
        raise ValueError("Not implemented")
        # bcType = self.mesh.boundary_label
        # if bcType == "PEC":
        #     Ebc = - E.transpose().take(self.vmap_b)
        #     Hbc = H.transpose().take(self.vmap_b)
        # elif bcType == "PMC":
        #     Hbc = - H.transpose().take(self.vmap_b)
        #     Ebc = E.transpose().take(self.vmap_b)
        # elif bcType == "SMA":
        #     Hbc = H.transpose().take(self.vmap_b) * 0.0
        #     Ebc = E.transpose().take(self.vmap_b) * 0.0
        # elif bcType == "Periodic":
        #     Ebc = E.transpose().take(self.vmap_b[::-1])
        #     Hbc = H.transpose().take(self.vmap_b[::-1])
        # else:
        #     raise ValueError("Invalid boundary label.")
        # return Ebc, Hbc

    def computeRHSE(self, fields):
        H = fields['H']
        rhsE = np.zeros(fields['E'].shape)
        rhsE[1:-1] = - (1.0/self.dxH) * (H[1:] - H[:-1])
        return rhsE

    def computeRHSH(self, fields):
        E = fields['E']
        return - (1.0/self.dx) * (E[1:] - E[:-1])

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}

    def buildEvolutionOperator(self, sorting='EH'):
        raise ValueError("Not implemented")
        Np = self.number_of_nodes_per_element()
        K = self.mesh.number_of_elements()
        N = 2 * Np * K
        A = np.zeros((N, N))
        for i in range(N):
            fields = self.buildFields()
            node = i % Np
            elem = int(np.floor(i / Np)) % K
            if i < N/2:
                fields['E'][node, elem] = 1.0
            else:
                fields['H'][node, elem] = 1.0
            fieldsRHS = self.computeRHS(fields)
            q0 = np.vstack([
                fieldsRHS['E'].reshape(Np*K, 1, order='F'),
                fieldsRHS['H'].reshape(Np*K, 1, order='F')
            ])
            A[:, i] = q0[:, 0]
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
