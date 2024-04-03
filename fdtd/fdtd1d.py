import numpy as np

# import sys, os
# sys.path.insert(0, os.path.abspath('..'))


if __package__ == 'pydg1d.fdtd':
    from ..dgtd.spatialDiscretization import *
    from ..dgtd.mesh1d import Mesh1D
else:
    from dgtd.spatialDiscretization import *
    from dgtd.mesh1d import Mesh1D


class FDTD1D(SpatialDiscretization):
    def __init__(self, mesh: Mesh1D):
        SpatialDiscretization.__init__(self, mesh)

        self.x = mesh.vx
        self.xH = (self.x[:-1] + self.x[1:]) / 2.0

        self.dx = self.x[1:] - self.x[:-1]
        self.dxH = self.xH[1:] - self.xH[:-1]

        if self.mesh.boundary_label == 'Periodic':
            self.x = mesh.vx[:-1]

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

        if self.mesh.boundary_label == "PEC":
            rhsE[1:-1] = - (1.0/self.dxH) * (H[1:] - H[:-1])
            rhsE[0] = 0.0
            rhsE[-1] = 0.0

        elif self.mesh.boundary_label  == "Periodic":
            rhsE[1:] = - (1.0/self.dxH) * (H[1:] - H[:-1])
            rhsE[0] = - (1.0/self.dxH[0]) * (H[0] - H[-1])         
            #rhsE[-1] = rhsE[0]

        elif self.mesh.boundary_label =="PMC":
            rhsE[1:-1] = - (1.0/self.dxH) * (H[1:] - H[:-1])
            rhsE[0] = rhsE[0] - (2*H[0])
            rhsE[-1] = rhsE[-1] - (-2*H[-1])

        elif self.mesh.boundary_label == "PML": #[WIP]       
            boundary_low = [0, 0]
            boundary_high = [0, 0]

            rhsE[0] = boundary_low.pop(0)
            boundary_low.append(rhsE[1])

            rhsE[-1] = boundary_high.pop(0)
            boundary_high.append(rhsE[-2])
        
        return rhsE

    def computeRHSH(self, fields):
        E = fields['E']
        rhsH = np.zeros(fields['H'].shape)
        if self.mesh.boundary_label == "PEC":
            rhsH = - (1.0/self.dx) * (E[1:] - E[:-1])

        elif self.mesh.boundary_label == "Periodic":
            rhsH[:-1] = - (1.0/self.dx[:-1]) * (E[1:] - E[:-1])
            rhsH[-1] = - (1.0/self.dx[0]) * (E[0] - E[-1])

        elif self.mesh.boundary_label == "PMC": #[WIP]
             rhsH[0] = 0.0
             rhsH[-1] = 0.0
            
        #elif self.mesh.boundary_label == "PML": #[WIP]

        return rhsH

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}

    def isStaggered(self):
        return True

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
            q0 = np.concatenate([fieldsRHS['E'], fieldsRHS['H']])
            A[:, i] = q0[:]
        return A

    def reorder_array(self, A, ordering):
        # Assumes that the original array contains all DoF ordered as:
        # [ E_0, ..., E_{NE-1}, H_0, ..., H_{NH-1} ]
        N = A.shape[0]
        NE = len(self.x)
        NH = len(self.xH)
        if NE != NH:
            raise ValueError("Unable to order by elements with different size fields.")
        N = NE + NH
        new_order = np.zeros(N, dtype=int) - 1
        if ordering == 'byElements':
            for i in range(N):
                if i < NE:
                    new_order[2*i] = i
                else:
                    new_order[2*int(i - NE)+1] = i
        if (len(A.shape) == 1):
            A1 = [A[i] for i in new_order]
        elif (len(A.shape) == 2):
            A1 = [[A[i][j] for j in new_order] for i in new_order]
        return np.array(A1)

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
