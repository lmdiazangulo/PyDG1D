import numpy as np

# import sys, os
# sys.path.insert(0, os.path.abspath('..'))

from ..spatialDiscretization import *
from ..dg.mesh1d import Mesh1D

import copy


class FD1D(SpatialDiscretization):
    def __init__(self, mesh: Mesh1D):
        SpatialDiscretization.__init__(self, mesh)

        self.x = mesh.vx
        self.xH = (self.x[:-1] + self.x[1:]) / 2.0

        self.dx = self.x[1:] - self.x[:-1]
        self.dxH = self.xH[1:] - self.xH[:-1]

        K = self.mesh.number_of_elements()

        self.c0 = 1.0
        self.tfsf = False
        self.source = None

    def TFSF_conditions(self, setup):

        self.tfsf = True
        self.source = setup["source"]
        self.left_TF_limit = (np.absolute(self.x - setup["left"])).argmin()
        self.right_TF_limit = (np.absolute(self.x - setup["right"])).argmin()
        if not "source" in setup.keys() or not "left" in setup.keys() or not "right" in setup.keys():
            raise ValueError('Missing TFSF setup variables')

    def buildFields(self):
        E = np.zeros(self.x.shape)
        H = np.zeros(self.xH.shape)

        if (self.source != None and self.tfsf):
            self.buildIncidentFields()

        return {"E": E, "H": H}

    def buildIncidentFields(self):
        self.Einc = np.ndarray(self.x.shape)
        self.Einc[:] = self.source(self.x[:])

        self.Eprev = np.zeros(self.x.shape)

        self.Hinc = np.ndarray(self.xH.shape)
        self.Hinc[:] = self.source(self.xH[:] - 0.5*self.dt)

    def get_minimum_node_distance(self):
        return np.min(self.dx)

    def computeRHSE(self, fields):
        H = fields['H']
        E = fields['E']
        rhsE = np.zeros(fields['E'].shape)

        rhsE[1:-1] = - (1.0/self.dxH) * (H[1:] - H[:-1])

        if self.tfsf == True:

            self.updateIncidentFieldE()
            rhsE[self.left_TF_limit] += (1.0/self.dxH[0]) * \
                self.Hinc[self.left_TF_limit-1]
            rhsE[self.right_TF_limit] -= (1.0/self.dxH[0]) * \
                self.Hinc[self.right_TF_limit]

        for bdr, label in self.mesh.boundary_label.items():

            if bdr == "LEFT":
                if label == "PEC":
                    rhsE[0] = 0.0

                if label == "PMC":
                    rhsE[0] = - (1.0/self.dxH[0]) * (2 * H[0])

                if label == "Periodic":
                    rhsE[0] = - (1.0/self.dxH[0]) * (H[0] - H[-1])
                    rhsE[-1] = rhsE[0]

                if label == "Mur":

                    rhsE[0] = E[1] + \
                        (self.c0 * self.dt - self.dx[0]) / \
                        (self.c0 * self.dt + self.dx[0]) * \
                        (rhsE[1] - E[0])

                    rhsE[0] -= E[0]
                    rhsE[0] /= self.dt

            if bdr == "RIGHT":
                if label == "PEC":
                    rhsE[-1] = 0.0

                if label == "PMC":
                    rhsE[-1] = - (1.0/self.dxH[0]) * (-2 * H[-1])

                if label == "Periodic":
                    rhsE[0] = - (1.0/self.dxH[0]) * (H[0] - H[-1])
                    rhsE[-1] = rhsE[0]

                if label == "Mur":

                    rhsE[-1] = E[-2] + \
                        (self.c0 * self.dt - self.dx[0]) / \
                        (self.c0 * self.dt + self.dx[0]) * \
                        (rhsE[-2] - E[-1])

                    rhsE[-1] -= E[-1]
                    rhsE[-1] /= self.dt

        return rhsE

    def computeRHSH(self, fields):
        E = fields['E']
        rhsH = - (1.0/self.dx) * (E[1:] - E[:-1])

        if self.tfsf == True:
            self.updateIncidentFieldH()
            rhsH[self.left_TF_limit -
                 1] += (1.0/self.dx[0]) * self.Einc[self.left_TF_limit]
            rhsH[self.right_TF_limit] -= (1.0/self.dx[0]) * \
                self.Einc[self.right_TF_limit]

        return rhsH

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}

    def updateIncidentFieldE(self):
        self.Einc[1:-1] = self.Einc[1:-1] - self.dt * \
            (1.0/self.dxH) * (self.Hinc[1:] - self.Hinc[:-1])

        self.Einc[0] = \
            self.Eprev[1] - \
            (self.c0 * self.dt - self.dx[0]) / \
            (self.c0 * self.dt + self.dx[0]) * \
            (self.Einc[1] - self.Eprev[0])

        self.Einc[-1] = \
            self.Eprev[-2] - \
            (self.c0 * self.dt - self.dx[0]) / \
            (self.c0 * self.dt + self.dx[0]) * \
            (self.Einc[-2] - self.Eprev[-1])

        self.Eprev[:] = self.Einc[:]

    def updateIncidentFieldH(self):
        self.Hinc = self.Hinc - self.dt * \
            (1.0/self.dx) * (self.Einc[1:] - self.Einc[:-1])

    def isStaggered(self):
        return True

    def number_of_nodes_per_element(self):
        return 1

    def number_of_unknowns(self, field='all', reduceToEssentialDoF=False):
        if field == 'all':
            return self.number_of_unknowns('E', reduceToEssentialDoF) \
                + self.number_of_unknowns('H', reduceToEssentialDoF)
        elif field == 'E':
            if reduceToEssentialDoF:
                if self.mesh.boundary_label['LEFT'] == 'Periodic' and \
                        self.mesh.boundary_label['RIGHT'] == 'Periodic':
                    return len(self.x) - 1
                elif self.mesh.boundary_label['LEFT'] == 'PEC' and \
                        self.mesh.boundary_label['LEFT'] == 'PEC':
                    return len(self.x) - 2
                else:
                    raise ValueError('Invalid boundary labels for reduction.')
            else:
                return len(self.x)
        elif field == 'H':
            return len(self.xH)
        else:
            raise ValueError('Invalid field label.')

    def setFieldWithIndex(self, fields, i, val):
        NE = fields['E'].size
        if i < NE:
            fields['E'][i] = val
        else:
            fields['H'][i - NE] = val
        return fields

    def reduceToEssentialDoF(self, A):
        NE = self.buildFields()['E'].size
        if self.mesh.boundary_label['LEFT'] == 'Periodic'\
                and self.mesh.boundary_label['RIGHT'] == 'Periodic':
            A = np.delete(A, NE-1, 0)
            A[:, 0] += A[:, NE-1]
            A = np.delete(A, NE-1, 1)
        elif self.mesh.boundary_label['LEFT'] == 'PEC'\
                and self.mesh.boundary_label['RIGHT'] == 'PEC':
            A = np.delete(A, NE-1, 0)
            A = np.delete(A, NE-1, 1)
            A =  np.delete(A, 0, 0)
            A = np.delete(A, 0, 1)
        else:
            raise ValueError(
                "Periodic conditions must be ensured at both ends")

        return A

    def buildEvolutionOperator(self, reduceToEssentialDoF=True):
        N = self.number_of_unknowns()
        A = np.zeros((N, N))
        for i in range(N):
            fields = self.buildFields()
            self.setFieldWithIndex(fields, i, 1.0)
            fieldsRHS = self.computeRHS(fields)
            q0 = np.concatenate([fieldsRHS['E'], fieldsRHS['H']])
            A[:, i] = q0[:]

        if reduceToEssentialDoF:
            A = self.reduceToEssentialDoF(A)
        return A

    def getEnergy(self, field, removeLast=False):
        h = self.x[1] - self.x[0]
        assert np.allclose(h, self.x[1:] - self.x[:-1])
        f = copy.deepcopy(field)
        if removeLast:
            f = np.zeros(len(field)-1)
            f[:] = field[:-1]
    
        return 0.5 * h * f.T.dot(f)

    def getTotalEnergy(self, G, fields):
        dt = self.dt

        Gdt = G*dt

        N = self.number_of_unknowns(      reduceToEssentialDoF=True)
        NE = self.number_of_unknowns('E', reduceToEssentialDoF=True)
        NH = self.number_of_unknowns('H', reduceToEssentialDoF=True)

        S_E = np.zeros((N, N))
        S_E[:NE, :NE] = np.eye(NE)
        S_H = np.zeros((N, N))
        S_H[NE:, NE:] = np.eye(NH)
        
        h = self.x[1] - self.x[0]
        assert np.allclose(h, self.x[1:] - self.x[:-1])

        M = np.eye(N)*h
        V = 0.5*(M + 0.5*S_E.T.dot(Gdt.T).dot(S_H) + 0.5*S_H.T.dot(Gdt).dot(S_E))
        
        if self.mesh.boundary_label['LEFT'] != 'Periodic':
            raise ValueError("Only implemented for periodic.")
            
        f = copy.deepcopy(fields)
        f['E'] = np.zeros(len(fields['E'])-1)
        f['E'][:] = fields['E'][:-1]
        q = self.fieldsAsStateVector(f)
        return q.T.dot(V).dot(q)

    def reorder_by_elements(self, A):
        # Assumes that the original array contains all DoF ordered as:
        # [ E_0, ..., E_{NE-1}, H_0, ..., H_{NH-1} ]
        N = A.shape[0]
        NE = len(self.x)
        NH = len(self.xH)
        if self.mesh.boundary_label['LEFT'] == 'Periodic' and \
                self.mesh.boundary_label['RIGHT'] == 'Periodic':
            NE -= 1
        else:
            raise ValueError(
                "Periodic conditions must be ensured at both ends")
        if NE != NH:
            raise ValueError(
                "Unable to order by elements with different size fields.")
        N = NE + NH
        new_order = np.zeros(N, dtype=int) - 1

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
