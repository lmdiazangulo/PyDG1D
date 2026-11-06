import numpy as np

# import sys, os
# sys.path.insert(0, os.path.abspath('..'))

from scipy.linalg import solve_banded

from ..spatialDiscretization import *
from ..dg.mesh1d import Mesh1D

import copy


class FD1D(SpatialDiscretization):
    def __init__(self, mesh: Mesh1D, epsilon=None, lorentz=None):
        SpatialDiscretization.__init__(self, mesh)

        self.x = mesh.vx
        self.xH = (self.x[:-1] + self.x[1:]) / 2.0

        self.dx = self.x[1:] - self.x[:-1]
        self.dxH = self.xH[1:] - self.xH[:-1]

        K = self.mesh.number_of_elements()

        self.c0 = 1.0
        self.tfsf = False
        self.source = None

        # Relative permittivity at infinite frequency (epsilon_inf in the
        # Lorentz model).
        if epsilon is None:
            self.epsilon = np.ones(len(self.x))
        else:
            self.epsilon = self._node_property(epsilon, "permittivity")

        # Single-pole Lorentz dispersion model (Jiang et al. 2018):
        #   dP/dt = J
        #   dJ/dt = -2 gamma J - omega_1^2 P + omega_p^2 E
        #   D = epsilon_inf E + P
        self.lorentz = lorentz
        if lorentz is not None:
            self.omega_p = self._node_property(
                lorentz.get("omega_p", 0.0), "omega_p")
            self.omega_1 = self._node_property(
                lorentz.get("omega_1", 0.0), "omega_1")
            self.gamma = self._node_property(
                lorentz.get("gamma", 0.0), "gamma")
            self.omega_p_sq = self.omega_p**2
            self.omega_1_sq = self.omega_1**2

        self._trapezoidal_cache = None

    def _node_property(self, value, name):
        """Broadcast a scalar or validate a per-node media property."""
        N = len(self.x)
        value = np.asarray(value, dtype=float)
        if value.ndim == 0:
            return np.full(N, float(value))
        if value.shape != (N,):
            raise ValueError(
                "The dimensions of the %s vector must align with the "
                "number of nodes in the mesh." % name)
        return value

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

        fields = {"E": E, "H": H}

        if self.lorentz is not None:
            fields["P"] = np.zeros(E.shape)
            fields["J"] = np.zeros(E.shape)

        return fields

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

        if self.lorentz is not None:
            rhsE[1:-1] -= fields['J'][1:-1]

        rhsE[1:-1] /= self.epsilon[1:-1]

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
                    # rhsE[0] = (0.0 - E[0])/self.dt

                elif label == "PMC":
                    rhsE[0] = - (1.0/self.dxH[0]) * (2 * H[0])

                elif label == "Periodic":
                    rhsE[0] = - (1.0/self.dxH[0]) * (H[0] - H[-1])
                    rhsE[-1] = rhsE[0]

                elif label == "Mur":
                    rhsE[0] = E[1] + \
                        (self.c0 * self.dt - self.dx[0]) / \
                        (self.c0 * self.dt + self.dx[0]) * \
                        (rhsE[1]*self.dt + E[1] - E[0])

                    rhsE[0] -= E[0]
                    rhsE[0] /= self.dt
                
                else: 
                    raise ValueError("Invalid boundary.")

            if bdr == "RIGHT":
                if label == "PEC":
                    rhsE[-1] = 0.0
                    #rhsE[-1] = (0.0 - E[-1])/self.dt

                elif label == "PMC":
                    rhsE[-1] = - (1.0/self.dxH[0]) * (-2 * H[-1])

                elif label == "Periodic":
                    rhsE[0] = - (1.0/self.dxH[0]) * (H[0] - H[-1])
                    rhsE[-1] = rhsE[0]

                elif label == "Mur":

                    rhsE[-1] = E[-2] + \
                        (self.c0 * self.dt - self.dx[-1]) / \
                        (self.c0 * self.dt + self.dx[-1]) * \
                        (rhsE[-2]*self.dt + E[-2] - E[-1])

                    rhsE[-1] -= E[-1]
                    rhsE[-1] /= self.dt
                    
                else: 
                    raise ValueError("Invalid boundary.")

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

        rhs = {'E': rhsE, 'H': rhsH}

        if self.lorentz is not None:
            E = fields['E']
            P = fields['P']
            J = fields['J']
            rhs['P'] = J.copy()
            rhs['J'] = (-2.0 * self.gamma * J
                        - self.omega_1_sq * P
                        + self.omega_p_sq * E)

        return rhs

    def _build_trapezoidal_cache(self, dt):
        """Pre-compute the time independent pieces of the trapezoidal scheme.

        Implements Eq. (4.2) of Jiang et al. (2018) for a single-pole Lorentz
        medium. The polarization variables are eliminated analytically, which
        leaves a symmetric tridiagonal system for the new electric field.
        """
        N = len(self.x)
        h = self.x[1] - self.x[0]
        assert np.allclose(self.dx, h, rtol=1e-8, atol=0.0), \
            "The trapezoidal FDTD scheme requires a uniform mesh."

        if self.lorentz is not None:
            a = self.gamma * dt
            b = 0.5 * self.omega_1_sq * dt
            c = 0.5 * self.omega_p_sq * dt
            kappa = 1.0 + a + 0.5 * b * dt
            eps_star = self.epsilon + 0.5 * dt * c / kappa
        else:
            a = np.zeros(N)
            b = np.zeros(N)
            c = np.zeros(N)
            kappa = np.ones(N)
            eps_star = self.epsilon.copy()

        lam = dt * dt / (4.0 * h * h)

        ab = np.zeros((3, N - 2))
        ab[0, 1:] = -lam
        ab[1, :] = eps_star[1:-1] + 2.0 * lam
        ab[2, :-1] = -lam

        self._trapezoidal_cache = {
            'dt': dt, 'h': h, 'lam': lam, 'ab': ab, 'a': a, 'b': b,
            'c': c, 'kappa': kappa, 'eps_star': eps_star}

        return self._trapezoidal_cache

    def _electric_boundary_value(self, label, inner_new, boundary_old,
                                 inner_old, h, dt):
        if label == "PEC":
            return 0.0
        elif label == "Mur":
            return inner_old + (self.c0 * dt - h) / (self.c0 * dt + h) * \
                (inner_new - boundary_old)
        else:
            raise NotImplementedError(
                "The trapezoidal FDTD scheme supports PEC and Mur "
                "boundaries, not '%s'." % label)

    def computeTrapezoidalStep(self, fields, dt):
        """One step of the (2, 2) trapezoidal FDTD scheme of Eq. (4.2).

        Jiang et al. (2018), "Dispersion analysis of finite difference and
        discontinuous Galerkin schemes for Maxwell's equations in linear
        Lorentz media".  The scheme is unconditionally stable and second
        order accurate in time and space.
        """
        if self.tfsf:
            raise NotImplementedError(
                "The trapezoidal FDTD scheme does not support TFSF sources.")

        N = len(self.x)
        E = fields['E']
        H = fields['H']

        if self.lorentz is not None:
            P = fields['P']
            J = fields['J']
        else:
            P = np.zeros(N)
            J = np.zeros(N)

        cache = self._trapezoidal_cache
        if cache is None or cache['dt'] != dt:
            cache = self._build_trapezoidal_cache(dt)

        h = cache['h']
        lam = cache['lam']
        a, b, c = cache['a'], cache['b'], cache['c']
        kappa = cache['kappa']

        prevE = E.copy()

        SDH = np.zeros(N)
        SDH[1:-1] = -(H[1:] - H[:-1]) / h

        Lap = np.zeros(N)
        Lap[1:-1] = (E[2:] - 2.0 * E[1:-1] + E[:-2]) / (h * h)

        SP = P + 0.5 * dt * J
        SJ = J * (1.0 - a) - b * P + c * E
        p0 = SP + 0.5 * dt * (SJ - b * SP) / kappa

        base = (self.epsilon * E + P) + dt * SDH + lam * h * h * Lap - p0

        Enew = E.copy()
        if N > 2:
            Enew[1:-1] = solve_banded((1, 1), cache['ab'], base[1:-1])

        labels = self.mesh.boundary_label
        Enew[0] = self._electric_boundary_value(
            labels['LEFT'], Enew[1], prevE[0], prevE[1], self.dx[0], dt)
        Enew[-1] = self._electric_boundary_value(
            labels['RIGHT'], Enew[-2], prevE[-1], prevE[-2], self.dx[-1], dt)

        H += -0.5 * dt / h * ((E[1:] - E[:-1]) + (Enew[1:] - Enew[:-1]))

        if self.lorentz is not None:
            fields['J'][:] = (SJ - b * SP + c * Enew) / kappa
            fields['P'][:] = SP + 0.5 * dt * fields['J']

        E[:] = Enew

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
        N = self.number_of_unknowns(      reduceToEssentialDoF=True)
        NE = self.number_of_unknowns('E', reduceToEssentialDoF=True)
        NH = self.number_of_unknowns('H', reduceToEssentialDoF=True)

        L_E = np.zeros((N, N))
        L_E[:NE, :NE] = np.eye(NE)
        L_H = np.zeros((N, N))
        L_H[NE:, NE:] = np.eye(NH)
        
        h = self.x[1] - self.x[0]
        assert np.allclose(h, self.x[1:] - self.x[:-1])
        M = np.eye(N)*h
        P = 0.5*( L_E.dot(M).dot(L_E)
            + 0.5*L_H.dot(M).dot(L_H).dot(G)
            + 0.5*G.T.dot(L_H).dot(M).dot(L_H))
        
        f = copy.deepcopy(fields)
        
        if self.mesh.boundary_label['LEFT'] == 'Periodic':
            f['E'] = np.zeros(len(fields['E'])-1)
            f['E'][:] = fields['E'][:-1]
        else:
            raise ValueError("Only implemented for periodic.")
            
        q = self.fieldsAsStateVector(f)
        return q.T.dot(P).dot(q)

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
