import numpy as np
import math
from .dg1d import *
from .spatialDiscretization import *
from .mesh1d import Mesh1D


class Maxwell1D(SpatialDiscretization):
    def __init__(self, n_order: int, mesh: Mesh1D, fluxType="Upwind"):
        SpatialDiscretization.__init__(self, mesh)
        
        assert n_order > 0
        self.n_order = n_order

        self.fluxType = fluxType


        alpha = 0
        beta = 0

        # Set up material parameters
        self.epsilon = np.ones(mesh.number_of_elements())
        self.mu = np.ones(mesh.number_of_elements())

        self.x = nodes_coordinates(n_order, mesh.EToV, mesh.vx)
        self.nx = normals(mesh.number_of_elements())

        etoe, etof = connect(mesh.EToV)
        self.vmap_m, self.vmap_p, self.vmap_b, self.map_b = build_maps(
            n_order, self.x, etoe, etof)

        r = jacobiGL(alpha, beta, n_order)
        self.fmask, self.fmask_1, self.fmask_2 = buildFMask(r)

        self.mass = mass_matrix(n_order, r)
        self.lift = surface_integral_dg(n_order, r)
        self.diff_matrix = differentiation_matrix(n_order, r)

        self.rx, self.jacobian = geometric_factors(self.x, self.diff_matrix)
        self.f_scale = 1/self.jacobian[self.fmask]

        K = self.mesh.number_of_elements()
        Z_imp = self.get_impedance()

        self.Z_imp_m = Z_imp.transpose().take(self.vmap_m)
        self.Z_imp_p = Z_imp.transpose().take(self.vmap_p)
        self.Z_imp_m = self.Z_imp_m.reshape(
            self.n_fp*self.n_faces, K, order='F')
        self.Z_imp_p = self.Z_imp_p.reshape(
            self.n_fp*self.n_faces, K, order='F')

        self.Y_imp_m = 1.0 / self.Z_imp_m
        self.Y_imp_p = 1.0 / self.Z_imp_p

        self.Z_imp_sum = self.Z_imp_m + self.Z_imp_p
        self.Y_imp_sum = self.Y_imp_m + self.Y_imp_p

    def number_of_nodes_per_element(self):
        return self.n_order + 1

    def get_nodes(self):
        return set_nodes(self.n_order, self.mesh.vx[self.mesh.EToV])

    def get_minimum_node_distance(self):
        return min(np.abs(self.x[0, :] - self.x[1, :]))

    def buildFields(self):
        E = np.zeros([self.number_of_nodes_per_element(),
                      self.mesh.number_of_elements()])
        H = np.zeros(E.shape)

        return {"E": E, "H": H}

    def get_impedance(self):
        Z_imp = np.zeros(self.x.shape)
        for i in range(Z_imp.shape[1]):
            Z_imp[:, i] = np.sqrt(self.mu[i] / self.epsilon[i])

        return Z_imp

    def fieldsOnBoundaryConditions(self, E, H):
        bcType = self.mesh.boundary_label
        if bcType == "PEC":
            Ebc = - E.transpose().take(self.vmap_b)
            Hbc = H.transpose().take(self.vmap_b)
        elif bcType == "PMC":
            Hbc = - H.transpose().take(self.vmap_b)
            Ebc = E.transpose().take(self.vmap_b)
        elif bcType == "SMA":
            Hbc = H.transpose().take(self.vmap_b) * 0.0
            Ebc = E.transpose().take(self.vmap_b) * 0.0
        elif bcType == "Periodic":
            Ebc = E.transpose().take(self.vmap_b[::-1])
            Hbc = H.transpose().take(self.vmap_b[::-1])
        else:
            raise ValueError("Invalid boundary label.")
        return Ebc, Hbc

    def computeFluxE(self, E, H):
        dE, dH = self.computeJumps(E, H)

        if self.fluxType == "Upwind":
            flux_E = 1/self.Z_imp_sum*(self.nx*self.Z_imp_p*dH-dE)
        elif self.fluxType == "Centered":
            flux_E = 1/self.Z_imp_sum*(self.nx*self.Z_imp_p*dH)
        else:
            raise ValueError("Invalid fluxType label")
        return flux_E

    def computeFluxH(self, E, H):
        dE, dH = self.computeJumps(E, H)

        if self.fluxType == "Upwind":
            flux_H = 1/self.Y_imp_sum*(self.nx*self.Y_imp_p*dE-dH)
        elif self.fluxType == "Centered":
            flux_H = 1/self.Y_imp_sum*(self.nx*self.Y_imp_p*dE)
        else:
            raise ValueError("Invalid fluxType label")
        return flux_H

    def computeFlux(self, E, H):
        dE, dH = self.computeJumps(E, H)

        if self.fluxType == "Upwind":
            flux_E = 1/self.Z_imp_sum*(self.nx*self.Z_imp_p*dH-dE)
            flux_H = 1/self.Y_imp_sum*(self.nx*self.Y_imp_p*dE-dH)
        elif self.fluxType == "Centered":
            flux_E = 1/self.Z_imp_sum*(self.nx*self.Z_imp_p*dH)
            flux_H = 1/self.Y_imp_sum*(self.nx*self.Y_imp_p*dE)
        else:
            raise ValueError("Invalid fluxType label")
        return flux_E, flux_H

    def computeJumps(self, E, H):
        Ebc, Hbc = self.fieldsOnBoundaryConditions(E, H)
        dE = E.transpose().take(self.vmap_m) - E.transpose().take(self.vmap_p)
        dH = H.transpose().take(self.vmap_m) - H.transpose().take(self.vmap_p)
        dE[self.map_b] = E.transpose().take(self.vmap_b)-Ebc
        dH[self.map_b] = H.transpose().take(self.vmap_b)-Hbc
        dE = dE.reshape(self.n_fp*self.n_faces,
                        self.mesh.number_of_elements(), order='F')
        dH = dH.reshape(self.n_fp*self.n_faces,
                        self.mesh.number_of_elements(), order='F')
        return dE, dH

    def computeRHSE(self, fields):
        E = fields['E']
        H = fields['H']

        flux_E = self.computeFluxE(E, H)
        rhs_drH = np.matmul(self.diff_matrix, H)
        rhsE = 1/self.epsilon * \
            (np.multiply(-1*self.rx, rhs_drH) +
             np.matmul(self.lift, self.f_scale * flux_E))

        return rhsE

    def computeRHSH(self, fields):
        E = fields['E']
        H = fields['H']

        flux_H = self.computeFluxH(E, H)
        rhs_drE = np.matmul(self.diff_matrix, E)
        rhsH = 1/self.mu * (np.multiply(-1*self.rx, rhs_drE) +
                            np.matmul(self.lift, self.f_scale * flux_H))
        return rhsH

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}

    def convertToVector(self, fields):
        return np.concatenate((
            fields['E'].ravel(order='F'),
            fields['H'].ravel(order='F')
        ))

    def copyVectorToFields(self, vec, fields):
        Np = self.number_of_nodes_per_element()
        K = self.mesh.number_of_elements()
        fields['E'][:, :] = vec[:(vec.size//2)].reshape(Np, K, order='F')
        fields['H'][:, :] = vec[(vec.size//2):].reshape(Np, K, order='F')

    def buildEvolutionOperator(self):
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

    def buildGlobalMassMatrix(self):
        Np = self.number_of_nodes_per_element()
        K = self.mesh.number_of_elements()
        N = 2 * Np * K
        M = np.zeros((N, N))
        for k in range(K):
            ini = k*Np
            end = (k+1)*Np
            M[ini:end, ini:end] = self.mass * self.jacobian[0, k]
        M[int(N/2):, int(N/2):] = M[:int(N/2), :int(N/2)]

        return M

    def getEnergy(self, field):
        '''
        Gets energy stored in field by computing
            field^T * MassMatrix * field * Jacobian.
        for each element and then the sum.
        '''
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
