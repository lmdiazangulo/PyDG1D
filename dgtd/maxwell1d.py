import numpy as np
import scipy.special
import numpy.linalg
import math
from dgtd.dg1d import *
from dgtd.meshUtils import Mesh1D

rk4a = np.array([])
rk4b = np.array([])
rk4c = np.array([])


class SpatialDiscretization:
    def __init__(self, n_order: int, mesh: Mesh1D):
        assert n_order > 0
        assert mesh.number_of_elements() > 0

        self.mesh = mesh
        self.n_order = n_order

        n_faces = 2
        n_fp = 1

        alpha = 0
        beta = 0
        r = jacobiGL(alpha, beta, n_order)
        vander = vandermonde_1d(n_order, r)

        self.lift = surface_integral_dg(n_order, vander)
        self.diff_matrix = differentiation_matrix(n_order, r, vander)

    def number_of_nodes_per_element(self):
        return self.n_order + 1

    def get_nodes(self):
        return set_nodes_1d(self.n_order, self.mesh.vx)


def maxwellRHS1D(E, H, eps, mu, sp: SpatialDiscretization):
    Z_imp = np.sqrt(mu / eps)

    K = sp.mesh.number_of_elements()

    dE = np.zeros([n_fp*n_faces, K])
    dE = E[vmap_m] - E[vmap_p]

    dH = np.zeros([n_fp*n_faces, K])
    dH = H[vmap_m] - H[vmap_p]

    # Define field differences at faces
    Z_imp_m = np.zeros([n_fp*n_faces, K])
    Z_imp_m = Z_imp[vmap_m]
    Z_imp_p = np.zeros([n_fp*n_faces, K])
    Z_imp_p = Z_imp[vmap_p]
    Y_imp_m = np.zeros([n_fp*n_faces, K])
    Y_imp_m = np.linealg.inv(Z_imp_m)
    Y_imp_p = np.zeros([n_fp*n_faces, K])
    Y_imp_p = np.linealg.inv(Z_imp_p)

    # Homogeneous boundary conditions
    Ebc = -E[vmap_b]
    dE[map_b] = E[map_b] - Ebc
    Hbc = H[vmap_b]
    dH[map_b] = H[vmap_b] - Hbc

    # Evaluate upwind fluxes
    Z_imp_sum = Z_imp_m + Z_imp_p
    Z_imp_mult = np.multiply(nx, Z_imp_p)
    Z_imp_mult2 = np.multiply(Z_imp_mult, dH) - dE
    Z_imp_O = np.multiply(Z_imp_sum, Z_imp_mult2)
    flux_E = 1/Z_imp_O

    Y_imp_sum = Y_imp_m + Y_imp_p
    Y_imp_mult = np.multiply(nx, Y_imp_p)
    Y_imp_mult2 = np.multiply(Y_imp_mult, dE) - dH
    Y_imp_O = np.multiply(Y_imp_sum, Y_imp_mult2)
    flux_H = 1/Y_imp_O

    # Compute right hand sides of the PDE’s
    f_scale = 1/J[fmask]
    rsh_drH = np.matmul(sp.diff_matrix, H)
    rsh_fsflE = np.multiply(f_scale, flux_E)

    rsh_drE = np.matmul(diff_matrix, E)
    rsh_fsflH = np.multiply(f_scale, flux_H)

    rhs_E = 1/eps*(np.multiply(-1*rx, rsh_drH) + np.matmul(lift, rsh_fsflE))
    rhs_H = 1/mu*(np.multiply(-1*rx, rsh_drE) + np.matmul(lift, rsh_fsflH))

    return [rhs_E, rhs_H]


def maxwell1D(E, H, eps, mu, final_time, sp: SpatialDiscretization):

    # Compute time step size
    x = sp.get_nodes()
    x_min = min(np.abs(x[0, :] - x[1, :]))
    CFL = 1.0
    dt = CFL * x_min
    N_steps = math.ceil(final_time/dt)
    dt = final_time/N_steps

    n_p = sp.number_of_nodes_per_element()
    k = sp.mesh.number_of_elements()

    # Runge Kutta storage
    res_E = np.zeros(n_p, k)
    res_H = np.zeros(n_p, k)

    # Outer time step loop
    time = 0
    for t_step in range(1, N_steps):
        for INTRK in range(0, 4):
            [rhs_E, rhs_H] = maxwellRHS1D(E, H, eps, mu, sp)

            res_E = rk4a(INTRK)*res_E + dt*rhs_E
            res_H = rk4a(INTRK)*res_H + dt*rhs_H

            E = E + rk4b(INTRK)*res_E
            H = H + rk4b(INTRK)*res_H

        time = time + dt

    return E, H
