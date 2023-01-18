import numpy as np
import scipy.special
import numpy.linalg
import math
from dgtd.dg1d import *
from dgtd.meshUtils import Mesh1D

rk4a = np.array([0,	-0.417890474499852,	-1.19215169464268,	-1.69778469247153,	-1.51418344425716])
rk4b = np.array([0.149659021999229,	0.379210312999627,	0.822955029386982,	0.699450455949122,	0.153057247968152])
rk4c = np.array([0,	0.149659021999229,	0.370400957364205,	0.622255763134443,	0.958282130674690])

# Set inicial conditions

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
        jacobi_p = jacobi_polynomial(r, alpha, beta, n_order)
        vander = vandermonde_1d(n_order, r)
        self.nodes_c = nodes_coordinates(n_order, mesh.EToV, mesh.vx) 
        self.nx = normals(mesh.number_of_elements()) 
        
        etoe, etof = connect(mesh.EToV)
        self.vmap_m, self.vmap_p, self.vmap_b, self.map_b = build_maps(n_order, self.nodes_c, etoe, etof)
        
        self.fmask_1 = np.where(np.abs(r+1)<1e-10)[0][0]
        self.fmask_2 = np.where(np.abs(r-1)<1e-10)[0][0]
        self.fmask = [self.fmask_1,self.fmask_2]

        self.lift = surface_integral_dg(n_order, vander)
        self.diff_matrix = differentiation_matrix(n_order, r, vander)
        self.rx , self.jacobian = geometric_factors(self.nodes_c, self.diff_matrix)
        

    def number_of_nodes_per_element(self):
        return self.n_order + 1

    def get_nodes(self):
        return set_nodes_1d(self.n_order, self.mesh.vx[self.mesh.EToV])


def maxwellRHS1D(E, H, eps, mu, sp: SpatialDiscretization):
    # n_fp = 2
    # n_faces = 1
    Z_imp = np.sqrt(mu / eps)

    K = sp.mesh.number_of_elements()      

    dE = np.zeros([n_fp*n_faces, K])
    dE = E.transpose().take(sp.vmap_m) - E.transpose().take(sp.vmap_p)

    dH = np.zeros([n_fp*n_faces, K])
  #  dH = H[sp.vmap_m] - H[sp.vmap_p]
    dH = H.transpose().take(sp.vmap_m) - H.transpose().take(sp.vmap_p)

    # Define field differences at faces
#    Z_imp_m = np.zeros([n_fp*n_faces, K])
    Z_imp_m = Z_imp.transpose().take(sp.vmap_m)
#    Z_imp_p = np.zeros([n_fp*n_faces, K])
    Z_imp_p = Z_imp.transpose().take(sp.vmap_p)
 #   Y_imp_m = np.zeros([n_fp*n_faces, K])
    Y_imp_m = 1/Z_imp_m #np.linalg.inv(Z_imp_m)
 #   Y_imp_p = np.zeros((n_fp*n_faces, K))
    Y_imp_p = 1/Z_imp_p #np.linalg.inv(Z_imp_p)

    #assignaments

    
    # Homogeneous boundary conditions
    Ebc = -1*E.transpose().take(sp.vmap_b)
    dE[sp.map_b] = E.transpose().take(sp.vmap_b) - Ebc
    Hbc = H.transpose().take(sp.vmap_b)
    dH[sp.map_b] = H.transpose().take(sp.vmap_b) - Hbc

    reshaped_nx = sp.nx.reshape((Y_imp_p.shape))
    # Evaluate upwind fluxes
    Z_imp_sum = Z_imp_m + Z_imp_p
   # Z_imp_mult = reshaped_nx*Z_imp_p
   # Z_imp_mult2 = np.multiply(Z_imp_mult, dH) - dE
   # Z_imp_O = np.multiply(Z_imp_sum, Z_imp_mult2)
    flux_E = 1/Z_imp_sum*(reshaped_nx*Z_imp_p*dH-dE)

    Y_imp_sum = Y_imp_m + Y_imp_p

   
   # Y_imp_mult = np.multiply(reshaped_nx, Y_imp_p)
   # Y_imp_mult2 = np.multiply(Y_imp_mult, dE) - dH
   # Y_imp_O = np.multiply(Y_imp_sum, Y_imp_mult2)
    flux_H = 1/Y_imp_sum*(reshaped_nx*Y_imp_p*dE-dH)

    # Compute right hand sides of the PDE’s
    f_scale = 1/sp.jacobian[sp.fmask]
    rsh_drH = np.matmul(sp.diff_matrix, H)
    reshaped_flux_E = flux_E.reshape((f_scale.shape))
    rsh_fsflE = np.multiply(f_scale, reshaped_flux_E)
    
    reshaped_flux_H = flux_H.reshape((f_scale.shape))
    rsh_drE = np.matmul(sp.diff_matrix, E)
    rsh_fsflH = np.multiply(f_scale, reshaped_flux_H)

    rhs_E = 1/eps*(np.multiply(-1*sp.rx, rsh_drH) + np.matmul(sp.lift, rsh_fsflE))
    rhs_H = 1/mu* (np.multiply(-1*sp.rx, rsh_drE) + np.matmul(sp.lift, rsh_fsflH))

    return [rhs_E, rhs_H]


def maxwell1D(E, H, eps, mu, final_time, sp: SpatialDiscretization):

    # Compute time step size
    x = sp.nodes_c
    x_min = min(np.abs(x[0, :] - x[1, :]))
    CFL = 0.8
    dt = CFL * x_min
    N_steps = math.ceil(final_time/dt)
    dt = final_time/N_steps

    n_p = sp.number_of_nodes_per_element()
    k = sp.mesh.number_of_elements()

    # Runge Kutta storage
    res_E = np.zeros([n_p, k])
    res_H = np.zeros([n_p, k])

    # Outer time step loop
    time = 0
    for t_step in range(1, N_steps):
        for INTRK in range(0, 4):
            [rhs_E, rhs_H] = maxwellRHS1D(E, H, eps, mu, sp)

            res_E = rk4a[INTRK]*res_E + dt*rhs_E
            res_H = rk4a[INTRK]*res_H + dt*rhs_H

            E = E + rk4b[INTRK]*res_E
            H = H + rk4b[INTRK]*res_H

        time = time + dt

    return E, H


# def maxwell_driver()