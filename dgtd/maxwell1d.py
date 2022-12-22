import numpy as np
import scipy.special
import numpy.linalg 
import math
from dg1d import build_maps, normals, geometric_factors, surface_integral_dg, differentiation_matrix
from meshUtils import mesh_generator

rk4a = np.array([])
rk4b = np.array([])
rk4c = np.array([])

def maxwellRHS1D(E, H, eps, mu):
    Z_imp   = np.sqrt(mu / eps)
    
    dE      = np.zeros([n_fp*n_faces,k_elem])
    dE      = E[vmap_m] - E[vmap_p]
    
    dH      = np.zeros([n_fp*n_faces,k_elem])
    dH      = H[vmap_m] - H[vmap_p]
    
    # Define field differences at faces
    Z_imp_m = np.zeros([n_fp*n_faces,k_elem]) 
    Z_imp_m = Z_imp[vmap_m]
    Z_imp_p = np.zeros([n_fp*n_faces,k_elem])
    Z_imp_p = Z_imp[vmap_p]
    Y_imp_m = np.zeros([n_fp*n_faces,k_elem])
    Y_imp_m = np.linealg.inv(Z_imp_m)
    Y_imp_p = np.zeros([n_fp*n_faces,k_elem])
    Y_imp_p = np.linealg.inv(Z_imp_p)
    
    # Homogeneous boundary conditions
    Ebc         = -E[vmap_b]
    dE[map_b]   = E[map_b] - Ebc
    Hbc         = H[vmap_b]
    dH[map_b]   = H[vmap_b] - Hbc
    
    # Evaluate upwind fluxes
    Z_imp_sum   = Z_imp_m + Z_imp_p
    Z_imp_mult  = np.multiply(nx, Z_imp_p)
    Z_imp_mult2 = np.multiply(Z_imp_mult, dH) - dE
    Z_imp_O     = np.multiply(Z_imp_sum, Z_imp_mult2)
    flux_E      = 1/Z_imp_O
    
    Y_imp_sum   = Y_imp_m + Y_imp_p
    Y_imp_mult  = np.multiply(nx, Y_imp_p)
    Y_imp_mult2 = np.multiply(Y_imp_mult, dE) - dH
    Y_imp_O     = np.multiply(Y_imp_sum, Y_imp_mult2)
    flux_H      = 1/Y_imp_O
    
    # Compute right hand sides of the PDE’s
    F_scale     = 1/J[fmask]
    rsh_drH     = np.matmul(Diff_matrix, H) 
    rsh_fsflE    = np.multiply(F_scale, flux_E)
    
    rsh_drE     = np.matmul(Diff_matrix, E) 
    rsh_fsflH   = np.multiply(F_scale, flux_H)
    
    rhs_E       = 1/eps*(np.multiply(-1*rx, rsh_drH) + np.matmul(lift, rsh_fsflE))
    rhs_H       = 1/mu*(np.multiply(-1*rx, rsh_drE) + np.matmul(lift, rsh_fsflH))
    
    return [rhs_E, rhs_H]


def maxwell1D(E, H, eps, mu, final_time):
    
    time = 0
    
    # Runge Kutta storage
    res_E = zeros(n_p, k_elements) 
    res_H = zeros(n_p, k_elements)
    
    # Compute time step size
    x_min   = min(np.abs(x[0,:] - x[1,:]))
    CFL     = 1.0
    dt      = CFL * x_min
    N_steps = math.ceil(final_time/dt)
    dt      = final_time/N_steps
    
    # Outer time step loop
    for t_step in range (1, N_steps):
        for INTRK in range (0,4):
            [rhs_E, rhs_H] = MaxwellRHS1D(E, H, eps, mu)
            
            res_E = rk4a(INTRK)*res_E + dt*rhs_E
            res_H = rk4a(INTRK)*res_H + dt*rhs_H
            
            E = E + rk4b(INTRK)*res_E
            H = H + rk4b(INTRK)*res_H
            
        time = time + dt
               
    return E, H