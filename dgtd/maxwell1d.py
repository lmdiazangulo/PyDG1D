import numpy as np
import math
from .dg1d import *
from .meshUtils import Mesh1D

rk4a_ = np.array([0,	-0.417890474499852,	-1.19215169464268,	-1.69778469247153,	-1.51418344425716])
rk4b = np.array([0.149659021999229,	0.379210312999627,	0.822955029386982,	0.699450455949122,	0.153057247968152])
rk4c = np.array([0,	0.149659021999229,	0.370400957364205,	0.622255763134443,	0.958282130674690])

class SpatialDiscretization:
    def __init__(self, n_order: int, mesh: Mesh1D):
        assert n_order > 0
        assert mesh.number_of_elements() > 0

        self.mesh = mesh
        self.n_order = n_order

        self.n_faces = 2
        self.n_fp = 1

        alpha = 0
        beta = 0

        # Set up material parameters
        self.epsilon = np.ones(mesh.number_of_elements())
        self.mu      = np.ones(mesh.number_of_elements())
        
    
        r = jacobiGL(alpha, beta, n_order)
        jacobi_p = jacobi_polynomial(r, alpha, beta, n_order)
        vander = vandermonde_1d(n_order, r)
        self.x = nodes_coordinates(n_order, mesh.EToV, mesh.vx) 
        self.nx = normals(mesh.number_of_elements()) 
        
        etoe, etof = connect(mesh.EToV)
        self.vmap_m, self.vmap_p, self.vmap_b, self.map_b = build_maps(n_order, self.x, etoe, etof)
        
        self.fmask_1 = np.where(np.abs(r+1)<1e-10)[0][0]
        self.fmask_2 = np.where(np.abs(r-1)<1e-10)[0][0]
        self.fmask = [self.fmask_1,self.fmask_2]

        self.lift = surface_integral_dg(n_order, vander)
        self.diff_matrix = differentiation_matrix(n_order, r, vander)
        self.rx , self.jacobian = geometric_factors(self.x, self.diff_matrix)
        

    def number_of_nodes_per_element(self):
        return self.n_order + 1

    def get_nodes(self):
        return set_nodes_1d(self.n_order, self.mesh.vx[self.mesh.EToV])
    
    def get_impedance(self):
        Z_imp = np.zeros(self.x.shape)
        for i in range(Z_imp.shape[1]):
            Z_imp[:,i] = np.sqrt(self.mu[i] / self.epsilon[i])
        
        return Z_imp

# class Options:
#     def __init__(self, sp:SpatialDiscretization):
#         self.sp = sp
    
#     def fieldsOnBoundaryConditions(self, bcType):
#         if bcType == "PEC":
#             self.Ebc = - self.E.transpose().take(self.sp.vmap_b)
#             self.Hbc =   self.H.transpose().take(self.sp.vmap_b)
#         elif bcType == "PMC":
#             self.Hbc = - self.H.transpose().take(self.sp.vmap_b)
#             self.Ebc =   self.E.transpose().take(self.sp.vmap_b)
#         else:
#             self.Ebc = self.E.transpose().take(self.sp.vmap_b[::-1]) 
#             self.Hbc = self.H.transpose().take(self.sp.vmap_b[::-1])
#         return self.Ebc, self.Hbc
    
#     def type_of_flux(self, fluxType):     
#         if fluxType == "Upwind":
#             self.flux_E = 1/self.Z_imp_sum*(self.sp.nx*self.Z_imp_p*self.dH-self.dE)
#             self.flux_H = 1/self.Y_imp_sum*(self.sp.nx*self.Y_imp_p*self.dE-self.dH)
#         else:
#             self.flux_E = 1/self.Z_imp_sum*(self.sp.nx*self.Z_imp_p*self.dH)
#             self.flux_H = 1/self.Y_imp_sum*(self.sp.nx*self.Y_imp_p*self.dE)
#         return self.flux_E, self.flux_H


class MaxwellDriver:
    def __init__(self, sp: SpatialDiscretization):
        self.sp = sp

        # Compute time step size
        x_min = min(np.abs(sp.x[0, :] - sp.x[1, :]))
        CFL = 1.0
        self.dt = CFL * x_min / 2      
        self.time = 0.0

        self.E = np.zeros([sp.number_of_nodes_per_element(), sp.mesh.number_of_elements()])
        self.H = np.zeros(self.E.shape)

        K = sp.mesh.number_of_elements()
        Z_imp = sp.get_impedance()
        
        self.Z_imp_m = Z_imp.transpose().take(sp.vmap_m)
        self.Z_imp_p = Z_imp.transpose().take(sp.vmap_p)
        self.Z_imp_m = self.Z_imp_m.reshape(sp.n_fp*sp.n_faces, K, order='F') 
        self.Z_imp_p = self.Z_imp_p.reshape(sp.n_fp*sp.n_faces, K, order='F') 
        
        self.Y_imp_m = 1.0 / self.Z_imp_m
        self.Y_imp_p = 1.0 / self.Z_imp_p

        self.Z_imp_sum = self.Z_imp_m + self.Z_imp_p
        self.Y_imp_sum = self.Y_imp_m + self.Y_imp_p
        

    def step(self, dt=0.0):           
        n_p = self.sp.number_of_nodes_per_element()
        k = self.sp.mesh.number_of_elements()
        
        if (dt == 0.0):
            dt = self.dt

        res_E = np.zeros([n_p, k])
        res_H = np.zeros([n_p, k])
        for INTRK in range(0, 5):
            rhs_E, rhs_H = self.computeRHS1D()

            res_E = rk4a_[INTRK]*res_E + dt*rhs_E
            res_H = rk4a_[INTRK]*res_H + dt*rhs_H

            self.E += rk4b[INTRK]*res_E
            self.H += rk4b[INTRK]*res_H

        self.time += dt

    def run(self, final_time):
        for t_step in range(1, math.ceil(final_time/self.dt)):
            self.step()

    def fieldsOnBoundaryConditions(self, bcType):
        if bcType == "PEC":
            Ebc = - self.E.transpose().take(self.sp.vmap_b)
            Hbc =   self.H.transpose().take(self.sp.vmap_b)
        elif bcType == "PMC":
            Hbc = - self.H.transpose().take(self.sp.vmap_b)
            Ebc =   self.E.transpose().take(self.sp.vmap_b)
        else:
            Ebc = self.E.transpose().take(self.sp.vmap_b[::-1]) 
            Hbc = self.H.transpose().take(self.sp.vmap_b[::-1])
        return Ebc, Hbc
    
    def type_of_flux(self, fluxType):     
        if fluxType == "Upwind":
            flux_E = 1/self.Z_imp_sum*(self.sp.nx*self.Z_imp_p*self.dH-self.dE)
            flux_H = 1/self.Y_imp_sum*(self.sp.nx*self.Y_imp_p*self.dE-self.dH)
        else:
            flux_E = 1/self.Z_imp_sum*(self.sp.nx*self.Z_imp_p*self.dH)
            flux_H = 1/self.Y_imp_sum*(self.sp.nx*self.Y_imp_p*self.dE)
        return flux_E, flux_H

    def computeRHS1D(self):
        sp = self.sp
        E = self.E
        H = self.H
        K = sp.mesh.number_of_elements()
                
        #Homogeneous boundary conditions for periodic condition
        Ebc, Hbc = self.fieldsOnBoundaryConditions("PEC")
        
        #Define field differences at faces
        self.dE = E.transpose().take(sp.vmap_m) - E.transpose().take(sp.vmap_p)
        self.dH = H.transpose().take(sp.vmap_m) - H.transpose().take(sp.vmap_p)

        self.dE[sp.map_b] = E.transpose().take(sp.vmap_b)-Ebc
        self.dH[sp.map_b] = H.transpose().take(sp.vmap_b)-Hbc
        self.dE = self.dE.reshape(sp.n_fp*sp.n_faces, K, order='F') 
        self.dH = self.dH.reshape(sp.n_fp*sp.n_faces, K, order='F') 

        # Evaluate fluxes
        flux_E, flux_H = self.type_of_flux("Central")

        # Compute right hand sides of the PDE’s
        f_scale = 1/sp.jacobian[sp.fmask]
        rhs_drH = np.matmul(sp.diff_matrix, H)
        rhs_drE = np.matmul(sp.diff_matrix, E)
        rhs_E = 1/sp.epsilon* (np.multiply(-1*sp.rx, rhs_drH) + np.matmul(sp.lift, f_scale * flux_E))
        rhs_H = 1/sp.mu     * (np.multiply(-1*sp.rx, rhs_drE) + np.matmul(sp.lift, f_scale * flux_H))

        return rhs_E, rhs_H