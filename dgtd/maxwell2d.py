import numpy as np
import math
from .dg2d import *
from .mesh2d import Mesh2D

rk4a = np.array([0, -0.417890474499852, -1.19215169464268,
                -1.69778469247153,	-1.51418344425716])
rk4b = np.array([0.149659021999229,	0.379210312999627,
                0.822955029386982,	0.699450455949122,	0.153057247968152])
rk4c = np.array([0,	0.149659021999229,	0.370400957364205,
                0.622255763134443,	0.958282130674690])

class SpatialDiscretization2D:
    def __init__(self, n_order: int, mesh: Mesh2D, fluxType="Upwind"):
        assert n_order > 0
        assert mesh.number_of_elements() > 0

        self.mesh = mesh
        self.n_order = n_order
        self.fluxType = fluxType

        self.n_faces = 3
        self.n_fp = n_order+1

        # Set up material parameters
        self.epsilon = np.ones(mesh.number_of_elements())
        self.mu = np.ones(mesh.number_of_elements())

        x, y = set_nodes(n_order)
        r, s = xy_to_rs(x, y)
        vander = vandermonde(n_order, r, s)
        Dr, Ds = Dmatrices2D(n_order, r, s, vander)
        self.x, self.y = nodes_coordinates(n_order, mesh)

        self.fmask_1 = np.where(np.abs(s+1) < 1e-10)[0][0]
        self.fmask_2 = np.where(np.abs(r+s) < 1e-10)[0][0]
        self.fmask_3 = np.where(np.abs(r+1) < 1e-10)[0][0]
        self.fmask = [self.fmask_1, self.fmask_2, self.fmask_3].transpose()

        self.lift = lift(n_order)

        etoe, etof = connect2D(mesh.EToV) #todo - connect2d
        va = self.mesh.EToV[:,0]
        vb = self.mesh.EToV[:,1] 
        vc = self.mesh.EToV[:,2]
        self.rx, self.sx, self.ry, self.sy, self.jacobian = geometricFactors(x, y, Dr, Ds) #todo - geometric factors
        self.nx, self.ny, sJ = normals2D() #todo - normals
        Fscale = sJ/(self.J[self.fmask,:])

        K = self.mesh.number_of_elements()

    def number_of_nodes_per_element(self):
        return (self.n_order + 1) * (self.n_order + 2) / 2
        
    def computeFlux(self): #Missing Z and Y for materials

        flux_Hx = -self.ny*self.dEz 
        flux_Hy = -self.nx*self.dEz
        flux_Ez = -self.nx*self.dHy + self.ny*self.dHx
        if self.fluxType == "Upwind":
            alpha = 1.0

            flux_Hx += self.ndotdH*self.nx-self.dHx
            flux_Hy += self.ndotdH*self.ny-self.dHy
            flux_Ez += alpha*(-self.dEz)

        return flux_Hx, flux_Hy, flux_Ez
    
    def computeJumps(self, Hbcx, Hbcy, Ebcz, Hx, Hy, Ez):

        dHx = Hx.transpose().take(self.vmap_m) - Hx.transpose().take(self.vmap_p)
        dHy = Hy.transpose().take(self.vmap_m) - Hy.transpose().take(self.vmap_p)
        dEz = Ez.transpose().take(self.vmap_m) - Ez.transpose().take(self.vmap_p)

        dHx[self.map_b] = Hx.transpose().take(self.vmap_b) - Hbcx
        dHy[self.map_b] = Hy.transpose().take(self.vmap_b) - Hbcy
        dEz[self.map_b] = Ez.transpose().take(self.vmap_b) - Ebcz

        dHx = dHx.reshape(self.n_fp*self.n_faces,
                        self.mesh.number_of_elements(), order='F')
        dHy = dHy.reshape(self.n_fp*self.n_faces,
                        self.mesh.number_of_elements(), order='F')
        dEz = dEz.reshape(self.n_fp*self.n_faces,
                        self.mesh.number_of_elements(), order='F')
        
        return dHx, dHy, dEz
    
    def computeRHS(self, Hx, Hy, Ez):

        Hbcx, Hbcy, Ebcz = self.fieldsOnBoundaryConditions(Hx, Hy, Ez) #todo - fobc
        self.dHx, self.dHy, self.dEz = self.computeJumps(Hbcx, Hbcy, Ebcz, Hx, Hy, Ez)

        flux_Hx, flux_Hy, flux_Ez = self.computeFlux()

        f_scale = 1/self.jacobian[self.fmask]
        rhs_Ezx, rhs_Ezy = Grad2D(Ez) #todo - Grad2D
        rhs_CuHx, rhs_CuHy, rhs_CuHz = Curl2D(Hx, Hy) #todo - Curl2D

        #missing material epsilon/mu
        rhs_Hx = -rhs_Ezy  + np.matmul(self.lift, f_scale * flux_Hx)/2.0
        rhs_Hy =  rhs_Ezx  + np.matmul(self.lift, f_scale * flux_Hy)/2.0
        rhs_Ez =  rhs_CuHz + np.matmul(self.lift, f_scale * flux_Ez)/2.0

        return rhs_Hx, rhs_Hy, rhs_Ez

        