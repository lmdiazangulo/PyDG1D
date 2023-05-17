import numpy as np
import math

from .dg2d import *
from .mesh2d import Mesh2D
from .lserk4 import *


class Maxwell2D(SpatialDiscretization):
    def __init__(self, n_order: int, mesh: Mesh2D, fluxType="Upwind"):
        assert n_order > 0
        assert mesh.number_of_elements() > 0

        self.n_order = n_order

        self.mesh = mesh
        self.fluxType = fluxType

        self.epsilon = np.ones(mesh.number_of_elements())
        self.mu = np.ones(mesh.number_of_elements())

        x, y = set_nodes_in_equilateral_triangle(n_order)
        r, s = xy_to_rs(x, y)
        Dr, Ds = derivateMatrix(n_order, r, s)
        self.x, self.y = nodes_coordinates(n_order, mesh)

        self.lift = lift(n_order)

        eToE, eToF = mesh.connectivityMatrices()
        va = self.mesh.EToV[:, 0]
        vb = self.mesh.EToV[:, 1]
        vc = self.mesh.EToV[:, 2]
        self.rx, self.sx, self.ry, self.sy, self.jacobian = geometricFactors(
            x, y, Dr, Ds)
        self.nx, self.ny, sJ = normals(
            self.x, self.y,
            Dr, Ds,
            n_order, self.mesh.number_of_elements()
        )

        self.buildMaps()

    def buildMaps(self):
        '''
        function [mapM, mapP, vmapM, vmapP, vmapB, mapB] = BuildMaps2D
        Purpose: Connectivity and boundary tables in the K # of Np elements        
        '''
        N = self.n_order
        msh = self.mesh
        k_elem = self.mesh.number_of_elements()
        n_p = self.number_of_nodes_per_element()
        n_faces = 3
        n_fp = N+1
        
        # mask defined in globals
        Fmask, _, _, _ = buildFMask(N)
        
        # number volume nodes consecutively
        node_ids = np.reshape(np.arange(k_elem*n_p), [n_p, k_elem], 'F')
        vmapM   = np.full([n_fp, n_faces, k_elem], 0)
        vmapP   = np.full([n_fp, n_faces, k_elem], 0) 
        mapM    = np.arange(k_elem*n_fp*n_faces)
        mapP    = np.reshape(mapM, (n_fp, n_faces, k_elem))

        # find index of face nodes with respect to volume node ordering
        for k1 in range(k_elem):
            for f1 in range (n_faces):
                vmapM[:,f1, k1] = node_ids[Fmask[:,f1], k1]
            
        one = np.ones(n_fp)
        EToE, EToF = msh.connectivityMatrices()
        for k1 in range(k_elem):
            for f1 in range(n_faces):
                # find neighbor
                k2 = EToE[k1,f1]
                f2 = EToF[k1,f1]
                
                # reference length of edge
                v1 = msh.EToV[k1,f1]
                v2 = msh.EToV[k1, np.mod(f1,n_faces)]
                refd = np.sqrt( 
                    (msh.vx[v1]-msh.vx[v2])**2 + (msh.vy[v1]-msh.vy[v2])**2 
                )

                # find find volume node numbers of left and right nodes 
                vidM = vmapM[:, f1, k1]
                vidP = vmapM[:, f2, k2]   
                x1 = np.outer(self.x.ravel('F')[vidM], one)
                y1 = np.outer(self.y.ravel('F')[vidM], one)
                x2 = np.outer(self.x.ravel('F')[vidP], one)
                y2 = np.outer(self.y.ravel('F')[vidP],one)

                # Compute distance matrix
                distance = np.sqrt(np.abs((x1 - x2.transpose())**2 + (y1-y2.transpose())**2))
                idM, idP = np.where(distance <= NODETOL*refd)
                vmapP[idM, f1, k1] = vidP[idP]
                mapP[idM, f1, k1] = idP + (f2-1)*n_fp+(k2-1)*n_faces*n_fp

        self.vmapM = vmapM.ravel('F')
        self.vmapP = vmapP.ravel('F')
        self.vmapB = vmapM[vmapP == vmapM]

    def get_minimum_node_distance(self):
        points, _ = jacobi_gauss(0, 0, self.n_order)
        return abs(points[0]-points[1])

    def number_of_nodes_per_element(self):
        return int((self.n_order + 1) * (self.n_order + 2) / 2)

    def buildFields(self):
        Hx = np.zeros([self.number_of_nodes_per_element(),
                       self.mesh.number_of_elements()])
        Hy = np.zeros(Hx.shape)
        Ez = np.zeros(Hx.shape)

        return {'Hx': Hx, 'Hy': Hy, 'Ez': Ez}

    def computeFlux(self):  # Missing Z and Y from materials

        flux_Hx = -self.ny*self.dEz
        flux_Hy = -self.nx*self.dEz
        flux_Ez = -self.nx*self.dHy + self.ny*self.dHx

        if self.fluxType == "Upwind":

            alpha = 1.0
            ndotdH = self.nx*self.dHx+self.ny*self.dHy

            flux_Hx += ndotdH*self.nx-self.dHx
            flux_Hy += ndotdH*self.ny-self.dHy
            flux_Ez += alpha*(-self.dEz)

        return flux_Hx, flux_Hy, flux_Ez

    def fieldsOnBoundaryConditions(self, Hx, Hy, Ez):

        bcType = self.mesh.boundary_label
        if bcType == "PEC":
            Hbcx = Hx.transpose().take(self.vmap_b)
            Hbcx = Hy.transpose().take(self.vmap_b)
            Ebcz = - Ez.transpose().take(self.vmap_b)
        elif bcType == "PMC":
            Hbcx = - Hx.transpose().take(self.vmap_b)
            Hbcy = - Hy.transpose().take(self.vmap_b)
            Ebcz = Ez.transpose().take(self.vmap_b)
        elif bcType == "SMA":
            Hbcx = Hx.transpose().take(self.vmap_b) * 0.0
            Hbcy = Hx.transpose().take(self.vmap_b) * 0.0
            Ebcz = Ez.transpose().take(self.vmap_b) * 0.0
        elif bcType == "Periodic":
            Hbcx = Hx.transpose().take(self.vmap_b[::-1])
            Hbcy = Hy.transpose().take(self.vmap_b[::-1])
            Ebcz = Ez.transpose().take(self.vmap_b[::-1])
        else:
            raise ValueError("Invalid boundary label.")
        return Hbcx, Hbcy, Ebcz

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

    def computeRHS(self, fields):
        Hx = fields['Hx']
        Hy = fields['Hy']
        Ez = fields['Ez']

        Hbcx, Hbcy, Ebcz = self.fieldsOnBoundaryConditions(
            Hx, Hy, Ez)  # todo - fobc
        self.dHx, self.dHy, self.dEz = self.computeJumps(
            Hbcx, Hbcy, Ebcz, Hx, Hy, Ez)

        flux_Hx, flux_Hy, flux_Ez = self.computeFlux()

        f_scale = 1/self.jacobian[self.fmask]
        rhs_Ezx, rhs_Ezy = self.grad2D(
            self.Dr, self.Ds, Ez, geometricFactors(self.x, self.y, self.Dr, self.Ds))
        rhs_CuHx, rhs_CuHy, rhs_CuHz = self.curl2D(
            self.Dr, self.Ds, Hx, Hy, geometricFactors(self.x, self.y, self.Dr, self.Ds))

        # missing material epsilon/mu
        rhs_Hx = -rhs_Ezy + np.matmul(self.lift, f_scale * flux_Hx)/2.0
        rhs_Hy = rhs_Ezx + np.matmul(self.lift, f_scale * flux_Hy)/2.0
        rhs_Ez = rhs_CuHz + np.matmul(self.lift, f_scale * flux_Ez)/2.0

        return {'Hx': rhs_Hx, 'Hy': rhs_Hy, 'Ez': rhs_Ez}
