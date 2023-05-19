import numpy as np
import matplotlib.pyplot as plt

from .dg2d import *
from .mesh2d import Mesh2D
from .lserk4 import *


class Maxwell2D(SpatialDiscretization):
    def __init__(self, n_order: int, mesh: Mesh2D, fluxType="Upwind"):
        assert n_order > 0
        assert mesh.number_of_elements() > 0

        self.n_order = n_order
        self.n_fp = n_order + 1
        self.n_faces = 3
        
        self.mesh = mesh
        self.fluxType = fluxType

        self.epsilon = np.ones(mesh.number_of_elements())
        self.mu = np.ones(mesh.number_of_elements())

        r, s = xy_to_rs(*set_nodes_in_equilateral_triangle(n_order))
        self.Dr, self.Ds = derivateMatrix(n_order, r, s)
        self.x, self.y = nodes_coordinates(n_order, mesh)

        self.lift = lift(n_order)

        eToE, eToF = mesh.connectivityMatrices()
        va = self.mesh.EToV[:, 0]
        vb = self.mesh.EToV[:, 1]
        vc = self.mesh.EToV[:, 2]
        self.rx, self.sx, self.ry, self.sy, self.jacobian = geometricFactors(
            self.x, self.y, self.Dr, self.Ds)
        
        fmask, _, _, _ = buildFMask(n_order)

        self.nx, self.ny, sJ = normals(
            self.x, self.y,
            self.Dr, self.Ds,
            n_order
        )
        self.f_scale = sJ/self.jacobian[fmask.ravel('F')]

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
        vmapM = np.full([n_fp, n_faces, k_elem], 0)
        vmapP = np.full([n_fp, n_faces, k_elem], 0)
        mapM = np.arange(k_elem*n_fp*n_faces)
        mapP = np.reshape(mapM, (n_fp, n_faces, k_elem))

        # find index of face nodes with respect to volume node ordering
        for k1 in range(k_elem):
            for f1 in range(n_faces):
                vmapM[:, f1, k1] = node_ids[Fmask[:, f1], k1]

        one = np.ones(n_fp)
        EToE, EToF = msh.connectivityMatrices()
        for k1 in range(k_elem):
            for f1 in range(n_faces):
                # find neighbor
                k2 = EToE[k1, f1]
                f2 = EToF[k1, f1]

                # reference length of edge
                v1 = msh.EToV[k1, f1]
                v2 = msh.EToV[k1, np.mod(f1, n_faces)]
                refd = np.sqrt(
                    (msh.vx[v1]-msh.vx[v2])**2 + (msh.vy[v1]-msh.vy[v2])**2
                )

                # find find volume node numbers of left and right nodes
                vidM = vmapM[:, f1, k1]
                vidP = vmapM[:, f2, k2]
                x1 = np.outer(self.x.ravel('F')[vidM], one)
                y1 = np.outer(self.y.ravel('F')[vidM], one)
                x2 = np.outer(self.x.ravel('F')[vidP], one)
                y2 = np.outer(self.y.ravel('F')[vidP], one)

                # Compute distance matrix
                distance = np.sqrt(
                    np.abs((x1 - x2.transpose())**2 + (y1-y2.transpose())**2))
                idM, idP = np.where(distance <= NODETOL*refd)
                vmapP[idM, f1, k1] = vidP[idP]
                mapP[idM, f1, k1] = idP + (f2-1)*n_fp+(k2-1)*n_faces*n_fp

        vmapM = vmapM.ravel('F')
        vmapP = vmapP.ravel('F')
        vmapB = vmapM[vmapP == vmapM]
        mapB = np.where(vmapP == vmapM)[0]

        self.vmapM = vmapM
        self.vmapP = vmapP
        self.vmapB = vmapB
        self.mapB = mapB

    def get_minimum_node_distance(self):
        points, _ = jacobi_gauss(0, 0, self.n_order)
        return abs(points[0]-points[1])
    
    def get_dt_scale(self):

        r, s = xy_to_rs(*set_nodes_in_equilateral_triangle(self.n_order))
        vmask1 = np.where(np.abs(s+r+2) < NODETOL)[0]
        vmask2 = np.where(np.abs(r-1) < NODETOL)[0]
        vmask3 = np.where(np.abs(s-1) < NODETOL)[0]
        vmask  = np.array([vmask1, vmask2, vmask3]).transpose()

        vx = self.x[np.squeeze(vmask.reshape(-1, 1)), :]
        vy = self.y[np.squeeze(vmask.reshape(-1, 1)), :]

        len1 = np.sqrt((vx[0,:]-vx[1,:])**2+(vy[0,:]-vy[1,:])**2)
        len2 = np.sqrt((vx[1,:]-vx[2,:])**2+(vy[1,:]-vy[2,:])**2)
        len3 = np.sqrt((vx[2,:]-vx[0,:])**2+(vy[2,:]-vy[0,:])**2)
        sper = (len1 + len2 + len3)/2.0
        area = np.sqrt(sper*(sper-len1)*(sper-len2)*(sper-len3))

        dtscale = area/sper

        return dtscale

    def get_mesh(self):
        return self.mesh

    def number_of_nodes_per_element(self):
        return int((self.n_order + 1) * (self.n_order + 2) / 2)
    
    def buildEvolutionOperator(self):
        Np = self.number_of_nodes_per_element()
        K = self.mesh.number_of_elements()
        N = 3 * Np * K
        A = np.zeros((N,N))
        for i in range(N):
            fields = self.buildFields()
            node = i % Np
            elem = int(np.floor(i / Np)) % K
            if i < N/3:
                fields['Ez'][node, elem] = 1.0
            elif i > N/3 and i < 2*N/3:
                fields['Hx'][node, elem] = 1.0
            else:
                fields['Hy'][node, elem] = 1.0
            fieldsRHS = self.computeRHS(fields)
            q0 = np.vstack([
                fieldsRHS['Ez'].reshape(Np*K,1, order='F'), 
                fieldsRHS['Hx'].reshape(Np*K,1, order='F'),
                fieldsRHS['Hy'].reshape(Np*K,1, order='F')
            ])
            A[:,i] = q0[:,0]

        return A

    def buildFields(self):
        Hx = np.zeros([self.number_of_nodes_per_element(),
                       self.mesh.number_of_elements()])
        Hy = np.zeros(Hx.shape)
        Ez = np.zeros(Hx.shape)

        return {'Hx': Hx, 'Hy': Hy, 'Ez': Ez}

    def computeFlux(self, Hx, Hy, Ez):
        dHx, dHy, dEz = self.computeJumps(Hx, Hy, Ez)
        flux_Hx =  self.ny * dEz
        flux_Hy = -self.nx * dEz
        flux_Ez = -self.nx * dHy + self.ny * dHx

        if self.fluxType == "Upwind":
            ndotdH = self.nx * dHx + self.ny * dHy
            flux_Hx += ndotdH * self.nx - dHx
            flux_Hy += ndotdH * self.ny - dHy
            flux_Ez -= dEz
        elif self.fluxType == "Centered":
            pass
        else:
            raise ValueError("Invalid flux type.")

        return flux_Hx, flux_Hy, flux_Ez

    def fieldsOnBoundaryConditions(self, Hx, Hy, Ez):

        bcType = self.mesh.boundary_label
        if bcType == "PEC":
            Hbcx = Hx.transpose().take(self.vmapB)
            Hbcy = Hy.transpose().take(self.vmapB)
            Ebcz = - Ez.transpose().take(self.vmapB)
        elif bcType == "PMC":
            Hbcx = - Hx.transpose().take(self.vmapB)
            Hbcy = - Hy.transpose().take(self.vmapB)
            Ebcz = Ez.transpose().take(self.vmapB)
        elif bcType == "SMA":
            Hbcx = Hx.transpose().take(self.vmapB) * 0.0
            Hbcy = Hx.transpose().take(self.vmapB) * 0.0
            Ebcz = Ez.transpose().take(self.vmapB) * 0.0
        elif bcType == "Periodic":
            Hbcx = Hx.transpose().take(self.vmapB[::-1])
            Hbcy = Hy.transpose().take(self.vmapB[::-1])
            Ebcz = Ez.transpose().take(self.vmapB[::-1])
        else:
            raise ValueError("Invalid boundary label.")
        return Hbcx, Hbcy, Ebcz

    def computeJumps(self, Hx, Hy, Ez):
        Hbcx, Hbcy, Ebcz = self.fieldsOnBoundaryConditions(Hx, Hy, Ez)
        dHx = Hx.transpose().take(self.vmapM) - Hx.transpose().take(self.vmapP)
        dHy = Hy.transpose().take(self.vmapM) - Hy.transpose().take(self.vmapP)
        dEz = Ez.transpose().take(self.vmapM) - Ez.transpose().take(self.vmapP)

        dHx[self.mapB] = Hx.transpose().take(self.vmapB) - Hbcx
        dHy[self.mapB] = Hy.transpose().take(self.vmapB) - Hbcy
        dEz[self.mapB] = Ez.transpose().take(self.vmapB) - Ebcz

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

        flux_Hx, flux_Hy, flux_Ez = self.computeFlux(Hx, Hy, Ez)

        rhs_Ezx, rhs_Ezy = grad(
            self.Dr, self.Ds, Ez, self.rx, self.sx, self.ry, self.sy
        )
        rhs_CuHz = curl(
            self.Dr, self.Ds, Hx, Hy, self.rx, self.sx, self.ry, self.sy
        )

        # missing material epsilon/mu
        rhs_Hx = -rhs_Ezy  + np.matmul(self.lift, self.f_scale * flux_Hx)/2.0
        rhs_Hy =  rhs_Ezx  + np.matmul(self.lift, self.f_scale * flux_Hy)/2.0
        rhs_Ez =  rhs_CuHz + np.matmul(self.lift, self.f_scale * flux_Ez)/2.0

        return {'Hx': rhs_Hx, 'Hy': rhs_Hy, 'Ez': rhs_Ez}
    
    def plot_field(self, Nout, field, fig):
        # Build equally spaced grid on reference triangle
        Npout = int((Nout+1)*(Nout+2)/2)
        rout = np.zeros((Npout))
        sout = np.zeros((Npout))
        counter = np.zeros((Nout+1, Nout+1))
        sk = 0
        for n in range (Nout+1):
            for m in range (Nout+1-n):
                rout[sk] = -1 + 2*m/Nout
                sout[sk] = -1 + 2*n/Nout
                counter[n,m] = sk
                sk += 1
                
        # Build matrix to interpolate field data to equally spaced nodes
        Vout = vandermonde(Nout, rout, sout)
        interp = Vout.dot(np.linalg.inv(Vout))

        # Build triangulation of equally spaced nodes on reference triangle
        tri = np.array([], dtype=int).reshape(0,3)
        for n in range (Nout+1):
            for m in range (Nout-n):
                v1 = counter[n,m]
                v2 = counter[n,m+1]
                v3 = counter[n+1,m]
                v4 = counter[n+1,m+1]
                if v4:
                    tri = np.vstack(([v1, v2, v3],[v2, v4, v3]))
                else:
                    tri = np.vstack((tri, [[v1, v2, v3]]))

        # Build triangulation for all equally spaced nodes on all elements
        TRI = np.array([], dtype=int).reshape(0,3)
        for k in range(self.mesh.number_of_elements()):
            TRI = np.vstack((TRI, tri+(k)*Npout))

        # Interpolate node coordinates and field to equally spaced nodes
        xout = interp.dot(self.x) 
        yout = interp.dot(self.y) 
        uout = interp.dot(field)

        # Render and format solution field
        
        plt.tricontourf(
            xout.ravel('F'), 
            yout.ravel('F'), 
            uout.ravel('F'), 
            triangles=TRI, cmap='viridis'
        )