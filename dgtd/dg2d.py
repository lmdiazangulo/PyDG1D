import numpy as np

import dgtd.dg1d as dg1d
import dgtd.mesh2d as mesh

N_FACES = 3

alpopt = np.array([0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
        1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258])

NODETOL = 1e-12

def warpFactor(N, rout):
    '''
        Purpose: Compute scaled warp function at order N based on rout interpolation nodes
    '''
    # Compute LGL and equidistant node distribution
    LGLr = dg1d.jacobiGL(0,0,N)
    req  = np.linspace(-1, 1, N+1)

    # Compute V based on req
    Veq = dg1d.vandermonde(N,req)

    # Evaluate Lagrange polynomial at rout
    Nr = len(rout)
    Pmat = np.zeros((N+1,Nr))
    for i in range(N+1):
        Pmat[i] = np.transpose(dg1d.jacobi_polynomial(rout, 0, 0, i))
    Lmat = np.linalg.solve(Veq.transpose(), Pmat)

    # Compute warp factor
    warp = Lmat.transpose().dot(LGLr - req)

    # Scale factor
    zerof = np.abs(rout) < 1.0 - 1.0e-10
    sf = 1.0 - (zerof*rout)**2

    warp = warp / sf + warp * (zerof-1)
    return warp


def set_nodes(N):
    '''
        x, y = set_nodes(N)
        Purpose  : Compute (x,y) nodes in equilateral triangle for polynomial of order N
    '''
    Np = int((N+1)*(N+2)/2)

    L1 = np.zeros(Np)
    L2 = np.zeros(Np)
    L3 = np.zeros(Np)
    sk = 0
    for n in range(1,N+2):
        for m in range(1,N+3-n):
            L1[sk] = (n-1) / N 
            L3[sk] = (m-1) / N
            sk += 1
    L2 = 1.0 - L1 - L3

    x = -L2+L3
    y = (-L2-L3+2*L1)/np.sqrt(3.0)

    # Compute blending function at each node for each edge
    blend1 = 4.0 * L2 * L3
    blend2 = 4.0 * L1 * L3
    blend3 = 4.0 * L1 * L2

    # Amount of warp for each node, for each edge
    warpf1 = warpFactor(N,L3-L2)
    warpf2 = warpFactor(N,L1-L3)
    warpf3 = warpFactor(N,L2-L1)

    if N<16:
        alpha = alpopt[N]
    else:
        alpha = 5/3

    # Combine blend and warp
    warp1 = blend1 * warpf1 * (1 + (alpha*L1)**2)
    warp2 = blend2 * warpf2 * (1 + (alpha*L2)**2)
    warp3 = blend3 * warpf3 * (1 + (alpha*L3)**2)

    # Accumulate deformations associated with each edge
    x = x + 1.0 * warp1 + np.cos(2*np.pi/3)*warp2 + np.cos(4.0*np.pi/3)*warp3
    y = y + 0.0 * warp1 + np.sin(2*np.pi/3)*warp2 + np.sin(4.0*np.pi/3)*warp3

    return x, y


def xy_to_rs(x,y):

    L1 = (np.sqrt(3.0)*y+1.0)/3.0
    L2 = (-3.0*x - np.sqrt(3.0)*y + 2.0)/6.0
    L3 = ( 3.0*x - np.sqrt(3.0)*y + 2.0)/6.0

    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

    return r, s


def simplex_polynomial(a, b, i: int, j: int):
    '''
        % function [P] = Simplex2DP(a,b,i,j);
        % Purpose : Evaluate 2D orthonormal polynomial
        %           on simplex at (a,b) of order (i,j).
    '''
    h1 = dg1d.jacobi_polynomial(a,     0, 0, i)
    h2 = dg1d.jacobi_polynomial(b, 2*i+1, 0, j)
    
    P = np.sqrt(2.0) * h1.transpose() * h2.transpose() * (1-b)**i
    return P

def rs_to_ab(r, s):
    '''
        % function [a,b] = rstoab(r,s)
        % Purpose : Transfer from (r,s) -> (a,b) coordinates in triangle
    '''
    Np = len(r)
    a = np.zeros(Np)

    for n in range(Np):
        if not np.isclose(s[n], 1.0):
            a[n] = 2.0 * (1.0+r[n]) / (1.0-s[n]) - 1
        else:
            a[n] = -1.0
    b = s

    return a, b

def vandermonde(N: int, r, s):
    '''
        % function [V2D] = Vandermonde2D(N, r, s);
        % Purpose : Initialize the 2D Vandermonde Matrix,  V_{ij} = phi_j(r_i, s_i);
    '''
    Np = int ((N+1) * (N+2) / 2)
    V = np.zeros((len(r), Np))

    a, b = rs_to_ab(r, s)
    sk = 0
    for i in range(N+1):
        for j in range(N-i+1):
            pol = simplex_polynomial(a, b, i, j)
            V[:, sk] = pol
            sk += 1

    return V

def derivateMatrix(N: int, r, s, V):

    Vr, Vs = gradVandermonde(N, r, s)
    Dr = np.matmul(Vr,np.linalg.inv(V))
    Ds = np.matmul(Vs,np.linalg.inv(V))

    return Dr, Ds

def gradVandermonde(N: int, r, s):

    V2Dr = np.zeros((len(r),int((N+1)*(N+2)/2)))
    V2Ds = np.zeros((len(r),int((N+1)*(N+2)/2)))

    a, b = rs_to_ab(r,s)

    sk = 0
    for i in range(0, N+1, 1):
        for j in range(0, N-i+1, 1):
            V2Drtmp, V2Dstmp = gradSimplexP(a, b, i, j)
            V2Dr[:,sk] = V2Drtmp
            V2Ds[:,sk] = V2Dstmp
            sk += 1

    return V2Dr, V2Ds

def gradSimplexP(a, b, i: int, j: int):

    fa  = dg1d.jacobi_polynomial     (a, 0, 0, i)
    dfa = dg1d.jacobi_polynomial_grad(a, 0, 0, i)
    gb  = dg1d.jacobi_polynomial     (b, 2.0*i+1.0, 0, j)
    dgb = dg1d.jacobi_polynomial_grad(b, 2.0*i+1.0, 0, j)

    dmodedr = dfa*gb
    bcoeff = (0.5*(1-b))**(i-1)

    if (i>0):
        dmodedr = dmodedr*bcoeff
    
    dmodeds = dfa*(gb*(0.5*(1+a)))

    if (i>0):
        dmodeds = dmodeds*bcoeff

    tmp = dgb*(0.5*(1-b))**(i)

    if (i>0):
        gbcoeff = 0.5*i*gb
        tmp -= gbcoeff*bcoeff

    dmodeds += fa*tmp

    dmodedr *= 2**(i+0.5)
    dmodeds *= 2**(i+0.5)

    return dmodedr, dmodeds

def nodes_coordinates(N, msh: mesh.Mesh2D):
    """
    Defined to be able to define methods depedent grid properties
    """

    va = msh.EToV[:,0].transpose()
    vb = msh.EToV[:,1].transpose()
    vc = msh.EToV[:,2].transpose()

    r, s  = xy_to_rs(*set_nodes(N))

    x = 0.5*(- np.outer(r+s, msh.vx[va]) + np.outer(1+r, msh.vx[vb]) + np.outer(1+s, msh.vx[vc]))
    y = 0.5*(- np.outer(r+s, msh.vy[va]) + np.outer(1+r, msh.vy[vb]) + np.outer(1+s, msh.vy[vc]))

    return x, y


def lift(N):
    r, s  = xy_to_rs(*set_nodes(N))

    fmask1 = np.where(np.abs(s+1) < NODETOL)[0]
    fmask2 = np.where(np.abs(r+s) < NODETOL)[0]
    fmask3 = np.where(np.abs(r+1) < NODETOL)[0]
    Fmask  = np.array([fmask1, fmask2, fmask3]).transpose()

    Nfp = int(N + 1)
    Np = int((N+1) * (N+2) / 2)
    Emat = np.zeros((Np, N_FACES*Nfp))

    # face 0
    faceR = r[Fmask[:,0]]
    V1D = dg1d.vandermonde(N, faceR)
    massEdge1 = np.linalg.inv(V1D.dot(V1D.transpose()))
    Emat[Fmask[:,0], :Nfp] = massEdge1

    # face 1
    faceR = r[Fmask[:,1]]
    V1D = dg1d.vandermonde(N, faceR)
    massEdge2 = np.linalg.inv(V1D.dot(V1D.transpose()))
    Emat[Fmask[:,1], Nfp:(2*Nfp)] = massEdge2

    # face 2
    faceS = s[Fmask[:,2]]
    V1D = dg1d.vandermonde(N, faceS)
    massEdge3 = np.linalg.inv(V1D.dot(V1D.transpose()))
    Emat[Fmask[:,2], (2*Nfp):] = massEdge3

    # inv(mass matrix)*\I_n (L_i,L_j)_{edge_n}
    V = vandermonde(N, r, s)
    lift = V.dot(V.transpose().dot(Emat))
    return lift

def geometricFactors(x, y, Dr, Ds):
    xr = np.matmul(Dr,x)
    xs = np.matmul(Ds,x)
    yr = np.matmul(Dr,y)
    ys = np.matmul(Ds,y)
    J = -xs*yr + xr*ys
    rx = ys/J
    sx =-yr/J 
    ry =-xs/J
    sy = xr/J
    return rx, sx, ry, sy, J

def normals(x, y, Dr, Ds, N, K):

    xr = np.matmul(Dr,x)
    xs = np.matmul(Ds,x)
    yr = np.matmul(Dr,y)
    ys = np.matmul(Ds,y)

    r, s  = xy_to_rs(*set_nodes(N))

    fmask1 = np.where(np.abs(s+1) < NODETOL)[0]
    fmask2 = np.where(np.abs(r+s) < NODETOL)[0]
    fmask3 = np.where(np.abs(r+1) < NODETOL)[0]
    Fmask  = np.transpose(np.array([fmask1, fmask2, fmask3]))

    # spNodeToNode = np.concatenate((
    #     id.reshape(id.size, 1), 
    #     np.arange(0, N_FACES*K, 1, dtype=int).reshape(N_FACES*K, 1), 
    #     EToE.reshape(N_FACES*K, 1, order='F'), 
    #     EToF.reshape(N_FACES*K, 1, order='F')
    # ), 1)

    fxr = np.concatenate((xr[fmask1], xr[fmask2], xr[fmask3]))
    fxs = np.concatenate((xs[fmask1], xs[fmask2], xs[fmask3]))
    fyr = np.concatenate((yr[fmask1], yr[fmask2], yr[fmask3]))
    fys = np.concatenate((ys[fmask1], ys[fmask2], ys[fmask3]))

    Nfp = int(N+1)

    nx = np.zeros((3*Nfp, K))
    ny = np.zeros((3*Nfp, K))

    fid1 = np.linspace(0,        Nfp-1, Nfp, True, False, int).transpose()
    fid2 = np.linspace(Nfp,  2*Nfp-1, Nfp, True, False, int).transpose()
    fid3 = np.linspace(2*Nfp,3*Nfp-1, Nfp, True, False, int).transpose()

    # face 1
    nx[fid1] =  fyr[fid1]
    ny[fid1] = -fxr[fid1]

    # face 2
    nx[fid2] =  fys[fid2]-fyr[fid2] 
    ny[fid2] = -fxs[fid2]+fxr[fid2]

    # face 3
    nx[fid3] = -fys[fid3] 
    ny[fid3] =  fxs[fid3]

    # normalise
    sJ = np.sqrt(nx*nx+ny*ny)
    nx = nx/sJ 
    ny = ny/sJ
    return nx, ny, sJ