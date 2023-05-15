import numpy as np
import scipy.special
import math

import dgtd.dg1d as dg1d

n_faces = 3

alpopt = np.array([0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
        1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258])

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
    Pmat = np.zeros(N+1,Nr)
    for i in range(N):
        Pmat[i,:] = dg1d.jacobi_polynomial(rout, 0, 0, i)
    Lmat = Veq.transpose().dot(np.linalg.inv(Pmat))

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
    for n in range(N):
        for m in range(N+1-n):
            L1[sk] = n / N 
            L3[sk] = m / N
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
        alpha = alpopt(N)
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


    return x, y


# def node_indices(N):
#     """
#     Generates number of node Indices for order N.
#     """
#     Np = N+1
#     nId = np.zeros([Np, 2])
#     for i in range(Np):
#         nId[i] = [N-i, i]
#     return nId.astype(int)
