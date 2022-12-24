import numpy as np
import scipy.special
import math

def set_nodes_1d(N, vertices):
    """ 
    Sets N+1 nodes in equispaced positions using the vertices indicated
    by vx.
    """
    K = vertices.shape[1] # vertices columns 
    x = np.zeros((N+1, K))
    for k in range(K):
        x[:,k] = np.linspace(vertices[0,k], vertices[1,k], num=N+1)
             
    return x


def _node_indices_1d(N):
    """
    Generates number of node Indices for order N.
    
    >>> _node_indices_1d(1)
    array([[1, 0],
           [0, 1]])
           
    >>> _node_indices_1d(2)
    array([[2, 0],
           [1, 1],
           [0, 2]])
    """
    Np = N+1;
    nId = np.zeros([Np, 2])
    for i in range(Np):
        nId[i] = [N-i, i]     
    return nId.astype(int)


def mesh_generator(xmin,xmax,k_elem):
    """
    Generate simple equidistant grid with K elements
    >>> [Nv, vx, K, etov] = mesh_generator(0,10,4)
    >>> Nv
    5
    >>> vx_test = ([0.00000000,2.50000000,5.00000000,7.50000000,10.00000000])
    >>> np.allclose(vx,vx_test)
    True
    >>> K
    4
    >>> etov_test = ([[1, 2],[2, 3],[3, 4],[4, 5]])
    >>> np.allclose(etov,etov_test)
    True
    """

    n_v = k_elem+1
    vx = np.linspace(xmin, xmax, num=n_v)
    
    #np.zeros creates a float array. etov should be an integer array
    etov = np.full((k_elem,2),0)
    #etov = np.zeros([K,2])
    for i in range(k_elem):
        etov[i,0] = i+1
        etov[i,1] = i+2

    return [n_v,vx,k_elem,etov]
    