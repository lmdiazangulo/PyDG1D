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
        for i in range(N+1):
            x[i,k] = i * (vertices[1,k] - vertices[0,k]) / N + vertices[0,k];
             
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