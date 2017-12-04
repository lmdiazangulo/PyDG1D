'''
Created on Nov 23, 2017

@author: luis
'''

import numpy as np
import scipy.special

if __name__ == '__main__':
    pass


def setNodes1D(N, vertices):
    """ 
    Sets N+1 nodes in equispaced positions using the vertices indicated
    by vx.
    """
    
    K = vertices.shape[1]
    x = np.zeros((N+1, K))
    for k in range(K):
        for i in range(N+1):
            x[i,k] = i * (vertices[1,k] - vertices[0,k]) / N + vertices[0,k];
            
    return x


def nodeIndices(N):
    """
    Generates number of node Indices for order N and dimension D
    """
    Np = N+1;
    nId = np.array(Np, 2)
    for i in range(N):
        nId[i] = [N-i, i]     
    return nId
    
    
def jacobiGL(alpha, beta, N):
    """
    Compute the order N Gauss Lobatto quadrature points, x, associated
    with the Jacobi polynomial.
    """     
    if N==0:
        return np.array([0.0])
    if N==1:
        return np.array([-1.0, 1.0])
    
    x = scipy.special.roots_jacobi(N-2, alpha+1, beta+1);
    return x
    