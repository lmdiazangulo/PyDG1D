'''
Created on Nov 23, 2017

@author: luis
'''

import numpy as np

if __name__ == '__main__':
    pass

def meshGrid(D, N, vertices):
    """ 
    Sets N+1 nodes in equispaced positions using the vertices indicated by vx.
    """
    if D != 1:
        raise ValueError("Not implemented")
    
    K = vertices.shape[1]
    x = np.zeros( (N+1, K) )
    for k in range(K):
        for i in range(N+1):
            x[i,k] = i * (vertices[1,k] -vertices[0,k]) / N + vertices[0,k];
            
    return x

def nodeIndices(D, N):
    """
    Generates number of node Indices for order N and dimension D
    """
    if D != 1:
        raise ValueError("Not implemented")
    
    Np = N+1;
    nId = np.array(Np, D+1)
    for i in range(N):
        nId[[i,0]] = N-i
        nId[[i,1]] = i
    
    return nId
    

def silvesterPolynomial(D, N):
    

def lagrangePolynomial(D, N):


def massMatrix(D, N):
    """
    Creates mass matrix analitically for given dimension and order.
    """
    if dimension != 1:
        raise ValueError("Not implemented")
    
    N = order;