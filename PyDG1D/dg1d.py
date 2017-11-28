'''
Created on Nov 23, 2017

@author: luis
'''

import numpy as np

if __name__ == '__main__':
    pass

def setNodes(order, vertices):
    """ 
    Sets N+1 nodes in equispaced positions using the vertices indicated by vx.
    """
    N = order;
    K = vertices.shape[1]
    
    x = np.zeros( (N+1, K) )
    for k in range(K):
        for i in range(N+1):
            x[i,k] = i * (vertices[1,k] -vertices[0,k]) / N + vertices[0,k];
            
    return x