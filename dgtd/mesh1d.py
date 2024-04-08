import numpy as np

class Mesh1D:
    def __init__(self, xmin, xmax, k_elem, boundary_label="PEC"):
        
        assert k_elem > 0
        _, vx, _, EToV = mesh_generator(xmin, xmax, k_elem)

        self.dimension = 1
        self.vx = vx
        self.EToV = EToV
        self.boundary_label = boundary_label

        if self.boundary_label not in ["PEC", "PMC", "SMA", "Periodic", "PML", "Mur"]:
            raise ValueError("Invalid boundary label.")

    def number_of_vertices(self):
        return self.vx.shape[0]

    def number_of_elements(self):
        return self.vx.shape[0] - 1

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
    >>> etov_test = ([0,1],[[1, 2],[2, 3],[3, 4]])
    >>> np.allclose(etov,etov_test)
    True
    """

    n_v = k_elem+1
    vx = np.linspace(xmin, xmax, num=n_v)
    
    #np.zeros creates a float array. etov should be an integer array
    EToV = np.full((k_elem,2),0)
    #etov = np.zeros([K,2])
    for i in range(k_elem):
        EToV[i,0] = i
        EToV[i,1] = i+1

    return [n_v,vx,k_elem,EToV]
    