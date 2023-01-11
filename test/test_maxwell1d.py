import numpy as np
import matplotlib.pyplot as plt

from dgtd.maxwell1d import *
from dgtd.meshUtils import *

def test_spatial_discretization_lift():
    sp = SpatialDiscretization(1, Mesh1D(0.0, 1.0, 1))
    assert  self.lift == surface_integral_dg(n_order, vander)


def test_empty_mesh():
    # Polynomial order for aproximation
    n_order = 3
    
    mesh = Mesh1D(-2.0, 2.0, 5)
    sp = SpatialDiscretization(n_order, mesh)
    
    # Set up material parameters
    epsilon = np.ones(mesh.number_of_elements())
    mu      = np.ones(mesh.number_of_elements())
    
    x       = set_nodes_1d(n_order, mesh.vx[mesh.EToV])
    
    # Set initial condition
    E_old = np.multiply(math.sin(np.pi*x) , x)
    H_old = np.zeros((sp.number_of_nodes_per_element(), mesh.number_of_elements))
    
    # Solve problem
    final_time = 10
    [E, H] = maxwell1D(E_old, H_old, epsilon, mu, final_time, sp)