import numpy as np
import matplotlib.pyplot as plt

from dgtd.maxwell1d import *
from dgtd.meshUtils import *

import dgtd.meshUtils as ms
import dgtd.maxwell1d as mw

def test_spatial_discretization_lift():
    sp = SpatialDiscretization(1, Mesh1D(0.0, 1.0, 1))
    assert   np.allclose(surface_integral_dg(1, vandermonde_1d(1, jacobiGL(0.0,0.0,1))), 
                         np.array([[2.0,-1.0],[-1.0,2.0]]))


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
    E_old = 0 #math.sin(np.pi*x)
    H_old = 0 #np.zeros((sp.number_of_nodes_per_element(), mesh.number_of_elements()))
    
    # Solve problem
    final_time = 10
    [E, H] = mw.maxwell1D(E_old, H_old, epsilon, mu, final_time, sp)
 #   assert np.allclose(E, )
    
# def test_maxwell1d_1():
#     assert np.allclose(maxwell1d(E, H, eps, mu, final_time, sp: SpatialDiscretization), )