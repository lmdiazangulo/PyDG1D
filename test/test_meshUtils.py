from pytest import approx
import numpy as np

from dgtd.meshUtils import *

def test_set_nodes_1d():
    vx = np.array([0.0, 1.0, 2.0])
    etov = np.array([[0, 1],
                     [1, 2]])
    N = 4
    x = set_nodes_1d(N, vx[etov])
    assert np.allclose(
        np.transpose(
            np.array([[0.00, 0.25, 0.50, 0.75, 1.00],
                      [1.00, 1.25, 1.50, 1.75, 2.00]])), 
            x)
    
def test_mesh_generator():
    [n_v, vx, k_elem, etov] = mesh_generator(0.0, 1.0, 2)
    assert np.allclose(np.array([0.0, 0.5, 1.0]), vx)
