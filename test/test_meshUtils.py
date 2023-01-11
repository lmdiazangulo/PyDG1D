from pytest import approx
import numpy as np

import dgtd.meshUtils as ms

def test_mesh_generator_n_1():
    [n_v, vx, k_elem, etov] = ms.mesh_generator(0.0, 1.0, 1)
    assert np.allclose(np.array([0.0, 1.0]), vx)
    
def test_mesh_generator_n_2():
    [n_v, vx, k_elem, etov] = ms.mesh_generator(0.0, 1.0, 2)
    assert np.allclose(np.array([0.0, 0.5, 1.0]), vx)
    assert np.allclose(np.array([[0,1],[1,2]]), etov)
    

