from pytest import approx
import numpy as np

import dgtd.mesh1d as ms

def test_mesh_generator_n_1():
    [n_v, vx, k_elem, etov] = ms.mesh_generator(0.0, 1.0, 1)
    assert np.allclose(np.array([0.0, 1.0]), vx)
    assert np.allclose(np.array([0,1]), etov)
    
def test_mesh_generator_n_2():
    [n_v, vx, k_elem, etov] = ms.mesh_generator(0.0, 1.0, 2)
    assert np.allclose(np.array([0.0, 0.5, 1.0]), vx)
    assert np.allclose(np.array([[0,1],[1,2]]), etov)
    
def test_mesh_generator_n_4():
    [n_v, vx, k_elem, etov] = ms.mesh_generator(0.0, 1.0, 4)
    assert np.allclose(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), vx)
    assert np.allclose(np.array([[0,1],[1,2],[2,3],[3,4]]), etov)
    
def test_mesh_generator_n_4_():
    [n_v, vx, k_elem, etov] = ms.mesh_generator(-1.0, 9.0, 4)
    assert np.allclose(np.array([-1.0, 1.5, 4.0, 6.5, 9.0]), vx)
    assert np.allclose(np.array([[0,1],[1,2],[2,3],[3,4]]), etov)

