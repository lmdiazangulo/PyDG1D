from pytest import approx
import numpy as np

import dgtd.dg1d as dg
import dgtd.mesh_utils as ms

def test_jacobiGL():
    assert np.all(np.array([-1. ,  1.])                               == dg.jacobiGL(0.0, 0.0, 1))
    assert np.all(np.array([-1. ,  0 ,  1.])                          == dg.jacobiGL(0.0, 0.0, 2))
    assert np.allclose(np.array([-1. , -0.4472136 , 0.4472136 , 1.0]) , dg.jacobiGL(0.0, 0.0, 3))
   
    
def test_jacobi_gauss():
    assert np.allclose(np.array([-0.2 , 2]) ,
                       dg.jacobi_gauss(2, 1, 0))
    assert np.allclose(np.array([[-0.54691816,  0.26120387], [0.76094757, 0.57238576]]) , 
                       dg.jacobi_gauss(2, 1, 1))
    assert np.allclose(np.array([[-0.70882014, -0.13230082,  0.50778763] , 
                                [0.39524241,  0.72312171,  0.21496922]]) , 
                                dg.jacobi_gauss(2.0, 1.0, 2))
    
def test_jacobi_polynomial():
    assert np.allclose(np.array([-1.87082869, 0.83666003, -0.83666003,  1.87082869]) ,
                       dg.jacobi_polynomial(dg.jacobiGL(0.0, 0.0, 3.),0. , 0. , 3.)
    # assert np.allclose(np.array([[-0.54691816,  0.26120387], [0.76094757, 0.57238576]]) , 
    #                    dg.jacobi_polynomial(2, 1, 1))
    # assert np.allclose(np.array([[-0.70882014, -0.13230082,  0.50778763] , 
    #                             [0.39524241,  0.72312171,  0.21496922]]) , 
    #                             dg.jacobi_polynomial(2.0, 1.0, 2))