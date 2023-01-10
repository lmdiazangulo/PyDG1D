from pytest import approx
import numpy as np

import dgtd.dg1d as dg
import dgtd.meshUtils as ms

def test_passing():
    assert 42 == 42
def test_failure():
    assert 42 == -13
    
def test_jacobiGL():
    assert np.all(np.array([-1.,  1.]) == dg.jacobiGL(0.0, 0.0, 1))
    assert np.all(np.array([-1.,  0,  1.]) == dg.jacobiGL(0.0, 0.0, 2))
    assert np.allclose(
        np.array([-1., -0.4472136, 0.4472136, 1.0]), dg.jacobiGL(0.0, 0.0, 3))


def test_jacobi_gauss():
    assert np.allclose(np.array([-0.2, 2]),
                       dg.jacobi_gauss(2, 1, 0))
    assert np.allclose(np.array([[-0.54691816,  0.26120387], [0.76094757, 0.57238576]]),
                       dg.jacobi_gauss(2, 1, 1))
    assert np.allclose(np.array([[-0.70882014, -0.13230082,  0.50778763],
                                [0.39524241,  0.72312171,  0.21496922]]),
                       dg.jacobi_gauss(2.0, 1.0, 2))


def test_jacobi_polynomial_order_0():
    assert np.allclose(np.array([0.70710678, 0.70710678, 0.70710678, 0.70710678]),
                       dg.jacobi_polynomial(dg.jacobiGL(0.0, 0.0, 0), 0., 0., 0))


def test_jacobi_polynomial_order_1():
    assert np.allclose(np.array([-1.22474487, 1.22474487]),
                       dg.jacobi_polynomial(dg.jacobiGL(0.0, 0.0, 1), 0., 0., 1))


def test_jacobi_polynomial_order_2_3_4():
    assert np.allclose(np.array([1.58113883, -0.79056942, 1.58113883]),
                       dg.jacobi_polynomial(dg.jacobiGL(0.0, 0.0, 2), 0., 0., 2))
    assert np.allclose(np.array([-1.87082869, 0.83666003, -0.83666003,  1.87082869]),
                       dg.jacobi_polynomial(dg.jacobiGL(0.0, 0.0, 3), 0., 0., 3))
    assert np.allclose(np.array([2.12132034, -0.90913729,  0.79549513, -0.90913729,  2.12132034]),
                       dg.jacobi_polynomial(dg.jacobiGL(0.0, 0.0, 4), 0., 0., 4))


def test_vandermonde_1d_order_1():
    assert np.allclose(np.array([[0.70710678, -1.22474487],
                                 [0.70710678, 1.22474487]]),
                       dg.vandermonde_1d(1, dg.jacobiGL(0.0, 0.0, 1)))
    
def test_vandermonde_1d_order_2_3_4():    
    assert np.allclose(np.array([[0.70710678, -1.22474487, 1.58113883],
                                 [0.70710678, 0.00000,    -0.79056942],
                                 [0.70710678, 1.22474487, 1.58113883]]),
                       dg.vandermonde_1d(2, dg.jacobiGL(0.0, 0.0, 2)))
    assert np.allclose(np.array([[0.70710678, -1.22474487, 1.58113883, -1.87082869],
                                 [0.70710678, -0.54772256, -0.31622777, 0.83666003],
                                 [0.70710678, 0.54772256, -0.31622777, -0.83666003],
                                 [0.70710678, 1.22474487, 1.58113883,  1.87082869]]),
                       dg.vandermonde_1d(3, dg.jacobiGL(0.0, 0.0, 3)))
    assert np.allclose(np.array([[0.70710678, -1.22474487, 1.58113883, -1.87082869, 2.12132034],
                                 [0.70710678, -0.80178373, 0.22587698, 0.52489066, -0.90913729],
                                 [0.70710678, -0.000000, -0.79056942, 0.000000, 0.79549513],
                                 [0.70710678, 0.80178373, 0.22587698, -0.52489066, -0.90913729],
                                 [0.70710678, 1.22474487, 1.58113883,  1.87082869, 2.12132034]]),
                       dg.vandermonde_1d(4, dg.jacobiGL(0.0, 0.0, 4)))

def test_jacobi_polynomial_grad_order_1():
    assert np.allclose(np.array([[1.22474487, 1.22474487]]),
                       dg.jacobi_polynomial_grad(dg.jacobiGL(0.0, 0.0, 1), 0.0, 0.0, 1))
 
def test_jacobi_polynomial_grad_order_2_3_4():
    assert np.allclose(np.array([[-4.74341649, 0.,  4.74341649]]),
                       dg.jacobi_polynomial_grad(dg.jacobiGL(0.0, 0.0, 2), 0.0, 0.0, 2))
    assert np.allclose(np.array([[11.22497216,  0.        ,  0.        , 11.22497216]]),
                       dg.jacobi_polynomial_grad(dg.jacobiGL(0.0, 0.0, 3), 0.0, 0.0, 3))
    assert np.allclose(np.array([[-21.2132497216,  0.        ,  0.        , 0.0,    21.2132497216]]),
                       dg.jacobi_polynomial_grad(dg.jacobiGL(0.0, 0.0, 4), 0.0, 0.0, 4))
    
    
def test_vandermonde_grad_1d_order_1():
    assert np.allclose(np.array([[0.0, 1.22474487],
                                 [0.0, 1.22474487]]),
                       dg.vandermonde_grad(1, dg.jacobiGL(0.0, 0.0, 1)))
    
def test_vandermonde_grad_1d_order_2_3_4():    
    assert np.allclose(np.array([[0., 1.22474487, -4.74341649],
                                 [0., 1.22474487, 0.         ],
                                 [0., 1.22474487, -4.74341649]]),
                       dg.vandermonde_grad(2, dg.jacobiGL(0.0, 0.0, 2)))
    assert np.allclose(np.array([[0., 1.22474487, -4.74341649, 11.22497216],
                                 [0., 1.22474487, -2.1213,      0.],
                                 [0., 1.22474487, 2.1213,       0.,
                                 [0., 1.22474487, 4.74341649,  11.22497216]]),
                       dg.vandermonde_grad(3, dg.jacobiGL(0.0, 0.0, 3)))
    # assert np.allclose(np.array([[0.70710678, -1.22474487, 1.58113883, -1.87082869, 2.12132034],
    #                              [0.70710678, -0.80178373, 0.22587698, 0.52489066, -0.90913729],
    #                              [0.70710678, -0.000000, -0.79056942, 0.000000, 0.79549513],
    #                              [0.70710678, 0.80178373, 0.22587698, -0.52489066, -0.90913729],
    #                              [0.70710678, 1.22474487, 1.58113883,  1.87082869, 2.12132034]]),
    #                    dg.vandermonde_grad(4, dg.jacobiGL(0.0, 0.0, 4)))
       
   