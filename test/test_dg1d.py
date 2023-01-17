from pytest import approx
import numpy as np

import dgtd.dg1d as dg
import dgtd.meshUtils as ms

    
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
    # assert np.allclose(np.array([[-4.74341649, 0.,  4.74341649]]),
    #                    dg.jacobi_polynomial_grad(dg.jacobiGL(0.0, 0.0, 2), 0.0, 0.0, 2))
    # assert np.allclose(np.array([[11.22497216,  0.        ,  0.        , 11.22497216]]),
    #                    dg.jacobi_polynomial_grad(dg.jacobiGL(0.0, 0.0, 3), 0.0, 0.0, 3))
    assert np.allclose(np.array([[-21.2132034,  0.        ,  0.        , 0.0,    21.2132034]]),
                       dg.jacobi_polynomial_grad(dg.jacobiGL(0.0, 0.0, 4), 0.0, 0.0, 4))
    
    
def test_vandermonde_grad_1d_order_1():
    assert np.allclose(np.array([[0.0, 1.22474487],
                                 [0.0, 1.22474487]]),
                       dg.vandermonde_grad(1, dg.jacobiGL(0.0, 0.0, 1)))
    
def test_vandermonde_grad_1d_order_2_3_4():    
    assert np.allclose(np.array([[0., 1.22474487, -4.74341649],
                                 [0., 1.22474487, 0.         ],
                                 [0., 1.22474487, 4.74341649]]),
                       dg.vandermonde_grad(2, dg.jacobiGL(0.0, 0.0, 2)))
    assert np.allclose(np.array([[0., 1.22474487, -4.74341649, 11.22497216],
                                 [0., 1.22474487, -2.1213,      0.],
                                 [0., 1.22474487, 2.1213,       0.],
                                 [0., 1.22474487, 4.74341649,  11.22497216]]),
                       dg.vandermonde_grad(3, dg.jacobiGL(0.0, 0.0, 3)))
    assert np.allclose(np.array([[0., 1.22474487, -4.74341649, 11.2249722, -21.2132034],
                                 [0., 1.22474487, -3.10529502, 3.20713490, 0.],
                                 [0., 1.22474487, 0. ,         -2.8062430, 0.        ],
                                 [0., 1.22474487, 3.10529502, 3.20713490,  0.],
                                 [0., 1.22474487, 4.74341649, 11.2249722, 21.2132034]]),
                       dg.vandermonde_grad(4, dg.jacobiGL(0.0, 0.0, 4)))
       
def test_differentation_matrix_order_1():    
    assert np.allclose(np.array([[-0.5, 0.5],
                                 [-0.5, 0.5]]),
                       dg.differentiation_matrix(1, dg.jacobiGL(0.0, 0.0, 1), 
                                                 dg.vandermonde_1d(1, dg.jacobiGL(0.0, 0.0, 1)
                                                                   )))   
    
def test_differentation_matrix_order_2_3_4():    
    assert np.allclose(np.array([[-1.5, 2.0, -0.5],
                                 [-0.5, 0.0, 0.5],
                                 [0.5, -2.0, 1.5]]),
                       dg.differentiation_matrix(2, dg.jacobiGL(0.0, 0.0, 2), 
                                                 dg.vandermonde_1d(2, dg.jacobiGL(0.0, 0.0, 2)
                                                                   ))) 
    assert np.allclose(np.array([[-3.0,           4.04508497, -1.54508497,   0.5    ],
                                 [-0.8090169904,  0.0,         1.11803399,  -0.3090169904],
                                 [0.3090169904,  -1.11803399,  0.,           0.8090169904],
                                 [-0.5,         1.54508497,    -4.04508497,  3.00]]),
                       dg.differentiation_matrix(3, dg.jacobiGL(0.0, 0.0, 3), 
                                                 dg.vandermonde_1d(3, dg.jacobiGL(0.0, 0.0, 3)
                                                                   ))) 
    assert np.allclose(np.array([[-5.0,         6.75650249,  -2.66666667,  1.41016418,  -0.500    ],
                                 [-1.24099025,  0.0,          1.74574312, -0.76376261,  0.259009747],
                                 [0.3750,       -1.33658458,   0.,         1.33658458,  -0.3750     ],
                                 [-0.259009747,  0.763762616, -1.74574312, 0.0,         1.24099025  ],
                                 [0.5,          -1.41016418,  2.6666667,   -6.75650249, 5.0       ]]),
                       dg.differentiation_matrix(4, dg.jacobiGL(0.0, 0.0, 4), 
                                                 dg.vandermonde_1d(4, dg.jacobiGL(0.0, 0.0, 4)
                                                                   ))) 
    
def test_surface_integral_dg_order_1():    
    assert np.allclose(np.array([[2., -1.0],
                                 [-1.0, 2.0]]),
                       dg.surface_integral_dg(1, dg.vandermonde_1d(1, dg.jacobiGL(0.0, 0.0, 1)
                                                                    ))) 

def test_surface_integral_dg_order_2_3_4():                                                                
    assert np.allclose(np.array([[4.5,      1.5],
                                 [-0.75, -0.75],
                                 [1.5, 4.5]]),
                       dg.surface_integral_dg(2, dg.vandermonde_1d(2, dg.jacobiGL(0.0, 0.0, 2)
                                                                   ))) 
    assert np.allclose(np.array([[8.00000000000000,  -2.00000000000000],
                                 [-0.894427190999917,  0.894427190999917],
                                 [0.894427190999917, -0.894427190999917],
                                 [-2.0,               8.0]]),
                       dg.surface_integral_dg(3, dg.vandermonde_1d(3, dg.jacobiGL(0.0, 0.0, 3)
                                                                   )))
    assert np.allclose(np.array([[12.5000000000000,	 2.50000000000000],
                                 [-1.07142857142857, -1.07142857142857],
                                 [0.937500000000001,  0.937500000000001],
                                 [-1.07142857142857, -1.07142857142857],
                                 [2.50000000000000,	 12.5000000000000]]),
                       dg.surface_integral_dg(4, dg.vandermonde_1d(4, dg.jacobiGL(0.0, 0.0, 4)
                                                                   )))
def test_normals_2_3_element():    
    assert np.allclose(np.array([[-1., -1.0],
                                 [1., 1.0]]),
                       dg.normals(2)) 
    assert np.allclose(np.array([[-1., -1.0, -1.0],
                                 [1., 1.0, 1.0]]),
                       dg.normals(3)) 
    
def test_filter():    
    assert np.allclose(np.array([[0.3333842 ,  0.9756328 , -0.14240119, -0.1666158 ],
                                 [ 0.19512656,  0.66667684,  0.16667684, -0.02848024],
                                 [-0.02848024,  0.16667684,  0.66667684,  0.19512656],
                                 [-0.1666158 , -0.14240119,  0.9756328 ,  0.3333842 ]]),
                       dg.filter(3,1,1,dg.vandermonde_1d(3, dg.jacobiGL(0.0, 0.0 , 3)))) 
    
def test_nodes_coordinates():
    [Nv,vx,K,etov] = ms.mesh_generator(0,10,4)
    assert np.allclose(np.array([[0.00000000,    2.50000000,    5.00000000,    7.50000000], 
                                 [0.43168291,    2.93168291,    5.43168291,    7.93168291], 
                                 [1.25000000,    3.75000000,    6.25000000,    8.75000000], 
                                 [2.06831709,    4.56831709,    7.06831709,    9.56831709], 
                                 [2.50000000,    5.00000000,    7.50000000,   10.00000000]]),
                       dg.nodes_coordinates(4,etov,vx )) 
    
def test_geometric_factors():
    [Nv,vx,K,etov] = ms.mesh_generator(0,10,4)
    x           = dg.nodes_coordinates(2,etov,vx)
    r           = dg.jacobiGL(0.0, 0.0, 2)
    V           = dg.vandermonde_1d(2, r)
    Dr          = dg.differentiation_matrix(2,r,V)
    [rx, J]     = dg.geometric_factors(x, Dr)
    assert np.allclose(np.array([[0.80000,   0.80000,   0.80000,   0.80000], \
                                 [0.80000,   0.80000,   0.80000,   0.80000], \
                                 [0.80000,   0.80000,   0.80000,   0.80000]]),
                       rx)
    assert np.allclose(np.array([[1.2500,   1.2500,   1.2500,   1.2500], 
                     [1.2500,   1.2500,   1.2500,   1.2500], 
                     [1.2500,   1.2500,   1.2500,   1.2500]]),
                       J)  
    
def test_connect_4_nodes():
    [Nv,vx,K,etov] = ms.mesh_generator(0,10,4)
    [etoe, etof] = dg.connect(etov)
    etoe_test =  np.array([[1,2], [1,3], [2,4], 
                           [3,4]])
    etof_test = np.array([[1,1], [2,1], [2,1],
                          [2,2]])
    assert np.allclose(etoe, etoe_test)
    assert np.allclose(etof,etof_test)
    
def test_connect_7_nodes():
    [Nv,vx,K,etov] = ms.mesh_generator(-1,22,7)
    [etoe, etof] = dg.connect(etov)
    etoe_test =  np.array([[1,2], [1,3], [2,4], 
                           [3,5], [4,6], [5,7],
                           [6,7]])
    etof_test = np.array([[1,1], [2,1], [2,1],
                          [2,1], [2,1], [2,1],
                          [2,2]])
    assert np.allclose(etoe, etoe_test)
    assert np.allclose(etof,etof_test)
    
def test_build_maps():
    [Nv,vx,K,etov] = ms.mesh_generator(0,10,4)
    x = dg.nodes_coordinates(4,etov,vx)
    [etoe, etof] = dg.connect(etov)
    [vmap_m,vmap_p,vmap_b,map_b] = dg.build_maps(4,x,etoe,etof)
    vmap_m_test = np.array([[0,4,5,9,10,14,15,19]])
    vmap_p_test = np.array([[0,5,4,10,9,15,14,19]])
    vmap_b_test = np.array([[0, 19]])
    map_b_test = np.array([[0,7]])
    assert np.allclose(vmap_m,vmap_m_test)
    assert np.allclose(vmap_p,vmap_p_test)
    assert np.allclose(vmap_b,vmap_b_test)
    assert np.allclose(map_b,map_b_test)

def test_set_nodes_1d():
    vx = np.array([0.0, 1.0, 2.0])
    etov = np.array([[0, 1],
                     [1, 2]])
    N = 4
    x = dg.set_nodes_1d(N, vx[etov])
    assert np.allclose(
        np.transpose(
            np.array([[0.00, 0.25, 0.50, 0.75, 1.00],
                      [1.00, 1.25, 1.50, 1.75, 2.00]])), 
            x)
  
def test_node_indices_1d_N_1_2():
    assert np.allclose(np.array([[1, 0], [0,1]]), 
                       dg.node_indices_1d(1))
    assert np.allclose(np.array([[2, 0], [1,1], 
                                 [0,2]]), 
                       dg.node_indices_1d(2))
    
