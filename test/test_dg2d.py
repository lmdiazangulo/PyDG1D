from pytest import approx
import numpy as np
import math
import dgtd.dg2d as dg
    
def test_set_nodes_N1():
    assert np.allclose(
        np.array(
            [[-1.0, -1/np.sqrt(3.0)],
             [ 1.0, -1/np.sqrt(3.0)],
             [ 0.0,  2/np.sqrt(3.0)]]
        ),
        dg.set_nodes(1)
    )

def test_set_nodes_N2():
    assert np.allclose(
        np.array(
            [[-1.0, -1/np.sqrt(3.0)],
             [ 0.0, -1/np.sqrt(3.0)],
             [ 1.0, -1/np.sqrt(3.0)],
             [-0.5,  1/2/np.sqrt(3.0)],
             [ 0.5,  1/2/np.sqrt(3.0)],
             [ 0.0,  2/np.sqrt(3.0)]]
        ),
        dg.set_nodes(2)
    )

# def test_node_indices_N_1_2():
#     assert np.allclose(
#         np.array(
#             [[1, 0, 0],
#              [0, 1, 0],
#              [0, 0, 1]]
#         ), 
#         dg.node_indices(1)
#     )
    
#     assert np.allclose(
#         np.array(
#          [[2, 0, 0],
#           [1, 1, 0],
#           [0, 2, 0],
#           [1, 0, 1],
#           [0, 0, 2]]), 
#         dg.node_indices_1d(2)
#     )

def test_simplex_polynomial():
    a, b = (
        np.array([-1,  0,  1, -1,  1, -1 ]),
        np.array([-1, -1, -1,  0,  0,  1 ])
    )
    p11 = dg.simplex_polynomial(a, b, 1, 1)
    p11Ref = np.array(
        [2.1213, 0.0000, -2.1213, -1.5910, 1.5910, 0.0000]
    )

    assert np.allclose(p11, p11Ref, rtol=1e-3)


def test_rs_to_ab():
    r, s = ( 
        np.array([-1,  0,  1, -1, 0, -1 ]),
        np.array([-1, -1, -1,  0, 0,  1 ])
    )

    a, b = dg.rs_to_ab(r, s)

    aRef, bRef = (
        np.array([-1,  0,  1, -1,  1, -1 ]),
        np.array([-1, -1, -1,  0,  0,  1 ])
    )

    assert(np.all(a == aRef))
    assert(np.all(b == bRef))

def test_vanderdmonde_N2():
    # For N = 2.
    r, s = ( 
        np.array([-1,  0,  1, -1, 0, -1 ]),
        np.array([-1, -1, -1,  0, 0,  1 ])
    )

    V = dg.vandermonde(2, r, s)

    VRef = np.array(
        [[0.7071, -1.0000,  1.2247, -1.7321,  2.1213,  2.7386],
         [0.7071, -1.0000,  1.2247,  0.0000,  0.0000, -1.3693],
         [0.7071, -1.0000,  1.2247,  1.7321, -2.1213,  2.7386],
         [0.7071,  0.5000, -0.8660, -0.8660, -1.5910,  0.6847],
         [0.7071,  0.5000,  0.8660,  0.8660,  1.5910,  0.6847],
         [0.7071,  2.0000,  0.0000,  0.0000,  0.0000,  0.0000]]
    )

    assert np.allclose(V, VRef, rtol=1e-3)