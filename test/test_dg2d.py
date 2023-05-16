from pytest import approx
import numpy as np

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
    
def test_xy_to_rs():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([1.0, 1.5, 2.0])
    assert np.allclose(
        np.array(
        [[-((np.sqrt(3.0)+1.0)/3.0),(1.0-3.0*np.sqrt(3.0))/6.0,(-2.0*np.sqrt(3.0)+2.0)/3.0],
        [(4.0*np.sqrt(3.0)-2.0)/6.0,(6.0*np.sqrt(3.0)-2.0)/6.0,(8.0*np.sqrt(3.0)-2.0)/6.0]]
        ),
        dg.xytors(x, y)
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
    
