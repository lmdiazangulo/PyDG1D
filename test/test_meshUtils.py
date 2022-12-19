from pytest import approx
import numpy as np

import dgtd.meshUtils as ms

def test_SetNodes1D():
    assert np.allclose(np.array([[0.,0.],[0.33333,0.],[0.66666,0.],[1.,0.]]), 
                        ms.set_nodes_1d(3,np.array([[0,0],[1,0]])))