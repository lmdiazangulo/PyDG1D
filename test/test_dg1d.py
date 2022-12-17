from pytest import approx
import numpy as np

import dgtd.dg1d as dg

def test_jacobiGL():
    assert np.all(np.array([-1.,  1.]) == dg.jacobiGL(0.0, 0.0, 1))