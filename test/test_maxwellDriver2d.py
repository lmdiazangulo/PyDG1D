import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_pec():
    sp = Maxwell2D(
        n_order = 3, 
        mesh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu'),
        fluxType="Upwind"
    )
    
    final_time = 1.0
    driver = MaxwellDriver(sp)
    
    x0 = 0.0
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver.E[:] = initialFieldE[:]
    
    driver.run_until(final_time)

    finalFieldE = driver.E
    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

def test_grad_2D():
        
    sp = Maxwell2D(
        n_order = 3, 
        mesh = mesh.readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2Triang.neu'),
        fluxType="Upwind"
    )
    N = 1
    x, y = nodes_coordinates(N,sp.mesh)
    r, s = xy_to_rs(*set_nodes(N))
    Dr, Ds = derivateMatrix(N, r, s, vandermonde(N, r, s))
    Ez = np.array([[1., 2.],[3., 4.], [5., 6.]])
    rx, sx, ry, sy, _ = geometricFactors(x, y, Dr, Ds)
    Ezx, Ezy = sp.grad2D(Dr, Ds, Ez, rx, sx, ry, sy)

    EzxExp = np.array([[ 4.,  2.],
                       [ 4.,  2.], 
                       [ 4.,  2.]])
    
    EzyExp = np.array([[-2., -4.],
                       [-2., -4.], 
                       [-2., -4.]])
    
    assert np.allclose(EzxExp, Ezx, rtol = 1e-3)
    assert np.allclose(EzyExp, Ezy, rtol = 1e-3)

    