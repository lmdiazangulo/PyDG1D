import numpy as np
import matplotlib.pyplot as plt
import pytest

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *
from fdtd.fdtd2d import *

TEST_DATA_FOLDER = 'testData/'

def resonant_cavity_ez_field(x, y, t):
    ''' Hesthaven's book p. 205 '''
    m = 1
    n = 1 
    w = np.pi * np.sqrt(m**2 + n**2)

    return np.sin(m*np.pi*x)*np.sin(n*np.pi*y)*np.cos(w*t)

def test_pec():
    N = 2
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(N, msh, 'Centered')
    
    driver = MaxwellDriver(sp, CFL=1)
    driver['Ez'][:] = resonant_cavity_ez_field(sp.x, sp.y, 0)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)  
    # ax.triplot(sp.mesh.getTriangulation(), c='k', lw=1.0)
    # plt.show()

    for _ in range(40):       
        # sp.plot_field(N, driver['Ez'])
        # plt.pause(0.001)
        # plt.cla()

        driver.step()

    ez_expected = resonant_cavity_ez_field(sp.x, sp.y, driver.timeIntegrator.time)
    R = np.corrcoef(ez_expected, driver['Ez'])
    assert R[0,1] > 0.9

@pytest.mark.skip(reason="Nothing is being tested.")
def test_fdtd2d_pec():
    sp = FDTD2D()
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    driver.run_until(2.0)

    finalFieldE = driver['E'][:]
    R = np.corrcoef(initialFieldE, finalFieldE)
    assert R[0, 1] > 0.9999