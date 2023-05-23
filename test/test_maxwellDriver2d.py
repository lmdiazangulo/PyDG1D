import numpy as np
import matplotlib.pyplot as plt

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def resonant_cavity_ez_field(x, y, t):
    ''' Hesthaven's book p. 205 '''
    m = 1
    n = 1 
    w = np.pi * np.sqrt(m**2 + n**2)

    return np.sin(m*np.pi*x)*np.sin(n*np.pi*y)*np.cos(w*t)

def test_pec():
    N = 8
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(N, msh, 'Upwind')
    
    driver = MaxwellDriver(sp)
    driver['Ez'][:] = resonant_cavity_ez_field(sp.x, sp.y, 0)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)  
    for _ in range(50):       
        # ax.triplot(sp.mesh.getTriangulation(), c='k', lw=1.0)
        # sp.plot_field(N, driver['Ez'])
        # plt.pause(0.001)
        # plt.cla()

        driver.step()

    ez_expected = resonant_cavity_ez_field(sp.x, sp.y, driver.timeIntegrator.time)
    R = np.corrcoef(ez_expected, driver['Ez'])
    assert R[0,1] > 0.999

    