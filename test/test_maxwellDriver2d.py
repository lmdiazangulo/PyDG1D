import numpy as np
import matplotlib.pyplot as plt

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_pec():
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(1, msh, 'Upwind')
    
    final_time = 1.0
    driver = MaxwellDriver(sp)
    
    x0 = 0.0
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver['Ez'][:] = initialFieldE[:]
    
    # plt.figure()
    # plt.triplot(msh.getTriangulation())

    for _ in range(10):
        driver.step()
    #     plt.tricontourf(msh.getTriangulation(), driver['Ez'])
    #     plt.pause(0.01)
    #     plt.cla()

    finalFieldE = driver['Ez']
    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

    