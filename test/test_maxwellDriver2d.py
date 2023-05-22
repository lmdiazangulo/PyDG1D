import numpy as np
import matplotlib.pyplot as plt

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_pec():
    N = 5
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(N, msh, 'Upwind')
    
    final_time = 4.0
    driver = MaxwellDriver(sp)
    
    s0 = 0.25
    initialFieldE = np.sin(np.pi*sp.x)*np.sin(np.pi*sp.y)
    # np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver['Ez'][:] = initialFieldE[:]
  
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for _ in range(100):       
        ax.triplot(sp.mesh.getTriangulation(), c='k', lw=1.0)
        sp.plot_field(N, initialFieldE[:], fig)
        plt.pause(0.1)
        plt.cla()
        driver.step()

    assert True

    