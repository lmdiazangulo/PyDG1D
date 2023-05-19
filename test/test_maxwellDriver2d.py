import numpy as np
import matplotlib.pyplot as plt

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_pec():
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(2, msh, 'Centered')
    
    final_time = 4.0
    driver = MaxwellDriver(sp)
    
    s0 = 0.25
    initialFieldE = np.sin(np.pi*sp.x)*np.sin(np.pi*sp.y)
    # np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver['Ez'][:] = initialFieldE[:]
  
    fig = plt.figure().add_subplot(projection='3d')
    plt.triplot(sp.mesh.getTriangulation(), c='k', lw=1.0)
    # plt.set_aspect('equal')
    for _ in range(100):       
        sp.plot_field(2, driver['Ez'], fig)
        plt.pause(0.01)
        plt.cla()
        driver.step()


    