import numpy as np
import matplotlib.pyplot as plt

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_pec():
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(3, msh, 'Upwind')
    
    final_time = 2.0
    driver = MaxwellDriver(sp)
    
    s0 = 0.25
    initialFieldE = np.exp(-sp.x**2/(2*s0**2))
    
    driver['Ez'][:] = initialFieldE[:]
  
    fig = plt.figure()
    for _ in range(100):       
        driver.step()

        sp.plot_field(3, driver['Ez'], fig)
        plt.pause(0.01)
        plt.cla()


    finalFieldE = driver['Ez']
    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

    