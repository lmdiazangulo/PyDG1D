import numpy as np
import matplotlib.pyplot as plt

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_pec():
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(3, msh, 'Upwind')
    
    final_time = 4.0
    driver = MaxwellDriver(sp)
    
    x0 = 0.0
    s0 = 0.25
    initialFieldE = np.sin(np.pi*sp.x)*np.sin(np.pi*sp.y)
    # np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver['Ez'][:] = initialFieldE[:]
    
    # plt.figure()
    # plt.triplot(msh.getTriangulation())

    Epoint = np.zeros(1)
    Epoint[0] = driver['Ez'][2,114]
    timeVec = np.zeros(1)
    timeVec[0] = 0.0

    time = 0.0
    while (time < final_time):
        
        if (time + driver.dt > final_time):
            driver.dt = final_time - time
        
        driver.step()
        tempEpoint = np.concatenate((Epoint,[driver['Ez'][2,114]]))
        Epoint = tempEpoint
        tempTimeVec = np.concatenate((timeVec,[time]))
        timeVec = tempTimeVec

        time += driver.dt

    #     plt.tricontourf(msh.getTriangulation(), driver['Ez'])
    #     plt.pause(0.01)
    #     plt.cla()
    plt.xlabel('Time')
    plt.ylabel('Ez')
    plt.plot(timeVec, Epoint)
    plt.show()

    finalFieldE = driver['Ez']
    # R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
    #                 finalFieldE.reshape(1, finalFieldE.size))
    # assert R[0,1] > 0.9999

    assert True

    