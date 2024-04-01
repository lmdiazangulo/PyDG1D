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

def test_fdtd2d_te_pec():
    sp = FDTD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_label="PEC")
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-xH**2/(2*s0**2)) #no seria test PMC esto??
    driver['H'][:,:] = initialFieldH[:,:]

    driver.run_until(final_time)

    finalFieldH = driver['H']
    R = np.corrcoef(initialFieldH.ravel(), finalFieldH.ravel())
    assert R[0, 1] > 0.9999

    # while driver.timeIntegrator.time < final_time:
    #     # plt.contourf(xH, yH, driver['H'], vmin=-1.0, vmax=1.0)
    #     # # plt.plot(sp.xH, driver['H'][4,:])
    #     # plt.ylim(-1, 1)
    #     # plt.grid(which='both')
    #     # plt.pause(0.01)
    #     # plt.cla()
    #     driver.step()

# def test_fdtd2d_te_pmc():
#     sp = FDTD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_label="PMC")
#     driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
#     s0 = 0.25
#     final_time = 2.0

#     xH, yH = np.meshgrid(sp.xH, sp.yH)
#     initialFieldH = np.exp(-xH**2/(2*s0**2))
#     driver['H'][:,:] = initialFieldH[:,:]
    
    #Problema TypeError: unhashable type: 'slice'

#     driver.run_until(final_time)

#     finalFieldH = driver['H']
#     R = np.corrcoef(initialFieldH.ravel(), finalFieldH.ravel())
#     assert R[0, 1] > 0.9999

#     # while driver.timeIntegrator.time < final_time:
#     #     # plt.contourf(xH, yH, driver['H'], vmin=-1.0, vmax=1.0)
#     #     # # plt.plot(sp.xH, driver['H'][4,:])
#     #     # plt.ylim(-1, 1)
#     #     # plt.grid(which='both')
#     #     # plt.pause(0.01)
#     #     # plt.cla()
#     #     driver.step()