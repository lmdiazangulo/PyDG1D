import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytest
import time


from dgtd.maxwellDriver import *
from dgtd.mesh1d import *
from dgtd.maxwell1d import *
from fdtd.fdtd1d import *


from nodepy import runge_kutta_method as rk

#······················································

def plot(sp, driver):
    for _ in range(1000):
        driver.step()
        plt.plot(sp.x, driver['E'],'b')
        plt.plot(sp.xH, driver['H'],'r')
        plt.ylim(-1.5, 1.5)
        plt.title(driver.timeIntegrator.time)
        plt.grid(which='both')
        plt.pause(0.01)
        plt.cla()
        
#······················································


def test_tfsf_null_field():


    t_final = 8.0

    sp = FDTD1D(mesh=Mesh1D(-1.0, 2.0, 300, boundary_label="Mur"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    driver['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
    driver['H'][:] = np.exp(-(sp.xH - driver.dt/2)**2/(2*s0**2))

    plot(sp, driver)

    driver.run_until(t_final)



    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.0, atol=1e-3)