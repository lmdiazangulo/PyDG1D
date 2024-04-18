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

def gaussian(s):
    return lambda x,t : np.exp(-(x-t)**2/(2*s**2))

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

#@pytest.mark.skip(reason="WIP")
def test_tfsf_null_field():

#La onda se genera en x=0 y nuestro limite de TF está en x=0.8

    t_final = 8.0
    s0 = 0.1

    sp = FDTD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="Mur"))
    TFSF_setup = {}
    TFSF_setup["left"] = -0.8
    TFSF_setup["right"] =  0.8
    TFSF_setup["source"] = gaussian(s0)
    sp.TFSF_conditions(TFSF_setup)

    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    # driver['E'][10] = 1
    # driver['H'][10] = 1
    driver['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
    driver['H'][:] = np.exp(-(sp.xH - 0.5*driver.dt)**2/(2*s0**2))

    plot(sp, driver)

    driver.run_until(t_final)

    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.0, atol=1e-3)