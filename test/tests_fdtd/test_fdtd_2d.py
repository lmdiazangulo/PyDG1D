import numpy as np
import matplotlib.pyplot as plt

from maxwell.dg.mesh2d import *
from maxwell.dg.dg2d import *
from maxwell.driver import *
from maxwell.fd.fd2d import *

import pytest

#······················································

def plot(sp, driver, final_time , xH, yH):
    while driver.timeIntegrator.time < final_time:
        plt.contourf(xH, yH, driver['H'], vmin=-1.0, vmax=1.0)
        plt.plot(sp.x, driver['E']['y'][4,:], 'b')
        plt.plot(sp.xH, driver['H'][4,:], 'r')
        plt.ylim(-1, 1)
        plt.grid(which='both')
        plt.pause(0.01)
        plt.cla()
        driver.step()
#······················································


def test_fdtd2d_te_pec_x():
    sp = FD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels="PEC")
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-xH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    #plot(sp, driver, final_time , xH, yH)

    driver.run_until(final_time)

    finalFieldH = driver['H']
    R = np.corrcoef(initialFieldH.ravel(), finalFieldH.ravel())
    assert R[0, 1] > 0.9999

def test_fdtd2d_te_pec_y():
    sp = FD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels="PEC")
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-yH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    #plot(sp, driver, final_time , xH, yH)

    driver.run_until(final_time)

    finalFieldH = driver['H']
    R = np.corrcoef(initialFieldH.ravel(), finalFieldH.ravel())
    assert R[0, 1] > 0.9999

def test_fdtd2d_te_pmc_x():
    bdrs = {
        "XL": "PMC",
        "XU": "PMC",
        "YL": "PEC",
        "YU": "PEC"
    }
    sp = FD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels=bdrs)
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.00)

    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-xH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    #plot(sp, driver, final_time , xH, yH)

    driver.run_until(final_time)
    

    finalFieldH = driver['H']
    R = np.corrcoef(initialFieldH.ravel(), -finalFieldH.ravel())
    assert R[0, 1] > 0.9999


def test_fdtd2d_te_pmc_y():
    bdrs = {
        "XL": "PEC",
        "XU": "PEC",
        "YL": "PMC",
        "YU": "PMC"
    }
    sp = FD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels=bdrs)
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-yH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    #plot(sp, driver, final_time , xH, yH)

    driver.run_until(final_time)

    finalFieldH = driver['H']
    R = np.corrcoef(initialFieldH.ravel(), -finalFieldH.ravel())
    assert R[0, 1] > 0.9999



def test_fd2d_check_initial_conditions_GW_right():

    bdrs = {
        "XL": "Mur",
        "XU": "Mur",
        "YL": "PEC",
        "YU": "PEC"
    }

    sp = FD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels=bdrs)
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-xH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    # x, y = np.meshgrid(sp.x, sp.y)
    # initialFieldE = np.exp(-(y - driver.dt/2)**2/(2*s0**2))
    # driver['E']['y'][:,:] = initialFieldE[:,:]

    plot(sp, driver, final_time , xH, yH)

    driver.run_until(final_time)

    finalFieldH = driver['H'][:,:]

    #assert np.allclose(finalFieldH, 0.0, atol=1e-2)


@pytest.mark.skip(reason="wip")
def test_tfsf_null_field():

    def gaussian(s):
        return lambda x : np.exp(-(x)**2/(2*s**2))
    
    bdrs = {
        "XL": "Mur",
        "XU": "Mur",
        "YL": "Mur",
        "YU": "Mur"
    }

    t_final = 8.0
    s0 = 0.1

    sp = FD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels=bdrs)

    TFSF_setup = {}
    TFSF_setup["XL"] = -0.8
    TFSF_setup["XU"] = 0.8
    TFSF_setup["YL"] = -0.8
    TFSF_setup["YU"] = 0.8
    TFSF_setup["source"] = gaussian(s0)
    sp.TFSF_conditions(TFSF_setup)

    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-yH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    x, y = np.meshgrid(sp.x, sp.y)
    initialFieldE = np.exp(-(y - driver.dt/2)**2/(2*s0**2))
    driver['E']['y'][:,:] = initialFieldE[:,:]

    #plot(sp, driver, final_time , xH, yH)

    driver.run_until(t_final)

    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.0, atol=1e-3)
