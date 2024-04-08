import numpy as np
import matplotlib.pyplot as plt
import pytest

from dgtd.mesh2d import *
from dgtd.maxwell2d import *
from dgtd.maxwellDriver import *
from fdtd.fdtd2d import *

def test_fdtd2d_te_pec_x():
    sp = FDTD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels="PEC")
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-xH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    # while driver.timeIntegrator.time < final_time:
    #     plt.contourf(xH, yH, driver['H'], vmin=-1.0, vmax=1.0)
    #     plt.plot(sp.x, driver['E']['y'][4,:])
    #     plt.plot(sp.xH, driver['H'][4,:])
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()
    #     driver.step()

    driver.run_until(final_time)

    finalFieldH = driver['H']
    R = np.corrcoef(initialFieldH.ravel(), finalFieldH.ravel())
    assert R[0, 1] > 0.9999

def test_fdtd2d_te_pec_y():
    sp = FDTD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels="PEC")
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-yH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    # while driver.timeIntegrator.time < final_time:
    #     plt.contourf(xH, yH, driver['H'], vmin=-1.0, vmax=1.0)
    #     plt.plot(sp.x, driver['E']['y'][4,:])
    #     plt.plot(sp.xH, driver['H'][4,:])
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()
    #     driver.step()

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
    sp = FDTD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels=bdrs)
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.00)

    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-xH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    # while driver.timeIntegrator.time < final_time:
    #     plt.contourf(xH, yH, driver['H'], vmin=-1.0, vmax=1.0)
    #     plt.plot(sp.x, driver['E']['y'][4,:])
    #     plt.plot(sp.xH, driver['H'][4,:])
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()
    #     driver.step()

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
    sp = FDTD2D(x_min=-1.0, x_max=1.0, kx_elem=100, boundary_labels=bdrs)
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    final_time = 2.0

    xH, yH = np.meshgrid(sp.xH, sp.yH)
    initialFieldH = np.exp(-yH**2/(2*s0**2))
    driver['H'][:,:] = initialFieldH[:,:]

    # while driver.timeIntegrator.time < final_time:
    #     plt.contourf(xH, yH, driver['H'], vmin=-1.0, vmax=1.0)
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()
    #     driver.step()

    driver.run_until(final_time)

    finalFieldH = driver['H']
    R = np.corrcoef(initialFieldH.ravel(), -finalFieldH.ravel())
    assert R[0, 1] > 0.9999
