import numpy as np
import matplotlib.pyplot as plt

from maxwell.driver import *
from maxwell.dg.mesh1d import *
from maxwell.dg.dg1d import *
from maxwell.fd.fd1d import *

# ······················································


def plot(sp, driver):
    for _ in range(1000):
        driver.step()
        plt.plot(sp.x, driver['E'], 'b')
        plt.plot(sp.xH, driver['H'], 'r')
        plt.ylim(-1, 1)
        plt.title(driver.timeIntegrator.time)
        plt.grid(which='both')
        plt.pause(0.01)
        plt.cla()

# ······················································


def test_buildDrivedEvolutionOperator():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    A = driver.buildDrivedEvolutionOperator(reduceToEsentialDoF=False)

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    q0 = np.concatenate([driver['E'], driver['H']])

    driver.step()
    qExpected = np.concatenate([driver['E'], driver['H']])

    q = A.dot(q0)

    assert np.allclose(qExpected, q)
    
def test_buildDrivedEvolutionOperator_reduced():
    K = 5
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 5, boundary_label="Periodic"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    A = driver.buildDrivedEvolutionOperator(reduceToEsentialDoF=True)
    
    # A = sp.reorder_by_elements(A)
    # plt.matshow(A, cmap='RdGy')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # for k in range(K):
    #     plt.vlines(k*2-0.5, -0.5, K*2-0.5, color='gray', linestyle='dashed')
    #     plt.hlines(k*2-0.5, -0.5, K*2-0.5, color='gray', linestyle='dashed')
    # plt.show()
    
    assert np.allclose(np.abs(np.linalg.eig(A)[0]), 1.0)


def test_fdtd_pec():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    # plot(sp, driver)

    driver.run_until(2.0)

    finalFieldE = driver['E'][:]
    R = np.corrcoef(initialFieldE, -finalFieldE)
    assert R[0, 1] > 0.9999


def test_fdtd_periodic():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="Periodic"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2')

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    # plot(sp, driver)

    driver.run_until(6.0)

    finalFieldE = driver['E'][:]
    R = np.corrcoef(initialFieldE, finalFieldE)
    assert R[0, 1] > 0.9999


def test_fdtd_pmc():

    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="PMC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2')

    s0 = 0.25
    initialFieldH = np.exp(-(sp.xH)**2/(2*s0**2))
    driver['H'][:] = initialFieldH[:]

    # plot(sp, driver)

    driver.run_until(2.0)

    finalFieldH = driver['H'][:]
    R = np.corrcoef(initialFieldH.ravel(), -finalFieldH.ravel())
    assert R[0, 1] > 0.9999


def test_fdtd_pmc_cfl_equals_half():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="PMC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=0.5)

    s0 = 0.25
    initialFieldH = np.exp(-(sp.xH)**2/(2*s0**2))
    driver['H'][:] = initialFieldH[:]

    # plot(sp, driver)

    driver.run_until(2.0)

    finalFieldH = driver['H'][:]

    R = np.corrcoef(initialFieldH, -finalFieldH)
    assert R[0, 1] > 0.9999


def test_fdtd_mur():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="Mur"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2')

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    # plot(sp, driver)

    driver.run_until(8.0)

    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.0, atol=1e-3)


def test_fdtd_mur_right_only():

    t_final = 8.0

    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="Mur"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    driver['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
    driver['H'][:] = np.exp(-(sp.xH - driver.dt/2)**2/(2*s0**2))

    # plot(sp, driver)

    driver.run_until(t_final)

    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.0, atol=1e-3)


def test_fdtd_right_only_mur_and_pec():

    bdrs = {
        "LEFT": "Mur",
        "RIGHT": "PEC",
    }

    t_final = 8.0

    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label=bdrs))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    driver['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
    initialFieldE = driver['E'][:]
    driver['H'][:] = np.exp(-(sp.xH - driver.dt/2)**2/(2*s0**2))

    # plot(sp, driver)

    driver.run_until(8.0)

    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.0, atol=1e-3)


def test_fdtd_check_initial_conditions_GW_right():

    x_min = -4.0
    x_max = 4.0
    k_elements = 400
    t_final = 1.0

    sp = FD1D(mesh=Mesh1D(x_min, x_max, k_elements, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    c0 = 1.0

    s0 = 0.25
    driver['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
    driver['H'][:] = np.exp(-(sp.xH - driver.dt/2)**2/(2*s0**2))

    # plot(sp, driver)

    driver.run_until(t_final)

    evolvedE = driver['E'][:]

    expectedE = np.exp(-(sp.x - c0*t_final)**2/(2*s0**2))

    R1 = np.corrcoef(expectedE, evolvedE)
    assert R1[0, 1] > 0.995


def test_fdtd_periodic_lserk():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="Periodic"))
    driver = MaxwellDriver(sp, CFL=1.5)

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    # plot(sp, driver)

    driver.run_until(2.0)

    finalFieldE = driver['E'][:]
    R = np.corrcoef(initialFieldE, finalFieldE)
    assert R[0, 1] > 0.9999


def test_tfsf_null_field():

    def gaussian(s):
        return lambda x : np.exp(-(x)**2/(2*s**2))
    
    t_final = 8.0
    s0 = 0.1

    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="Mur"))
    TFSF_setup = {}
    TFSF_setup["left"] = -0.8
    TFSF_setup["right"] = 0.8
    TFSF_setup["source"] = gaussian(s0)
    sp.TFSF_conditions(TFSF_setup)

    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    driver['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
    driver['H'][:] = np.exp(-(sp.xH - 0.5*driver.dt)**2/(2*s0**2))

    # plot(sp, driver)

    driver.run_until(t_final)

    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.0, atol=1e-3)
