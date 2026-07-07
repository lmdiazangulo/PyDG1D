import numpy as np
import matplotlib.pyplot as plt
import pytest

from maxwell.dg.mesh2d import *
from maxwell.dg.dg2d import *
from maxwell.driver import *
from maxwell.fd.fd2d import *

TEST_DATA_FOLDER = 'testData/'


def resonant_cavity_ez_field(x, y, t):
    """(1,1) TEz cavity mode on the mesh bounding box."""
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    Lx = xmax - xmin
    Ly = ymax - ymin
    m, n = 1, 1
    xhat = (x - xmin) / Lx
    yhat = (y - ymin) / Ly
    omega = np.pi * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)
    return (
        np.sin(m * np.pi * xhat)
        * np.sin(n * np.pi * yhat)
        * np.cos(omega * t)
    )


def test_pec():
    N = 2
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(N, msh, 'Centered')

    driver = MaxwellDriver(sp, CFL=1)
    driver['Ez'][:] = resonant_cavity_ez_field(sp.x, sp.y, 0)
    for _ in range(40):
        driver.step()

    ez_expected = resonant_cavity_ez_field(sp.x, sp.y, driver.timeIntegrator.time)
    R = np.corrcoef(ez_expected, driver['Ez'])
    assert R[0, 1] > 0.9


def test_pec_evolution():
    N = 5
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    sp = Maxwell2D(N, msh, 'Centered')
    driver = MaxwellDriver(sp, CFL=1)
    driver['Ez'][:] = resonant_cavity_ez_field(sp.x, sp.y, 0)

    for _ in range(40):
        driver.step()

    ez_expected = resonant_cavity_ez_field(
        sp.x, sp.y, driver.timeIntegrator.time
    )
    R = np.corrcoef(ez_expected.ravel(), driver['Ez'].ravel())
    assert R[0, 1] > 0.85


def test_periodic_upwind():
    N = 2
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K2.neu')
    msh.boundary_label = 'Periodic'
    sp = Maxwell2D(N, msh, 'Upwind')

    driver = MaxwellDriver(sp, CFL=0.5)
    driver['Ez'][:] = np.sin(np.pi * sp.x) * np.sin(np.pi * sp.y)

    for _ in range(20):
        driver.step()

    assert np.max(np.abs(driver['Ez'])) > 0.01
