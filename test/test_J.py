import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytest
import time

import maxwell.dg.dg1d_tools as dg
import maxwell.dg.mesh1d as ms
from maxwell.driver import *
from maxwell.dg.mesh1d import *
from maxwell.dg.dg1d import *
from maxwell.fd.fd1d import *


from nodepy import runge_kutta_method as rk

@pytest.mark.skip(reason="This test only exists to document how to define a material with conductivity.")
def test_pec_dielectrico_upwind_J():
    
    Z_0=376.73

    # Defining material properties
    epsilon_1 = 1.
    epsilon_2 = 1.
    sigma_1=0.
    sigma_2=20.

    # Defining mesh properties
    sigmas = sigma_1*np.ones(100)/Z_0
    sigmas[50:99] = sigma_2/Z_0
    epsilons = epsilon_1 * np.ones(100)
    epsilons[50:99] = epsilon_2


    # Setting up DG1D simulation
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-5.0, 5.0, 100, boundary_label="PEC"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    # Initial conditions
    final_time = 6
    s0 = 0.50
    initialFieldE = np.exp(-(sp.x+2.) ** 2 / (2 * s0 ** 2))
    initialFieldH = initialFieldE

    # Initialize fields in driver
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    
    # Run the simulation until the final time
    driver.run_until(final_time)

    # Animation loop
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]

    # for _ in range(300):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1.5)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()