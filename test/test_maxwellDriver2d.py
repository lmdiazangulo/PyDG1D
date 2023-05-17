import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dgtd.maxwellDriver2d import *
from dgtd.mesh2d import *


def test_pec():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Upwind"
    )
    
    final_time = 3.999
    driver = MaxwellDriver2D(sp)
    x0 = 0.0
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver.E[:] = initialFieldE[:]
    
    driver.run_until(final_time)

    finalFieldE = driver.E
    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999