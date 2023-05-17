import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dgtd.maxwellDriver1d import *
from dgtd.mesh1d import *

import dgtd.mesh1d as ms
import dgtd.maxwell1d as mw


def test_spatial_discretization_lift():
    sp = Maxwell1D(1, Mesh1D(0.0, 1.0, 1))
    assert   np.allclose(surface_integral_dg(1, vandermonde(1, jacobiGL(0.0,0.0,1))), 
                         np.array([[2.0,-1.0],[-1.0,2.0]]))


def test_pec():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Upwind"
    )
    
    final_time = 3.999
    driver = MaxwellDriver1D(sp)
    x0 = 0.0
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver.E[:] = initialFieldE[:]
    
    driver.run_until(final_time)

    finalFieldE = driver.E
    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

def test_periodic():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxType="Upwind"
    )
    
    final_time = 1.999
    driver = MaxwellDriver1D(sp)
    x0 = 0.0
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver.E[:] = initialFieldE[:]
    
    driver.run_until(final_time)

    finalFieldE = driver.E
    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

def test_sma():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="SMA"),
        fluxType="Upwind"
    )
    
    final_time = 3.999
    driver = MaxwellDriver1D(sp)
    x0 = 0.0
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver.E[:] = initialFieldE[:]
    
    driver.run_until(final_time)

    finalFieldE = driver.E
    assert np.allclose(0.0, finalFieldE, atol=1e-6)