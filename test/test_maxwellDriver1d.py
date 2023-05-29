import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dgtd.maxwellDriver import *
from dgtd.mesh1d import *
from dgtd.maxwell1d import *
from dgtd.mesh1d import *


def test_spatial_discretization_lift():
    sp = Maxwell1D(1, Mesh1D(0.0, 1.0, 1))
    assert   np.allclose(surface_integral_dg(1, jacobiGL(0.0,0.0,1)), 
                         np.array([[2.0,-1.0],[-1.0,2.0]]))


def test_pec():
    sp = Maxwell1D(
        n_order = 5, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    
    final_time = 3.999
    driver = MaxwellDriver(sp)
    x0 = 0.0
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

def test_energy_evolution_centered():
    ''' 
    Checks energy evolution. With Centered flux, energy should only 
    dissipate because of the LSERK4 time integration.
    '''
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    
    driver = MaxwellDriver(sp)
    driver['E'][:] = np.exp(-sp.x**2/(2*0.25**2))
    
    Nsteps = 100
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(driver['E'])
        energyH[n] = sp.getEnergy(driver['H'])
        driver.step()
        
    totalEnergy = energyE + energyH
    assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 3e-5


def test_periodic():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxType = "Upwind"
    )
    
    final_time = 1.999
    driver = MaxwellDriver(sp)
    initialFieldE = np.exp(- sp.x**2/(2*0.25**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

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
    driver = MaxwellDriver(sp)
    initialFieldE = np.exp(-(sp.x)**2/(2*0.25**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    assert np.allclose(0.0, finalFieldE, atol=1e-6)