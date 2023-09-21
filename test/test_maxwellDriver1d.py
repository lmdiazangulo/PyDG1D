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
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC")
    )
    driver = MaxwellDriver(sp)
    
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999
    
    # driver['E'][:] = initialFieldE[:]
    # for _ in range(172):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()


def test_pec_centered():
    sp = Maxwell1D(
        n_order = 5, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    driver = MaxwellDriver(sp)
    
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999
    
    # driver['E'][:] = initialFieldE[:]
    # for _ in range(172):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()



def test_pec_centered_lserk74():
    sp = Maxwell1D(
        n_order = 5, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LSERK74', CFL=1.0)
        
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.999

    driver['E'][:] = initialFieldE[:]
    # for _ in range(130):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.000001)
    #     plt.cla()

    #  #real solution
    #  lambdas = 1/np.sqrt(sp.epsilon*sp.mu)
    # real_solution = np.sin(sp.x-lambdas*t)
     
        
        
def test_pec_centered_lserk134():
    sp = Maxwell1D(
        n_order = 3, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LSERK134', CFL=3)
        
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.999

    driver['E'][:] = initialFieldE[:]
    # for _ in range(100):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()
        
def test_pec_centered_euler():
    sp = Maxwell1D(
        n_order = 3, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Upwind"
    )
    driver = MaxwellDriver(sp, timeIntegratorType='EULER',CFL=0.5659700)
        
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

    driver['E'][:] = initialFieldE[:]
    # for _ in range(1000):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.00001)
    #     plt.cla()

def test_pec_centered_lf2():
    sp = Maxwell1D(
        n_order = 3, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LF2')
        
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

    # driver['E'][:] = initialFieldE[:]
    # for _ in range(1000):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()

def test_pec_centered_lf2v():
    sp = Maxwell1D(
        n_order = 3, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LF2V')
        
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

    # driver['E'][:] = initialFieldE[:]
    # for _ in range(1000):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()

      
def test_pec_centered_ibe():
    sp = Maxwell1D(
        n_order = 3, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    driver = MaxwellDriver(sp, timeIntegratorType='IBE', CFL=1.5)
        
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    # R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
    #                 -finalFieldE.reshape(1, finalFieldE.size))
    # assert R[0,1] > 0.9999

    driver['E'][:] = initialFieldE[:]
    # for _ in range(150):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()
        
        
def test_pec_centered_cn():
    sp = Maxwell1D(
        n_order = 3, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    driver = MaxwellDriver(sp, timeIntegratorType='CN', CFL=1.0)
        
    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.9999

    driver['E'][:] = initialFieldE[:]
    # for _ in range(159):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()

        
def test_periodic_centered_dirk2():
    sp = Maxwell1D(
        n_order = 5, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxType="Upwind"
    )
    driver = MaxwellDriver(sp, timeIntegratorType='DIRK2', CFL=8)
        
    final_time = 5.999
    s0 = 0.25
    initialField = np.exp(-(sp.x)**2/(2*s0**2))
        
    driver['E'][:] = initialField[:]
    driver['H'][:] = initialField[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    driver['E'][:] = initialField[:]
    driver['H'][:] = initialField[:]
    for _ in range(1000):
        driver.step()
        plt.plot(sp.x, driver['E'],'b')
        plt.plot(sp.x, driver['H'],'r')
        plt.ylim(-1, 1)
        plt.grid(which='both')
        plt.pause(0.001)
        plt.cla()
       
# def test_pec_centered_iglrk4():
#     sp = Maxwell1D(
#         n_order = 3, 
#         mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
#         fluxType="Upwind"
#     )
#     driver = MaxwellDriver(sp, timeIntegratorType='IGLRK4', CFL=2)
        
#     final_time = 3.999
#     s0 = 0.25
#     initialField = np.exp(-(sp.x)**2/(2*s0**2))
        
#     driver['E'][:] = initialField[:]
#     driver['H'][:] = initialField[:]
#     finalFieldE = driver['E']
    
#     driver.run_until(final_time)

#     driver['E'][:] = initialField[:]
#     driver['H'][:] = initialField[:]
#     for _ in range(500):
#         driver.step()
#         plt.plot(sp.x, driver['E'],'b')
#         plt.plot(sp.x, driver['H'],'r')
#         plt.ylim(-1, 1)
#         plt.grid(which='both')
#         plt.pause(0.001)
#         plt.cla()
                
def test_energy_evolution_centered():
    ''' 
    Checks energy evolution. With Centered flux, energy should only 
    dissipate because of the LSERK4 time integration.
    '''
    sp = Maxwell1D(
        n_order = 5, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    
    driver = MaxwellDriver(sp)
    driver['E'][:] = np.exp(-sp.x**2/(2*0.25**2))
    
    Nsteps = 171
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(driver['E'])
        energyH[n] = sp.getEnergy(driver['H'])
        driver.step()
        
    totalEnergy = energyE + energyH
    assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 3e-5

    # plt.plot(energyE, 'b')
    # plt.plot(energyH, 'r')
    # plt.plot(totalEnergy, 'g')
    # plt.show()

def test_energy_evolution_centered_lf2():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=0.95)
    driver['E'][:] = np.exp(-sp.x**2/(2*0.25**2))
    
    Nsteps = 1500
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(driver['E'])
        energyH[n] = sp.getEnergy(driver['H'])
        # plt.plot(sp.x, driver['E'], 'b')
        # plt.ylim(-1,1)
        # plt.grid(which='both')
        # plt.pause(0.1)
        # plt.cla()
        driver.step()
        
    totalEnergy = energyE + energyH
    assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 0.01    
    
    # plt.figure()
    # plt.plot(energyE)
    # plt.plot(energyH)
    # plt.plot(totalEnergy)
    # plt.grid()
    # plt.show()

def test_energy_evolution_centered_lf2v():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxType="Centered"
    )
    
    driver = MaxwellDriver(sp, timeIntegratorType='LF2V', CFL=0.85)
    driver['E'][:] = np.exp(-sp.x**2/(2*0.25**2))
    
    Nsteps = 500
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(driver['E'])
        energyH[n] = sp.getEnergy(driver['H'])
        # plt.plot(sp.x, driver['E'], 'b')
        # plt.ylim(-1,1)
        # plt.grid(which='both')
        # plt.pause(0.1)
        # plt.cla()
        driver.step()
        
    totalEnergy = energyE + energyH
    assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 3e-3   
    
    # plt.figure()
    # # plt.plot(energyE)
    # # plt.plot(energyH)
    # plt.plot(totalEnergy)
    # plt.grid()
    # plt.show()


def test_periodic_tested():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxType = "Upwind"
    )
    
    final_time = 1.999
    driver = MaxwellDriver(sp)
    initialFieldE = np.sin(2*np.pi*sp.x)
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size), 
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0,1] > 0.99
    
    #Real_solution = np.cos(2*np.pi*sp.x)
    
    for _ in range(1000):
        driver.step()
        plt.plot(sp.x, driver['E'],'b')
        plt.ylim(-1, 1)
        plt.grid(which='both')
        plt.pause(0.00001)
        plt.cla()
       



def test_periodic_same_initial_conditions():
    sp = Maxwell1D(
        n_order = 2, 
        mesh = Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxType = "Upwind"
    )
    
    final_time = 1.999
    driver = MaxwellDriver(sp)
    initialFieldE = np.exp(- sp.x**2/(2*0.25**2))
    initialFieldH = np.exp(- sp.x**2/(2*0.25**2))
    
    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']
    
    driver['H'][:] = initialFieldH[:]
    finalFieldH = driver['H']
    
    driver.run_until(final_time)

    R = np.corrcoef(initialFieldH.reshape(1, initialFieldH.size), finalFieldE.reshape(1, finalFieldE.size))
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