import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytest
import time


from maxwell.driver import *
from maxwell.dg.mesh1d import *
from maxwell.dg.dg1d import *
from maxwell.fd.fd1d import *


from nodepy import runge_kutta_method as rk


def sinusoidal_wave_function(x, t):
    return np.sin(2*np.pi*x - 2*np.pi*t)


def test_spatial_discretization_lift():
    sp = DG1D(1, Mesh1D(0.0, 1.0, 1))
    assert np.allclose(surface_integral_dg(1, jacobiGL(0.0, 0.0, 1)),
                       np.array([[2.0, -1.0], [-1.0, 2.0]]))


def test_pec():
    sp = DG1D(
        n_order=5,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC")
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
    assert R[0, 1] > 0.9999

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
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 15, boundary_label="PEC"),
        fluxPenalty=0.0
    )
    driver = MaxwellDriver(sp, CFL=1.0)

    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-sp.x**2/(2*s0**2))

    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']

    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size),
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0, 1] > 0.9999

    # driver['E'][:] = initialFieldE[:]
    # for _ in range(172):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.05)
    #     plt.cla()


def test_pec_centered_lserk74():
    sp = DG1D(
        n_order=5,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxPenalty=0.0
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
    assert R[0, 1] > 0.999

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
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxPenalty=0.0
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LSERK134', CFL=2)

    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))

    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']

    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size),
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0, 1] > 0.999

    driver['E'][:] = initialFieldE[:]
    # for _ in range(100):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()


def test_pec_centered_euler():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxPenalty=1.0
    )
    driver = MaxwellDriver(sp, timeIntegratorType='EULER', CFL=0.5659700)

    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))

    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']

    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size),
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0, 1] > 0.9999

    driver['E'][:] = initialFieldE[:]
    # for _ in range(1000):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.00001)
    #     plt.cla()


def test_pec_centered_lf2():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxPenalty=0.0
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=0.8)

    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))

    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']

    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size),
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0, 1] > 0.9999

    # driver['E'][:] = initialFieldE[:]
    # for _ in range(1000):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'], 'b')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()


def test_pec_centered_lf2v():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxPenalty=0.0
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LF2V', CFL=0.8)

    final_time = 1.999
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))

    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']

    driver.run_until(final_time)

    R = np.corrcoef(initialFieldE.reshape(1, initialFieldE.size),
                    -finalFieldE.reshape(1, finalFieldE.size))
    assert R[0, 1] > 0.9999

    # driver['E'][:] = initialFieldE[:]
    # for _ in range(1000):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'], 'b')
    #     plt.plot(sp.x, driver['H'], 'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)
    #     plt.cla()


def test_pec_centered_lf2_and_lf2v_are_equivalent():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxPenalty=0.0
    )
    drLF = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=0.8)
    drVe = MaxwellDriver(sp, timeIntegratorType='LF2V', CFL=0.8)

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    drLF['E'][:] = initialFieldE
    drVe['E'][:] = initialFieldE

    for _ in range(200):
        drLF.step()
        drVe.step()
        assert np.allclose(drLF['H'] - drVe['H'], 0.0)

        # plt.plot(sp.x, drLF['H'], 'b')
        # plt.plot(sp.x, drVe['H'], 'g')
        # plt.ylim(-1, 1)
        # plt.grid(which='both')
        # plt.pause(0.01)
        # plt.cla()



def test_pec_centered_ibe():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxPenalty=0.0
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
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"),
        fluxPenalty=0.0
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
    assert R[0, 1] > 0.9999

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
    sp = DG1D(
        n_order=5,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxPenalty=1.0
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
    # for _ in range(100):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.001)
    #     plt.cla()

def test_buildCausallyConnectedOperators():
    sp = DG1D(
        n_order=1,
        mesh=Mesh1D(-1.0, 1.0, 11, boundary_label="Periodic"),
        fluxPenalty=1.0
    )
    dr = MaxwellDriver(sp)
    
    G = sp.reorder_by_elements(dr.buildDrivedEvolutionOperator())
    A,B,C,D,Mk,Mn = dr.buildCausallyConnectedOperators()
    
    G1 = np.concatenate([A,B], axis=1)
    G2 = np.concatenate([C,D], axis=1)
    G_reassembled = np.concatenate([G1, G2])
    
    assert np.all(G == G_reassembled)


def test_buildCausallyConnectedOperators_2():
    sp = DG1D(
        n_order=1,
        mesh=Mesh1D(-1.0, 1.0, 15, boundary_label="Periodic"),
        fluxPenalty=1.0
    )
    dr = MaxwellDriver(sp)
    
    G = sp.reorder_by_elements(dr.buildDrivedEvolutionOperator())
    A,B,C,D,Mk,Mn = dr.buildCausallyConnectedOperators(element=5)
    G1 = np.concatenate([A, B], axis=1)
    G2 = np.concatenate([C, D], axis=1)
    G_reassembled = np.concatenate([G1, G2])
    Nk = A.shape[0]
    Nn = int(B.shape[1]/2)
    new_order = np.concatenate([np.arange(Nk, Nk+Nn), np.arange(0, Nk), np.arange(Nn+Nk, 2*Nn+Nk)])
    G_f = np.array([[G_reassembled[i][j] for j in new_order] for i in new_order])
    
    assert np.all(G[:G_f.shape[0], :G_f.shape[1]] == G_f)    
   

def test_energy_evolution_centered():
    ''' 
    Checks energy evolution. With Centered flux, energy should only 
    dissipate because of the LSERK4 time integration.
    '''
    sp = DG1D(
        n_order=2,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxPenalty=0.0
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
    assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 2e-4

    # plt.plot(energyE, '.-b')
    # plt.plot(energyH, '.-r')
    # plt.plot(totalEnergy, '.-g')
    # plt.show()
    
    
def test_energy_evolution_upwind():
    ''' 
    Checks energy evolution. With Centered flux, energy should only 
    dissipate because of the LSERK4 time integration.
    '''
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxPenalty=1.0
    )

    driver = MaxwellDriver(sp)
    driver['E'][:] = np.exp(-sp.x**2/(2*0.25**2))
    driver['H'][:] = np.exp(-sp.x**2/(2*0.25**2))

    Nsteps = 171
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(driver['E'])
        energyH[n] = sp.getEnergy(driver['H'])
        driver.step()

    totalEnergy = energyE + energyH
    assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 2e-4

    # plt.plot(energyE, '.-b')
    # plt.plot(energyH, '.-r')
    # plt.plot(totalEnergy, '.-g')
    # plt.show()

def test_abnormal_energy_evolution_upwind():
    '''
        This abnormal "energy" evolution happens because the upwind fluxes 
        are not accounted correctly to compute the physical energy.
    '''
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxPenalty=1.0
    )
    dr = MaxwellDriver(sp)
    
    P = dr.buildPowerOperator()
    wP, vP = np.linalg.eig(P)
    
    unstableIndices = np.where(wP>0.0)[0]
    assert len(unstableIndices) > 0
    
    dr.fields = sp.stateVectorAsFields(np.real(vP[:,unstableIndices[0]]))

    Nsteps = 5
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(dr['E'])
        energyH[n] = sp.getEnergy(dr['H'])
        # plt.plot(sp.x, dr['E'],'b')
        # plt.plot(sp.x, dr['H'],'r')
        # plt.show()
        dr.step()

    totalEnergy = energyE + energyH
    assert (totalEnergy[1]-totalEnergy[0]) > 0
    assert np.all(totalEnergy[2:]-totalEnergy[1:-1] < 0)

    # plt.plot(energyE, '.-b')
    # plt.plot(energyH, '.-r')
    # plt.plot(totalEnergy, '.-g')
    # plt.show()


def test_energy_evolution_with_operators():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(0, 1.0, 10, boundary_label="Periodic"),
        fluxPenalty=1.0
    )
    dr = MaxwellDriver(sp)
    dr['E'][:] = np.exp(-sp.x**2/(2*0.25**2))
    dr['H'][:] = np.exp(-sp.x**2/(2*0.25**2))
    
    q0 = sp.fieldsAsStateVector(dr.fields)
    G = dr.buildDrivedEvolutionOperator()
    Mg = sp.buildGlobalMassMatrix()
    q = G.dot(q0)

    initialEnergy = q0.T.dot(Mg).dot(q0)
    finalEnergy = q.T.dot(Mg).dot(q)

    expectedPower = (finalEnergy - initialEnergy)/dr.dt
    
    PG = dr.buildPowerOperator()
    power = q0.T.dot(PG).dot(q0)

    assert np.isclose(power, expectedPower)

@pytest.mark.skip(reason="Unsure about the result.")
def test_energy_evolution_centered_lf2():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-5.0, 5.0, 50, boundary_label="Periodic"),
        fluxPenalty=0.0
    )

    dr = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=0.7)
    dr['E'][:] = np.exp(-sp.x**2/(2*0.25**2))
    dr['H'][:] = np.exp(-(sp.x-dr.dt/2)**2/(2*0.25**2))
    Nsteps = 500
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(dr['E'])
        energyH[n] = sp.getEnergy(dr['H'])
        # plt.plot(sp.x, dr['E'], 'b')
        # plt.plot(sp.x, dr['H'], 'r')
        # plt.ylim(-1,1)
        # plt.grid(which='both')
        # plt.pause(0.1)
        # plt.cla()
        dr.step()

    totalEnergy = energyE[:1] + (energyH[:-1]+ energyH[1:])*.5

    
    # assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 1e-6

    plt.plot(energyE)
    plt.plot(energyH)
    plt.plot(totalEnergy, '-b')
    plt.grid()
    plt.show()


def test_energy_evolution_centered_lf2v():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-5.0, 5.0, 50, boundary_label="Periodic"),
        fluxPenalty=0.0
    )

    dr = MaxwellDriver(sp, timeIntegratorType='LF2V', CFL=0.7)
    dr['E'][:] = np.exp(-sp.x**2/(2*0.25**2))
    dr['H'][:] = np.exp(-sp.x**2/(2*0.25**2))

    Nsteps = 500
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(dr['E'])
        energyH[n] = sp.getEnergy(dr['H'])
        # plt.plot(sp.x, dr['E'],'b')
        # plt.plot(sp.x, dr['H'],'r')
        # plt.ylim(-1,1)
        # plt.grid(which='both')
        # plt.pause(0.1)
        # plt.cla()
        dr.step()

    totalEnergy = energyE + energyH
    assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 1e-6

    # plt.figure()
    # plt.plot(energyE)
    # plt.plot(energyH)
    # plt.plot(totalEnergy)
    # plt.grid()
    # plt.show()


def test_periodic_tested():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=1.0
    )
    final_time = 1.999
    driver = MaxwellDriver(sp, timeIntegratorType='LSERK134')
    initialField = np.sin(2*np.pi*sp.x)

    driver['E'][:] = initialField[:]
    driver['H'][:] = initialField[:]
    finalFieldE = driver['E']

    driver.run_until(final_time)

    R = np.corrcoef(initialField.reshape(1, initialField.size),
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0, 1] > 0.99

    def real_function(x, t):
        return np.sin(2*np.pi*x - 2*np.pi*t)

    t = 0.0
    error = 0
    for _ in range(40):
        # plt.plot(sp.x, real_function(sp.x, t), 'g')
        # plt.plot(sp.x, driver['E'], 'b')
        # plt.plot(sp.x, driver['H'], 'r')
        # plt.ylim(-1, 1)
        # plt.grid(which='both')
        # plt.pause(0.00001)
        # plt.cla()

        error += (real_function(sp.x, t)-(driver['E']))**2

        driver.step()
        t += driver.dt

    assert (np.sqrt(error).max() < 1e-02, True)


@pytest.mark.skip(reason="Nothing is being tested.")
def test_periodic_LSERK_errors():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=0.0
    )
    final_time = 1.999
    drLSERK54 = MaxwellDriver(sp)
    drLSERK74 = MaxwellDriver(sp, timeIntegratorType='LSERK74')
    drLSERK134 = MaxwellDriver(sp, timeIntegratorType='LSERK134')

    initialField = np.sin(2*np.pi*sp.x)

    drLSERK54['E'][:] = initialField[:]
    drLSERK54['H'][:] = initialField[:]

    drLSERK74['E'][:] = initialField[:]
    drLSERK74['H'][:] = initialField[:]

    drLSERK134['E'][:] = initialField[:]
    drLSERK134['H'][:] = initialField[:]

    cpuTimeLSERK54 = time.time()
    drLSERK54.run_until(final_time*10)
    cpuTimeLSERK54 = time.time() - cpuTimeLSERK54

    # cpuTimeLSERK74 = time.time()
    # drLSERK74.run_until(final_time*10)
    # cpuTimeLSERK74 = time.time() - cpuTimeLSERK74

    cpuTimeLSERK134 = time.time()
    drLSERK134.run_until(final_time*10)
    cpuTimeLSERK134 = time.time() - cpuTimeLSERK134

    t = 0.0
    for _ in range(100):
        # plt.plot(sp.x, sinusoidal_wave_function(
        #     sp.x, drLSERK54.timeIntegrator.time), 'g')
        # plt.plot(sp.x, drLSERK54['E'], 'b')
        # plt.plot(sp.x, drLSERK74['E'], 'r')
        # plt.plot(sp.x, drLSERK134['E'], 'cyan')

        # plt.ylim(-1, 1)
        # plt.grid(which='both')
        # plt.pause(0.00001)
        # plt.cla()

        drLSERK54.step()
        drLSERK74.step()
        drLSERK134.step()

    # R = np.corrcoef(initialField.reshape(1, initialField.size),
    #                 finalFieldE.reshape(1, finalFieldE.size))
    # assert R[0,1] > -0.99


@pytest.mark.skip(reason="Nothing is being tested.")
def test_periodic_implicit_errors():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=1.0
    )
    final_time = 1.999

    drLF2V = MaxwellDriver(sp, timeIntegratorType='LF2V')
    drIBE = MaxwellDriver(sp, timeIntegratorType='IBE')
    drCN = MaxwellDriver(sp, timeIntegratorType='CN')

    initialField = sinusoidal_wave_function(sp.x, drCN.timeIntegrator.time)

    drLF2V['E'][:] = initialField[:]
    drLF2V['H'][:] = initialField[:]

    drIBE['E'][:] = initialField[:]
    drIBE['H'][:] = initialField[:]

    drCN['E'][:] = initialField[:]
    drCN['H'][:] = initialField[:]

    drLF2V.run_until(final_time)
    drIBE.run_until(final_time)
    drCN.run_until(final_time)
    for _ in range(100):
        # plt.plot(sp.x, sinusoidal_wave_function(
        #     sp.x, drCN.timeIntegrator.time), 'g')
        # plt.plot(sp.x, drIBE['E'], 'brown')
        # plt.plot(sp.x, drCN['E'], 'm')

        # plt.ylim(-1, 1)
        # plt.grid(which='both')
        # plt.pause(0.00001)
        # plt.cla()

        drIBE.step()
        drCN.step()


@pytest.mark.skip(reason="Nothing is being tested.")
def test_periodic_euler_errors():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=0.0
    )

    final_time = 1.999
    initialField = sinusoidal_wave_function(sp.x, 0.0)
    # initialField = np.exp(-(sp.x)**2/(0.25))
    # initialField = np.random.random(sp.x.shape)

    drEULER = MaxwellDriver(sp, timeIntegratorType='EULER', CFL=0.1)
    drEULER['E'][:] = initialField[:]
    drEULER['H'][:] = initialField[:]
    drEULER.run_until(final_time)

    Nsteps = 50
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(drEULER['E'])
        energyH[n] = sp.getEnergy(drEULER['H'])
        plt.plot(sp.x, drEULER['E'], 'b')
        plt.ylim(-1, 1)
        plt.grid(which='both')
        plt.pause(0.1)
        plt.cla()
        drEULER.step()

    totalEnergy = energyE + energyH
    # assert np.abs(totalEnergy[-1]-totalEnergy[0]) < 3e-3

    plt.figure()
    # plt.plot(energyE)
    # plt.plot(energyH)
    plt.plot(totalEnergy)
    plt.grid()
    plt.show()

    for _ in range(50):
        # plt.plot(sp.x, sinusoidal_wave_function(sp.x, drEULER.timeIntegrator.time), 'g')
        plt.plot(sp.x, drEULER['E'], 'y')
        plt.plot(sp.x, drEULER['H'], 'b')

        plt.ylim(-1, 1)
        plt.grid(which='both')
        plt.pause(0.00001)
        plt.cla()

        drEULER.step()


@pytest.mark.skip(reason="Nothing is being tested.")
def test_computational_cost():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=0.0
    )
    final_time = 1.999
    drLSERK54 = MaxwellDriver(sp, CFL=20)
    drLSERK134 = MaxwellDriver(sp, timeIntegratorType='LSERK134', CFL=26)
    drIBE = MaxwellDriver(sp, timeIntegratorType='IBE', CFL=75.8)
    drCN = MaxwellDriver(sp, timeIntegratorType='CN', CFL=75.8)

    initialField = np.sin(2*np.pi*sp.x)

    drLSERK54['E'][:] = initialField[:]
    drLSERK54['H'][:] = initialField[:]

    drLSERK134['E'][:] = initialField[:]
    drLSERK134['H'][:] = initialField[:]

    drIBE['E'][:] = initialField[:]
    drIBE['H'][:] = initialField[:]

    drCN['E'][:] = initialField[:]
    drCN['H'][:] = initialField[:]

    cpuTimeLSERK54 = time.time()
    drLSERK54.run_until(final_time*10)
    cpuTimeLSERK54 = time.time() - cpuTimeLSERK54

    cpuTimeLSERK134 = time.time()
    drLSERK134.run_until(final_time*10)
    cpuTimeLSERK134 = time.time() - cpuTimeLSERK134

    cpuTimeIBE = time.time()
    drIBE.run_until(final_time*10)
    cpuTimeIBE = time.time() - cpuTimeIBE

    cpuTimeCN = time.time()
    drCN.run_until(final_time*10)
    cpuTimeCN = time.time() - cpuTimeCN

    print("dt", drLSERK134.dt)
    print("Final Time", drLSERK134.timeIntegrator.time)

    print("CPU TIME LSERK134", cpuTimeLSERK134/drLSERK134.timeIntegrator.time)
    print("CPU TIME LSERK54", cpuTimeLSERK54/drLSERK54.timeIntegrator.time)
    print("CPU TIME IBE", cpuTimeIBE/drIBE.timeIntegrator.time)
    print("CPU TIME CN", cpuTimeCN / drCN.timeIntegrator.time)


@pytest.mark.skip(reason="Nothing is being tested.")
def test_errors():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=0.0
    )
    final_time = 1.999
    drLSERK54 = MaxwellDriver(sp, CFL=3)
    drLSERK134 = MaxwellDriver(sp, timeIntegratorType='LSERK134', CFL=3.2)
    drIBE = MaxwellDriver(sp, timeIntegratorType='IBE', CFL=75.8)
    drCN = MaxwellDriver(sp, timeIntegratorType='CN', CFL=75.8)
    initialField = np.sin(2*np.pi*sp.x)

    drLSERK54['E'][:] = initialField[:]
    drLSERK54['H'][:] = initialField[:]
    finalFieldE1 = drLSERK54['E']

    drLSERK134['E'][:] = initialField[:]
    drLSERK134['H'][:] = initialField[:]
    finalFieldE3 = drLSERK134['E']

    drIBE['E'][:] = initialField[:]
    drIBE['H'][:] = initialField[:]
    finalFieldE6 = drIBE['E']

    drCN['E'][:] = initialField[:]
    drCN['H'][:] = initialField[:]
    finalFieldE7 = drCN['E']

    drLSERK54.run_until(final_time*10)
    drLSERK134.run_until(final_time*10)
    drIBE.run_until(final_time*10)
    drCN.run_until(final_time*10)

    # R = np.corrcoef(initialField.reshape(1, initialField.size),
    #                 finalFieldE.reshape(1, finalFieldE.size))
    # assert R[0,1] > 0.99

    def real_function(x, t):
        return np.sin(2*np.pi*x - 2*np.pi*t)

    t = 0.0
    error_abs = 0
    error = 0

    # for _ in range(500):
    #     # plt.plot(sp.x, real_function(sp.x, driver1.timeIntegrator.time), 'g')
    #     # #plt.plot(error_abs,'b')
    #     # plt.plot(sp.x, driver1['E'],'b')
    #     # plt.plot(sp.x, driver2['E'],'r')
    #     # plt.plot(sp.x, driver3['E'],'cyan')
    #     # plt.plot(sp.x, driver4['E'],'y')

    #     # plt.plot(sp.x, driver6['E'],'brown')
    #     # plt.plot(sp.x, driver7['E'],'m')

    #     # plt.ylim(-1, 1)
    #     # plt.grid(which='both')
    #     # plt.pause(0.00001)
    #     # plt.cla()
    #     # real_value = real_function(sp.x, t)
    #     # aprox_value = (driver2['E'])

    #     drLSERK54.step()
    #     drLSERK134.step()
    #     drIBE.step()
    #     drCN.step()

    print(drLSERK54.timeIntegrator.time)
    plt.plot(sp.x, real_function(sp.x, drIBE.timeIntegrator.time)+1, 'y')
    plt.plot(sp.x, drCN['E']-1, 'b')
    plt.plot(sp.x, drIBE['E'], 'g')
    plt.pause(1)
    plt.cla()
    plt.show()

    error_RMSE_LSERK54 = np.sqrt(np.sum(real_function(
        sp.x, drLSERK54.timeIntegrator.time)-(drLSERK54['E']))**2)/drLSERK54['E'].size
    error_RMSE_LSERK134 = np.sqrt(np.sum(real_function(
        sp.x, drLSERK134.timeIntegrator.time)-(drLSERK134['E']))**2)/drLSERK134['E'].size
    error_RMSE_CN = np.sqrt(np.sum(real_function(
        sp.x, drCN.timeIntegrator.time)-(drCN['E']))**2)/drCN['E'].size
    error_RMSE_IBE = np.sqrt(np.sum(real_function(
        sp.x, drIBE.timeIntegrator.time)-(drIBE['E']))**2)/drIBE['E'].size

    print("dt ", drCN.dt)
    print("final time ", drCN.timeIntegrator.time)
    print("dt ", drIBE.dt)
    print("final time ", drIBE.timeIntegrator.time)

    print("ERROR IBE ", error_RMSE_IBE)
    print("ERROR CN ", error_RMSE_CN)
    print("ERROR LSERK54 ", error_RMSE_LSERK54)
    print("ERROR LSERK134", error_RMSE_LSERK134)
    # Show dt in driver

    assert (np.sqrt(error_abs).max() < 1e-02, True)


@pytest.mark.skip(reason="Nothing is being tested.")
def test_max_time_step():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=0.0
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LSERK134')
    p, q = rk.loadRKM('SSP75').stability_function(mode='float')

    A_cen = sp.buildEvolutionOperator()
    A_cen_eig, _ = np.linalg.eig(A_cen)
    tol = 1e-3
    max_norm = 0.0
    iterations = 0
    dt = 2
    out = False

    while out == False and iterations != 1e4:

        eigs, _ = np.linalg.eig(A_cen)
        eigs *= dt
        z_rk44 = p(eigs)/q(eigs)
        z_rk44_mod = np.imag(z_rk44)

        if (np.max(z_rk44_mod) > 1.0):
            dt *= 0.5
        elif (np.max(z_rk44_mod) <= 1.0):
            while (np.max(z_rk44_mod) <= 1.0):
                dt *= 1.05
                eigs, _ = np.linalg.eig(A_cen)
                eigs *= dt
                z_rk44 = p(eigs)/q(eigs)
                z_rk44_mod = np.imag(z_rk44)
                if (np.max(z_rk44_mod) > 1.0):
                    dt /= 1.05
                    eigs, _ = np.linalg.eig(A_cen)
                    eigs *= dt
                    z_rk44 = p(eigs)/q(eigs)
                    z_rk44_mod = np.imag(z_rk44)
                    max_norm = np.max(z_rk44_mod)
                    out = True
                    break

        max_norm = np.max(z_rk44_mod)
        iterations += 1

    print("Method")
    print("Time Step: " + str(dt))
    print("Final Max Module: " + str(np.max(np.imag(z_rk44))))
    print("Final Min Module: " + str(np.min(np.imag(z_rk44))))
    print("Iterations: " + str(iterations))
    if (iterations == 100):
        print("Iterations reached.")

    x = np.linspace(-1.0, 1.0, 20)
    X, Y = np.meshgrid(x, x)
    z = X+1j*Y
    f = p(z)/q(z)

    plt.figure(dpi=200)
    levels = np.linspace(0, 5, 100)
    c1 = plt.contour(X, X, np.abs(f), levels=[1.0], colors='red')
    plt.scatter(np.real(eigs), np.imag(eigs), s=3)
    plt.grid(which='both')
    plt.grid(which='both')
    plt.pause(0.00001)
    plt.cla()


def test_periodic_same_initial_conditions():
    sp = DG1D(
        n_order=2,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="Periodic"),
        fluxPenalty=1.0
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

    R = np.corrcoef(initialFieldH.reshape(1, initialFieldH.size),
                    finalFieldE.reshape(1, finalFieldE.size))
    assert R[0, 1] > 0.9999


def test_sma():
    sp = DG1D(
        n_order=2,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="SMA"),
        fluxPenalty=1.0
    )

    final_time = 3.999
    driver = MaxwellDriver(sp)
    initialFieldE = np.exp(-(sp.x)**2/(2*0.25**2))

    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']

    driver.run_until(final_time)

    assert np.allclose(0.0, finalFieldE, atol=1e-6)

def test_abc_centered():
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="ABC"),
        fluxPenalty=0.0
    )

    final_time = 3.999
    driver = MaxwellDriver(sp)
    initialFieldE = np.exp(-(sp.x)**2/(2*0.25**2))

    driver['E'][:] = initialFieldE[:]
    finalFieldE = driver['E']

    driver.run_until(final_time)

    assert np.allclose(0.0, finalFieldE, atol=1e-4)


    # driver['E'][:] = initialFieldE[:]
    # for _ in range(172):
    #     driver.step()
    #     plt.plot(sp.x, driver['E'],'b')
    #     plt.plot(sp.x, driver['H'],'r')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.pause(0.05)
    #     plt.cla()


def test_buildDrivedEvolutionOperator():
    sp = DG1D(
        n_order=2,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=1.0
    )
    driver = MaxwellDriver(sp)

    G = driver.buildDrivedEvolutionOperator()

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    q0 = sp.fieldsAsStateVector(driver.fields)
    
    q = np.zeros_like(q0)
    q[:] = q0[:] 
    for _ in range(50):
        driver.step()
        qExpected = sp.fieldsAsStateVector(driver.fields)
        q = G.dot(q)
        assert np.allclose(qExpected, q)
    


def test_buildDrivedEvolutionOperator_LF2V():
    sp = DG1D(
        n_order=2,
        mesh=Mesh1D(-1.0, 1.0, 20, boundary_label="Periodic"),
        fluxPenalty=0.0
    )
    driver = MaxwellDriver(sp, timeIntegratorType='LF2V')

    G = driver.buildDrivedEvolutionOperator(reduceToEssentialDoF=False)

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    q0 = sp.fieldsAsStateVector(driver.fields)

    driver.step()
    qExpected = sp.fieldsAsStateVector(driver.fields)

    q = G.dot(q0)

    assert np.allclose(qExpected, q)
