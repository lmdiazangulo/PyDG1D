import numpy as np
import matplotlib.pyplot as plt

from maxwell.driver import *
from maxwell.mor import *
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
        plt.pause(0.1)
        plt.cla()

# ······················································


def test_buildDrivedEvolutionOperator():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 10, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    A = driver.buildDrivedEvolutionOperator(reduceToEssentialDoF=False)

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
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="PEC"))
    
    A_0_9 = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=0.9).buildDrivedEvolutionOperator()
    A_1_0 = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0).buildDrivedEvolutionOperator()
    A_1_01 = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.01).buildDrivedEvolutionOperator()
    
    # A = sp.reorder_by_elements(A)
    # plt.matshow(A, cmap='RdGy')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # for k in range(K):
    #     plt.vlines(k*2-0.5, -0.5, K*2-0.5, color='gray', linestyle='dashed')
    #     plt.hlines(k*2-0.5, -0.5, K*2-0.5, color='gray', linestyle='dashed')
    # plt.show()
    
    assert np.allclose(np.abs(np.linalg.eig(A_0_9)[0]), 1.0)
    assert np.allclose(np.abs(np.linalg.eig(A_1_0)[0]), 1.0)
    assert np.any(np.abs(np.linalg.eig(A_1_01)[0]) - 1.0 > 0)


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
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    # plot(sp, driver)

    driver.run_until(8.0)

    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.0, atol=1e-3)
    
def test_fdtd_mur_constant_mode():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label='Mur'))

    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    driver['E'][:] = 0.1
    driver['H'][:] = 0.1
    
    # plot(sp, driver)

    driver.run_until(1.0)

    finalFieldE = driver['E'][:]
    assert np.allclose(finalFieldE, 0.1, atol=1e-12)


def test_fdtd_mur_right_only():

    t_final = 8.0

    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="Mur"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

    s0 = 0.25
    driver['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
    driver['H'][:] = np.exp(-(sp.xH + driver.dt/2)**2/(2*s0**2))

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

    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 50, boundary_label=bdrs))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=0.7)
    
    s0 = 0.25
    driver['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
    initialFieldE = driver['E'][:]
    driver['H'][:] = -np.exp(-(sp.xH - driver.dt/2)**2/(2*s0**2))

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
    
def test_energy_evolution():
    '''
        Energy evolution for LF2 needs to account for the fact that
        the magnetic field is staggered in time. 
        This requires a special operator to compute energy.
    '''
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 100, boundary_label="Periodic"))
    dr = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=0.7)

    G = dr.buildDrivedEvolutionOperator(reduceToEssentialDoF=True)    
    s0 = 0.25
    dr['E'][:] = np.exp(-(sp.x)**2/(2*s0**2))
   
    if sp.mesh.boundary_label['LEFT'] == 'Periodic':    
        removeLastE = True

    Nsteps = 300
    energyE = np.zeros(Nsteps)
    energyH = np.zeros(Nsteps)
    totalEnergy = np.zeros(Nsteps)
    for n in range(Nsteps):
        energyE[n] = sp.getEnergy(dr['E'], removeLast=removeLastE)
        energyH[n] = sp.getEnergy(dr['H'])
        totalEnergy[n] = sp.getTotalEnergy(G, dr.fields)
        dr.step()

    # plt.plot(energyE + energyH) 
    # plt.plot((energyH[:-1] + energyH[1:])*0.5 + energyE[:-1])
    # # plt.plot(energyE)
    # # plt.plot(energyH)
    # plt.plot(totalEnergy)
    # plt.show()
    assert np.isclose(totalEnergy[0],totalEnergy[-1])


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

def test_comparison_DrivedEvolutionOperator_with_OperatorWithAlternateBase():
    for k in range(5, 51, 1):
        sp = FD1D(mesh=Mesh1D(-1.0, 1.0, k, boundary_label="Mur"))
        driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)

        A = driver.buildDrivedEvolutionOperator(reduceToEssentialDoF=False)

        A_alternate = driver.buildDrivedEvolutionOperator_FromAlternateBasis()

        if (scipy.sparse.issparse(A_alternate)):
            A_alternate = A_alternate.todense()

        assert np.allclose(A_alternate, A)

def test_snapshots_creation():

    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 5, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    rom = ModelOrderReduction(sp, driver.buildDrivedEvolutionOperator_FromAlternateBasis())

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    Q_skip1 = rom.buildSnapshots_fromInitialState(driver.sp.fieldsAsStateVector(driver.fields), finalTime=4.0, time_step_skip=1)
    Q_skip2 = rom.buildSnapshots_fromInitialState(driver.sp.fieldsAsStateVector(driver.fields), finalTime=8.0, time_step_skip=2)
    Q_skip3 = rom.buildSnapshots_fromInitialState(driver.sp.fieldsAsStateVector(driver.fields), finalTime=12.0, time_step_skip=3)

    assert np.allclose(Q_skip1[:, 2], Q_skip2[:, 1])
    assert np.allclose(Q_skip1[:, 3], Q_skip3[:, 1])
    assert np.allclose(Q_skip2[:, 3], Q_skip3[:, 2])

def test_unitary_vectors_ChangeBasis_operator():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 1000, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    rom = ModelOrderReduction(sp, driver.buildDrivedEvolutionOperator_FromAlternateBasis())
    
    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]

    Q = rom.buildSnapshots_fromInitialState(driver.sp.fieldsAsStateVector(driver.fields), finalTime=4.0, time_step_skip=1)
    Ur, Ar, Ur1, Ar1 = rom.buildReducedOrderModel_truncated_SVD()

    for k in range(Ur1.shape[1]):
        assert np.isclose(1, np.linalg.norm(Ur1[:,k]), rtol=1e-3)

    for k in range(Ur.shape[1]):
        assert np.isclose(1, np.linalg.norm(Ur[:,k]), rtol=1e-3)

def test_orthogonality_ChangeBasis_operator():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 1000, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    rom = ModelOrderReduction(sp, driver.buildDrivedEvolutionOperator_FromAlternateBasis())

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]
    rom.buildSnapshots_fromInitialState(driver.sp.fieldsAsStateVector(driver.fields), finalTime=4.0, time_step_skip=1)

    Ur, Ar, Ur1, Ar1 = rom.buildReducedOrderModel_truncated_SVD()

    assert np.allclose(Ur.T.dot(Ur), np.eye(Ur.shape[1]), atol=1e-3)
    assert np.allclose(Ur1.T.dot(Ur1), np.eye(Ur1.shape[1]), atol=1e-2)


def test_comparison_fullsolver_reducedOrderModel_POD():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 1000, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    rom = ModelOrderReduction(sp, driver.buildDrivedEvolutionOperator_FromAlternateBasis())

    number_of_time_steps = 1750
    final_time_of_simulation = number_of_time_steps * driver.dt

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]
    rom.buildSnapshots_fromInitialState(driver.sp.fieldsAsStateVector(driver.fields), finalTime=1.0, time_step_skip=10)

    Ur, Ar, Ur1, Ar1 = rom.buildReducedOrderModel_truncated_SVD()
    qf_r = rom.run_until_ROM(driver.sp.fieldsAsStateVector(driver.fields), Ur, Ar, Ur1, Ar1, final_time_of_simulation, errorCriterionForAdaptative=1e-6)
    
    driver.run_until(final_time_of_simulation)
    qf_solver = driver.sp.fieldsAsStateVector(driver.fields)

    # q = copy.deepcopy(driver.sp.fieldsAsStateVector(driver.fields))
    # q_r = Ur.T @ q
    # q_r1 = Ur1.T @ q
    # for t in range(number_of_time_steps):
    #     q_r, q_r1, Ur, Ar, Ur1, Ar1 = rom.step_ROM(q_r, q_r1, Ur, Ar, Ur1, Ar1, errorCriterionForAdaptative=1e-6)
    #     driver.step()
    #     plt.plot(sp.x, driver['E'], label='Full solver electric field')
    #     plt.plot(sp.x, driver.sp.stateVectorAsFields(Ur @ q_r)['E'], '--', label='Reduced-order model electric field')
    #     plt.title(f'Time = {t * driver.dt:.4f} s')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.legend()
    #     plt.pause(0.001)
    #     plt.cla()


    # FullFields = driver.sp.stateVectorAsFields(qf_solver)
    # Fields_reduced = driver.sp.stateVectorAsFields(qf_r)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # axes[0].plot(sp.xH, FullFields['H'], '-', label='Full solver magnetic field')
    # axes[0].plot(sp.xH, Fields_reduced['H'], '--', label='Reduced-order model magnetic field')
    # axes[0].set_title('Magnetic Field (H)')
    # axes[0].grid()
    # axes[0].legend()
    # axes[1].plot(sp.x, FullFields['E'], '-', label='Full solver electric field')
    # axes[1].plot(sp.x, Fields_reduced['E'], '--', label='Reduced-order model electric field')
    # axes[1].set_title('Electric Field (E)')
    # axes[1].grid()
    # axes[1].legend()
    # plt.tight_layout()
    # plt.show()

    assert np.linalg.norm(qf_solver - qf_r, ord=1) / np.linalg.norm(qf_solver, ord=1) < 5e-3
    
def test_comparison_fullsolver_ROM_specific_point():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 1000, boundary_label="PEC"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    rom = ModelOrderReduction(sp, driver.buildDrivedEvolutionOperator_FromAlternateBasis())

    number_of_time_steps = 1750
    final_time_of_simulation = number_of_time_steps * driver.dt

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]
    
    rom.buildSnapshots_fromInitialState(driver.sp.fieldsAsStateVector(driver.fields), finalTime=1.0, time_step_skip=10)
    Ur, Ar, Ur1, Ar1 = rom.buildReducedOrderModel_truncated_SVD()

    q = copy.deepcopy(driver.sp.fieldsAsStateVector(driver.fields))
    q_r = Ur.T @ q
    q_r1 = Ur1.T @ q

    for t in range(number_of_time_steps):
        q_r, q_r1, Ur, Ar, Ur1, Ar1 = rom.step_ROM(q_r, q_r1, Ur, Ar, Ur1, Ar1, errorCriterionForAdaptative=1e-6)
        driver.step()

        assert np.isclose((driver.sp.stateVectorAsFields(Ur @ q_r))['E'][500], driver['E'][500], atol=5e-3)


def test_comparison_fullsolver_ROM_MurBoundaries():
    sp = FD1D(mesh=Mesh1D(-1.0, 1.0, 1000, boundary_label="Mur"))
    driver = MaxwellDriver(sp, timeIntegratorType='LF2', CFL=1.0)
    rom = ModelOrderReduction(sp, driver.buildDrivedEvolutionOperator_FromAlternateBasis(), energyThreshold=1e-9)

    number_of_time_steps = 2250
    final_time_of_simulation = number_of_time_steps * driver.dt

    s0 = 0.25
    initialFieldE = np.exp(-(sp.x)**2/(2*s0**2))
    driver['E'][:] = initialFieldE[:]
    rom.buildSnapshots_fromInitialState(driver.sp.fieldsAsStateVector(driver.fields), finalTime=1.0, time_step_skip=10)

    Ur, Ar, Ur1, Ar1 = rom.buildReducedOrderModel_truncated_SVD()
    qf_r = rom.run_until_ROM(driver.sp.fieldsAsStateVector(driver.fields), Ur, Ar, Ur1, Ar1, final_time_of_simulation, errorCriterionForAdaptative=1e-6, adaptativeSteps=10)
    
    driver.run_until(final_time_of_simulation)
    qf_solver = driver.sp.fieldsAsStateVector(driver.fields)

    # q = copy.deepcopy(driver.sp.fieldsAsStateVector(driver.fields))
    # q_r = Ur.T @ q
    # q_r1 = Ur1.T @ q
    # for t in range(number_of_time_steps):
    #     q_r, q_r1, Ur, Ar, Ur1, Ar1 = rom.step_ROM(q_r, q_r1, Ur, Ar, Ur1, Ar1, errorCriterionForAdaptative=1e-6, adaptativeSteps=10)
    #     driver.step()
    #     plt.plot(sp.x, driver['E'], label='Full solver electric field')
    #     plt.plot(sp.x, driver.sp.stateVectorAsFields(Ur @ q_r)['E'], '--', label='Reduced-order model electric field')
    #     plt.title(f'Time = {t * driver.dt:.4f} s')
    #     plt.ylim(-1, 1)
    #     plt.grid(which='both')
    #     plt.legend()
    #     plt.pause(0.001)
    #     plt.cla()

    assert np.allclose(qf_solver, qf_r, atol=5e-4)
