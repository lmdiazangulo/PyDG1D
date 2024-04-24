
from maxwell.dg.dg1d import *
from maxwell.dg.mesh1d import *

import matplotlib.pyplot as plt


def test_get_energy_N1():
    m = Mesh1D(0, 1, 10)
    sp = DG1D(1, m)
    fields = sp.buildFields()

    fields['E'].fill(0.0)
    fields['E'][0, 0] = 1.0
    assert np.isclose(sp.getEnergy(fields['E']), 0.1*1.0/3.0, rtol=1e-9)

    fields['E'].fill(1.0)
    assert np.isclose(sp.getEnergy(fields['E']),         1.0, rtol=1e-9)


def test_buildEvolutionOperator_PEC():
    m = Mesh1D(0, 1, 5, boundary_label='PEC')
    sp = DG1D(1, m, "Centered")
    A = sp.buildEvolutionOperator()
    A = sp.reorder_array(A, 'byElements')
    M = sp.buildGlobalMassMatrix()

    
    assert np.allclose(A.T.dot(M) + (M).dot(A), 0.0)
    assert A.shape == (20, 20)
    assert np.allclose(np.real(np.linalg.eig(A)[0]), 0)


def test_buildEvolutionOperator_Periodic():
    m = Mesh1D(0, 1, 5, boundary_label='Periodic')
    sp = DG1D(1, m, "Centered")
    A = sp.buildEvolutionOperator()
    M = sp.buildGlobalMassMatrix()

    assert np.allclose(A.T.dot(M) + (M).dot(A), 0.0)
    assert A.shape == (20, 20)
    assert np.allclose(np.real(np.linalg.eig(A)[0]), 0)


def test_buildEvolutionOperator_sorting():
    m = Mesh1D(0, 1, 3)
    sp = DG1D(2, m, "Centered")
    Np = sp.number_of_nodes_per_element()
    K = m.number_of_elements()

    A = sp.buildEvolutionOperator()
    eigA, _ = np.linalg.eig(A)

    A_by_elem = sp.reorder_array(A, 'byElements')
    eigA_by_elem, _ = np.linalg.eig(A_by_elem)

    assert A.shape == A_by_elem.shape
    assert np.allclose(np.real(eigA_by_elem), 0)
    assert np.allclose(np.sort(np.imag(eigA)),
                       np.sort(np.imag(eigA_by_elem)))


def test_build_global_mass_matrix():
    sp = DG1D(2, Mesh1D(0, 1, 3))
    M = sp.buildGlobalMassMatrix()

    # plt.spy(M)
    # plt.show()

    N = 2 * sp.mesh.number_of_elements() * sp.number_of_nodes_per_element()
    assert M.shape == (N, N)
