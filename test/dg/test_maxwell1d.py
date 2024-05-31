
from maxwell.dg.dg1d import *
from maxwell.dg.mesh1d import *

import matplotlib.pyplot as plt


def test_get_energy_N1():
    m = Mesh1D(0, 1, 10)
    sp = DG1D(1, m)
    fields = sp.buildFields()

    fields['E'].fill(0.0)
    fields['E'][0, 0] = 1.0
    assert np.isclose(sp.getEnergy(fields['E']), 0.1*1.0/3.0/2, rtol=1e-9)

    fields['E'].fill(1.0)
    assert np.isclose(sp.getEnergy(fields['E']),         1.0/2, rtol=1e-9)


def test_energy_with_operators():
    m = Mesh1D(0, 1, 10)
    sp = DG1D(1, m)

    fields = sp.buildFields()
    fields['E'].fill(1.0)
    fields['H'].fill(2.0)

    expectedEnergy = sp.getEnergy(fields['E']) + sp.getEnergy(fields['H'])

    q = sp.fieldsAsStateVector(fields)
    Mg = sp.buildGlobalMassMatrix()
    energy = 0.5*q.T.dot(Mg).dot(q)

    assert np.isclose(expectedEnergy, energy)


def test_buildEvolutionOperator_PEC():
    m = Mesh1D(0, 1, 5, boundary_label='PEC')
    sp = DG1D(1, m, 0.0)
    A = sp.buildEvolutionOperator()
    A = sp.reorder_by_elements(A)
    M = sp.buildGlobalMassMatrix()

    assert np.allclose(A.T.dot(M) + (M).dot(A), 0.0)
    assert A.shape == (20, 20)
    assert np.allclose(np.real(np.linalg.eig(A)[0]), 0)


def test_buildEvolutionOperator_Periodic():
    m = Mesh1D(0, 1, 5, boundary_label='Periodic')
    sp = DG1D(1, m, 0.0)
    A = sp.buildEvolutionOperator()
    M = sp.buildGlobalMassMatrix()

    assert np.allclose(A.T.dot(M) + (M).dot(A), 0.0)
    assert A.shape == (20, 20)
    assert np.allclose(np.real(np.linalg.eig(A)[0]), 0)


def test_stiffness_and_flux_operators():
    m = Mesh1D(0, 1, 5, boundary_label='Periodic')
    sp = DG1D(1, m, 0.0)

    A = sp.buildEvolutionOperator()
    S = sp.buildStiffnessMatrix()
    F = sp.buildFluxMatrix()

    assert np.allclose(A, S+F)


def test_buildEvolutionOperator_sorting():
    m = Mesh1D(0, 1, 3)
    sp = DG1D(2, m, 0.0)
    Np = sp.number_of_nodes_per_element()
    K = m.number_of_elements()

    A = sp.buildEvolutionOperator()
    eigA, _ = np.linalg.eig(A)

    A_by_elem = sp.reorder_by_elements(A)
    eigA_by_elem, _ = np.linalg.eig(A_by_elem)

    assert A.shape == A_by_elem.shape
    assert np.allclose(np.real(eigA_by_elem), 0)
    assert np.allclose(np.sort(np.imag(eigA)),
                       np.sort(np.imag(eigA_by_elem)))


def test_build_connected_operators():
    sp = DG1D(2, Mesh1D(0, 1, 3), 0.0)

    Ag = sp.reorder_by_elements(sp.buildEvolutionOperator())
    eigAg = np.sort(np.linalg.eig(Ag)[0])
    A, B, C, D, _, _ = sp.buildConnectedOperators(1, 1)

    Ag_reassembled_1 = np.concatenate([A, B], axis=1)
    Ag_reassembled_2 = np.concatenate([C, D], axis=1)
    Ag_reassembled = np.concatenate([Ag_reassembled_1, Ag_reassembled_2])
    eigAg_reassembled = np.sort(np.linalg.eig(Ag_reassembled)[0])

    assert np.allclose(eigAg, eigAg_reassembled)


def test_build_global_mass_matrix():
    sp = DG1D(2, Mesh1D(0, 1, 3))
    M = sp.buildGlobalMassMatrix()

    N = 2 * sp.mesh.number_of_elements() * sp.number_of_nodes_per_element()
    assert M.shape == (N, N)

    # plt.spy(M)
    # plt.show()


def test_stateVectorAsFields():
    sp = DG1D(2, Mesh1D(0, 1, 3))

    fields = sp.buildFields()
    for l, f in fields.items():
        fields[l] = np.random.rand(*f.shape)

    q_from_fields = sp.fieldsAsStateVector(fields)
    fields_from_q = sp.stateVectorAsFields(q_from_fields)

    for label, f in fields_from_q.items():
        assert np.all(fields[label] == f)
