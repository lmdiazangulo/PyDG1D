
from fdtd.fdtd1d import *
from dgtd.mesh1d import *

import matplotlib.pyplot as plt


def test_ctor():
    K = 10
    m = Mesh1D(0, 1, K)
    sp = FDTD1D(m)

    assert len(sp.x) == K + 1
    assert len(sp.xH) == K

    assert len(sp.xH) == m.number_of_elements()


def test_buildFields():
    K = 20
    sp = FDTD1D(Mesh1D(0, 5, K))

    fields = sp.buildFields()

    assert len(fields['E']) == K + 1
    assert len(fields['H']) == K


def test_buildEvolutionOperator():
    K = 5
    sp = FDTD1D(Mesh1D(0, 5, K))

    try:
        A = sp.buildEvolutionOperator()
    except ValueError:
        assert False, "buildEvolutionOperator() raised ValueError unexpectedly!"

    # plt.spy(A)
    # plt.show()


def test_buildEvolutionOperator_periodic():
    K = 5
    sp = FDTD1D(Mesh1D(0, 5, K, boundary_label='Periodic'))

    try:
        A = sp.buildEvolutionOperator()
    except ValueError:
        assert False, "buildEvolutionOperator() raised ValueError unexpectedly!"

    # plt.spy(A)
    # plt.show()



def test_buildEvolutionOperator_sorting():
    m = Mesh1D(0, 1, 3, 'Periodic')
    sp = FDTD1D(m)

    A = sp.buildEvolutionOperator()
    eigA, _ = np.linalg.eig(A)

    A_by_elem = sp.reorder_array(A, 'byElements')
    eigA_by_elem, _ = np.linalg.eig(A_by_elem)

    assert A.shape == A_by_elem.shape
    assert np.allclose(np.real(eigA_by_elem), 0)
    assert np.allclose(np.sort(np.imag(eigA)),
                       np.sort(np.imag(eigA_by_elem)))