
from fdtd.fdtd1d import *
from dgtd.mesh1d import *

import matplotlib.pyplot as plt


def test_fdtd1d_ctor():
    K = 10
    m = Mesh1D(0, 1, K)
    sp = FDTD1D(m)

    assert len(sp.x) == K + 1
    assert len(sp.xH) == K

    assert len(sp.xH) == m.number_of_elements()


def test_fdtd1d_buildFields():
    K = 20
    sp = FDTD1D(Mesh1D(0, 5, K))

    fields = sp.buildFields()

    assert len(fields['E']) == K + 1
    assert len(fields['H']) == K