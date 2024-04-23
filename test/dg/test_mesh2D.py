import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import maxwell.dg.mesh2d as ms
import maxwell.dg.dg2d_tools as dg2d_tools

TEST_DATA_FOLDER = 'testData/'

def test_read_mesh_K8():
    msh = ms.readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K8.neu')

    assert msh.number_of_elements() == 8
    assert msh.number_of_vertices() == 9


def test_read_mesh_K146():
    msh = ms.readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')

    assert msh.number_of_elements() == 146
    assert msh.number_of_vertices() == 90


def test_connectivityMatrices():
    msh = ms.readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K8.neu')

    EToE, EToF = msh.connectivityMatrices()

    assert np.all(np.array([0, 1, 5], dtype=int) == EToE[0, :])
    assert np.all(np.array([0, 0, 1], dtype=int) == EToF[0, :])
    
def test_plot_mesh():
    msh = ms.readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')
    tri = msh.getTriangulation()
    # plt.triplot(tri, c='k', lw=1.0)
    # plt.gca().set_aspect('equal')
    # plt.show()
    assert True