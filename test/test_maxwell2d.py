
from dgtd.maxwell2d import *
from dgtd.mesh2d import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_build_maps():
    sp = Maxwell2D(
        1, 
        readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K8.neu')
    )

    assert sp.vmapM.size == 48
    assert sp.vmapP.size == 48
    assert sp.vmapB.size == 16
    assert sp.mapB.size == 16

def test_dt_scale():

    sp = Maxwell2D(
        1, 
        readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2Triang.neu')
    )
    tol = 1e-3
    assert (np.min(sp.get_dt_scale()) - 0.2929 <= tol)

   
def test_plot_field():
    sp = Maxwell2D(1, readFromGambitFile(TEST_DATA_FOLDER+'Maxwell2D_K8.neu'))
    uin = np.array([
        [-1.9385e-02,  -1.2718e-01,  -1.4559e-02,   1.4345e-03,   2.5271e-03,  -1.7483e-02,  -1.7840e-02,  -4.3292e-02],
        [-1.6564e-03,  -1.6564e-03,   2.5271e-03,  -8.2897e-03,  -1.4559e-02,  -4.3292e-02,  -1.7840e-02,  -1.7483e-02],
        [-1.2718e-01,  -1.9385e-02,  -2.2464e-02,  -8.2897e-03,  -2.2464e-02,  -1.1068e-01,  -9.7667e-02,  -1.1068e-01]
    ])
    fig = plt.figure()
    sp.plot_field(1, uin, fig)
    plt.show()