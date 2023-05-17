
from dgtd.maxwell2d import *
from dgtd.mesh2d import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_build_maps():
    sp = Maxwell2D(
        1, 
        readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K8.neu')
    )

    assert sp.vmap_m.size == 48
    assert sp.vmap_p.size == 48
    assert sp.vmap_b.size == 16
    assert sp.map_b.size == 16
