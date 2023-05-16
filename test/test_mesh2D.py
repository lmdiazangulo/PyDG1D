import dgtd.mesh2d as ms

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_read_mesh_K8():
    msh = ms.readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K8.neu')

    assert msh.number_of_elements() == 8
    assert msh.number_of_vertices() == 9


def test_read_mesh_K146():
    msh = ms.readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')

    assert msh.number_of_elements() == 146
    assert msh.number_of_vertices() == 90