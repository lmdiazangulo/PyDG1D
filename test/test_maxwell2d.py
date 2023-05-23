
from dgtd.maxwell2d import *
from dgtd.mesh2d import *

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_build_maps_sizes():
    sp = Maxwell2D(
        1, 
        readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K8.neu')
    )

    assert sp.vmapM.size == 48
    assert sp.vmapP.size == 48
    assert sp.vmapB.size == 16
    assert sp.mapB.size == 16

def test_build_maps_values_N2():
    sp = Maxwell2D(
        2, 
        readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K8.neu')
    )

    vmapPRef = np.array([
        1        ,2        ,3        ,9        ,8        ,7        ,33        ,35        ,36        ,6        ,5        ,3        ,9        ,11        ,12        ,48
        ,46        ,43        ,13        ,14        ,15        ,15        ,17        ,18        ,31        ,32        ,33        ,19        ,20        ,21        ,39
        ,38        ,37        ,19        ,22        ,24        ,25        ,26        ,27        ,45        ,44        ,43        ,25        ,28        ,30
        ,13        ,16        ,18        ,1        ,4        ,6        ,39        ,41        ,42        ,24        ,23        ,21        ,31        ,34        ,36
        ,45        ,47        ,48        ,30        ,29        ,27        ,37        ,40        ,42        ,12        ,10        ,7    ])
    vmapBRef = np.array([
    1
    ,2
    ,3
    ,9
    ,11
    ,12
    ,13
    ,14
    ,15
    ,15
    ,17
    ,18
    ,19
    ,20
    ,21
    ,19
    ,22
    ,24
    ,25
    ,26
    ,27
    ,25
    ,28
    ,30
    ])
    vmapMRef = np.array([
        1
        ,2
        ,3
        ,3
        ,5
        ,6
        ,1
        ,4
        ,6
        ,7
        ,8
        ,9
        ,9
        ,11
        ,12
        ,7
        ,10
        ,12
        ,13
        ,14
        ,15
        ,15
        ,17
        ,18
        ,13
        ,16
        ,18
        ,19
        ,20
        ,21
        ,21
        ,23
        ,24
        ,19
        ,22
        ,24
        ,25
        ,26
        ,27
        ,27
        ,29
        ,30
        ,25
        ,28
        ,30
        ,31
        ,32
        ,33
        ,33
        ,35
        ,36
        ,31
        ,34
        ,36
        ,37
        ,38
        ,39
        ,39
        ,41
        ,42
        ,37
        ,40
        ,42
        ,43
        ,44
        ,45
        ,45
        ,47
        ,48
        ,43
        ,46
        ,48   
    ])

    vmapPRef -= 1
    vmapBRef -= 1
    vmapMRef -= 1

    assert np.allclose(vmapPRef,sp.vmapP)
    assert np.allclose(vmapMRef,sp.vmapM)
    assert np.allclose(vmapBRef,sp.vmapB)

def test_build_maps_values_N3():
    sp = Maxwell2D(
        3, 
        readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K8.neu')
    )

    vmapPRef = np.array([
    1
    ,2
    ,3
    ,4
    ,14
    ,13
    ,12
    ,11
    ,54
    ,57
    ,59
    ,60
    ,10
    ,9
    ,7
    ,4
    ,14
    ,17
    ,19
    ,20
    ,80
    ,78
    ,75
    ,71
    ,21
    ,22
    ,23
    ,24
    ,24
    ,27
    ,29
    ,30
    ,51
    ,52
    ,53
    ,54
    ,31
    ,32
    ,33
    ,34
    ,64
    ,63
    ,62
    ,61
    ,31
    ,35
    ,38
    ,40
    ,41
    ,42
    ,43
    ,44
    ,74
    ,73
    ,72
    ,71
    ,41
    ,45
    ,48
    ,50
    ,21
    ,25
    ,28
    ,30
    ,1
    ,5
    ,8
    ,10
    ,64
    ,67
    ,69
    ,70
    ,40
    ,39
    ,37
    ,34
    ,51
    ,55
    ,58
    ,60
    ,74
    ,77
    ,79
    ,80
    ,50
    ,49
    ,47
    ,44
    ,61
    ,65
    ,68
    ,70
    ,20
    ,18
    ,15
    ,11    
    ])
    vmapMRef = np.array([
        1
        ,2
        ,3
        ,4
        ,4
        ,7
        ,9
        ,10
        ,1
        ,5
        ,8
        ,10
        ,11
        ,12
        ,13
        ,14
        ,14
        ,17
        ,19
        ,20
        ,11
        ,15
        ,18
        ,20
        ,21
        ,22
        ,23
        ,24
        ,24
        ,27
        ,29
        ,30
        ,21
        ,25
        ,28
        ,30
        ,31
        ,32
        ,33
        ,34
        ,34
        ,37
        ,39
        ,40
        ,31
        ,35
        ,38
        ,40
        ,41
        ,42
        ,43
        ,44
        ,44
        ,47
        ,49
        ,50
        ,41
        ,45
        ,48
        ,50
        ,51
        ,52
        ,53
        ,54
        ,54
        ,57
        ,59
        ,60
        ,51
        ,55
        ,58
        ,60
        ,61
        ,62
        ,63
        ,64
        ,64
        ,67
        ,69
        ,70
        ,61
        ,65
        ,68
        ,70
        ,71
        ,72
        ,73
        ,74
        ,74
        ,77
        ,79
        ,80
        ,71
        ,75
        ,78
        ,80
    ])
    vmapBRef = np.array([
    1
    ,2
    ,3
    ,4
    ,14
    ,17
    ,19
    ,20
    ,21
    ,22
    ,23
    ,24
    ,24
    ,27
    ,29
    ,30
    ,31
    ,32
    ,33
    ,34
    ,31
    ,35
    ,38
    ,40
    ,41
    ,42
    ,43
    ,44
    ,41
    ,45
    ,48
    ,50
    ])
    
    vmapPRef -= 1
    vmapBRef -= 1
    vmapMRef -= 1
   
    assert np.allclose(vmapPRef,sp.vmapP)
    assert np.allclose(vmapMRef,sp.vmapM)
    assert np.allclose(vmapBRef,sp.vmapB)

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
    # fig = plt.figure()
    sp.plot_field(1, uin)
    # plt.show()