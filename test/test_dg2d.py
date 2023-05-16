from pytest import approx
import numpy as np

import dgtd.dg2d as dg
import dgtd.mesh2d as mesh

TEST_DATA_FOLDER = 'dgtd/testData/'

def test_set_nodes_N1():
    x, y = dg.set_nodes(1)
    assert np.allclose(np.array([-1.0, 1.0, 0.0]), x, rtol=1e-3)
    assert np.allclose(
        np.array([-1/np.sqrt(3.0), -1/np.sqrt(3.0),  2/np.sqrt(3.0)]), y, rtol=1e-3)

def test_set_nodes_N2():
    x, y = dg.set_nodes(2)
    assert np.allclose(
        np.array([-1.0, 0.0, 1.0, -0.5, 0.5,  0.0]), x, rtol=1e-3)
    assert np.allclose(
        np.array(
            [-1/np.sqrt(3.0), -1/np.sqrt(3.0), -1/np.sqrt(3.0),
              1/2/np.sqrt(3.0), 1/2/np.sqrt(3.0), 2/np.sqrt(3.0)]), y, rtol=1e-3)
    
def test_xy_to_rs():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([1.0, 1.5, 2.0])
    assert np.allclose(
        np.array(
        [[-((np.sqrt(3.0)+1.0)/3.0),(1.0-3.0*np.sqrt(3.0))/6.0,(-2.0*np.sqrt(3.0)+2.0)/3.0],
        [(4.0*np.sqrt(3.0)-2.0)/6.0,(6.0*np.sqrt(3.0)-2.0)/6.0,(8.0*np.sqrt(3.0)-2.0)/6.0]]
        ),
        dg.xy_to_rs(x, y)
    )


def test_warp_N1():
    L1 = np.array([0,0,1])
    L2 = np.array([1,0,0])
    L3 = np.array([0,1,0])

    N = 1 
    Np = 3
    assert np.allclose(np.zeros(Np), dg.warpFactor(N, L3-L2))
    assert np.allclose(np.zeros(Np), dg.warpFactor(N, L1-L3))
    assert np.allclose(np.zeros(Np), dg.warpFactor(N, L2-L1))


def test_simplex_polynomial():
    a, b = (
        np.array([-1,  0,  1, -1,  1, -1 ]),
        np.array([-1, -1, -1,  0,  0,  1 ])
    )
    p11 = dg.simplex_polynomial(a, b, 1, 1)
    p11Ref = np.array(
        [2.1213, 0.0000, -2.1213, -1.5910, 1.5910, 0.0000]
    )

    assert np.allclose(p11, p11Ref, rtol=1e-3)


def test_rs_to_ab():
    r, s = ( 
        np.array([-1,  0,  1, -1, 0, -1 ]),
        np.array([-1, -1, -1,  0, 0,  1 ])
    )

    a, b = dg.rs_to_ab(r, s)

    aRef, bRef = (
        np.array([-1,  0,  1, -1,  1, -1 ]),
        np.array([-1, -1, -1,  0,  0,  1 ])
    )

    assert(np.all(a == aRef))
    assert(np.all(b == bRef))

def test_vandermonde_N2():
    # For N = 2.
    r, s = ( 
        np.array([-1,  0,  1, -1, 0, -1 ]),
        np.array([-1, -1, -1,  0, 0,  1 ])
    )

    V = dg.vandermonde(2, r, s)

    VRef = np.array(
        [[0.7071, -1.0000,  1.2247, -1.7321,  2.1213,  2.7386],
         [0.7071, -1.0000,  1.2247,  0.0000,  0.0000, -1.3693],
         [0.7071, -1.0000,  1.2247,  1.7321, -2.1213,  2.7386],
         [0.7071,  0.5000, -0.6124, -0.8660, -1.5910,  0.6847],
         [0.7071,  0.5000, -0.6124,  0.8660,  1.5910,  0.6847],
         [0.7071,  2.0000,  3.6742,  0.0000,  0.0000,  0.0000]]
    )

    assert np.allclose(V, VRef, rtol=1e-3)

def test_nodes_coordinates():
    m = mesh.readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2D_K146.neu')

    x, y = dg.nodes_coordinates(2, m)

    assert x.shape == y.shape
    assert x.shape == (6, 146)

    xRef = np.array([-1.0000, -1.0000, -1.0000, -0.9127, -0.9127, -0.8253])
    yRef = np.array([-0.7500, -0.8750, -1.0000, -0.7872, -0.9122, -0.8245])

    assert np.allclose(x[:,0], xRef, rtol=1e-3)
    assert np.allclose(y[:,0], yRef, rtol=1e-3)

def test_lift_N1():
    lift = dg.lift(1)

    liftRef = np.array(
        [[ 2.5000,  0.5000,  -1.5000,  -1.5000,   2.5000,   0.5000],
         [ 0.5000,  2.5000,   2.5000,   0.5000,  -1.5000,  -1.5000],
         [-1.5000, -1.5000,   0.5000,   2.5000,   0.5000,   2.5000]]
    )
    assert np.allclose(lift, liftRef, rtol=1e-3)

    
def test_gradsimplex_2DP():
        
    a, b = (
    np.array([-1,  0,  1, -1,  1, -1 ]),
    np.array([-1, -1, -1,  0,  0,  1 ])
    )

    dmodedr, dmodeds = dg.GradSimplex2DP(a, b, 1, 1)

    dmodedrRef = np.array(
        [[-2.1213, -2.1213, -2.1213, -2.1213, -2.1213, -2.1213],
         [-2.1213, -2.1213, -2.1213, -2.1213, -2.1213, -2.1213],
         [-2.1213, -2.1213, -2.1213, -2.1213, -2.1213, -2.1213],
         [ 3.1820,  3.1820,  3.1820,  3.1820,  3.1820,  3.1820],
         [ 3.1820,  3.1820,  3.1820,  3.1820,  3.1820,  3.1820],
         [ 8.4853,  8.4853,  8.4853,  8.4853,  8.4853,  8.4853]]
    )

    dmodedsRef = np.array(
        [[-6.3640, -6.3640, -6.3640, -3.7123, -3.7123, -1.0607],
         [-1.0607, -1.0607, -1.0607, -1.0607, -1.0607, -1.0607],
         [ 4.2426,  4.2426,  4.2426,  1.5910,  1.5910, -1.0607],
         [-3.7123, -3.7123, -3.7123, -1.0607, -1.0607,  1.5910],
         [ 6.8943,  6.8943,  6.8943,  4.2426,  4.2426,  1.5910],
         [-1.0607, -1.0607, -1.0607,  1.5910,  1.5910,  4.2426]]
    )

    assert np.allclose(dmodedr, dmodedrRef, rtol=1e-3)
    assert np.allclose(dmodeds, dmodedsRef, rtol=1e-3)

def test_gradvandermonde_2D_N1():

    r, s = ( 
        np.array([-1,  0,  1, -1, 0, -1 ]),
        np.array([-1, -1, -1,  0, 0,  1 ])
    )

    V2DrExp = np.array([
        [0.0, 0.0, 1.7321],
        [0.0, 0.0, 1.7321],
        [0.0, 0.0, 1.7321],
        [0.0, 0.0, 1.7321],
        [0.0, 0.0, 1.7321],
        [0.0, 0.0, 1.7321]]
        )
    
    V2DsExp = np.array([
        [0.0, 1.5000, 0.8660],
        [0.0, 1.5000, 0.8660],
        [0.0, 1.5000, 0.8660],
        [0.0, 1.5000, 0.8660],
        [0.0, 1.5000, 0.8660],
        [0.0, 1.5000, 0.8660]]
        )
    
    V2Dr, V2Ds = dg.GradVandermonde2D(2,r,s)

    assert np.allclose(V2Dr,V2DrExp, rtol=1e-3)
    assert np.allclose(V2Ds,V2DsExp, rtol=1e-3)
    

