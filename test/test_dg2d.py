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
            [[-((np.sqrt(3.0)+1.0)/3.0), (1.0-3.0*np.sqrt(3.0))/6.0, (-2.0*np.sqrt(3.0)+2.0)/3.0],
             [(4.0*np.sqrt(3.0)-2.0)/6.0, (6.0*np.sqrt(3.0)-2.0)/6.0, (8.0*np.sqrt(3.0)-2.0)/6.0]]
        ),
        dg.xy_to_rs(x, y)
    )


def test_warp_N1():
    L1 = np.array([0, 0, 1])
    L2 = np.array([1, 0, 0])
    L3 = np.array([0, 1, 0])

    N = 1
    Np = 3
    assert np.allclose(np.zeros(Np), dg.warpFactor(N, L3-L2))
    assert np.allclose(np.zeros(Np), dg.warpFactor(N, L1-L3))
    assert np.allclose(np.zeros(Np), dg.warpFactor(N, L2-L1))


def test_simplex_polynomial():
    a, b = (
        np.array([-1,  0,  1, -1,  1, -1]),
        np.array([-1, -1, -1,  0,  0,  1])
    )
    p11 = dg.simplex_polynomial(a, b, 1, 1)
    p11Ref = np.array(
        [2.1213, 0.0000, -2.1213, -1.5910, 1.5910, 0.0000]
    )

    assert np.allclose(p11, p11Ref, rtol=1e-3)


def test_rs_to_ab():
    r, s = (
        np.array([-1,  0,  1, -1, 0, -1]),
        np.array([-1, -1, -1,  0, 0,  1])
    )

    a, b = dg.rs_to_ab(r, s)

    aRef, bRef = (
        np.array([-1,  0,  1, -1,  1, -1]),
        np.array([-1, -1, -1,  0,  0,  1])
    )

    assert (np.all(a == aRef))
    assert (np.all(b == bRef))


def test_vandermonde_N2():
    # For N = 2.
    r, s = (
        np.array([-1,  0,  1, -1, 0, -1]),
        np.array([-1, -1, -1,  0, 0,  1])
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

    assert np.allclose(x[:, 0], xRef, rtol=1e-3)
    assert np.allclose(y[:, 0], yRef, rtol=1e-3)


def test_lift_N1():
    lift = dg.lift(1)

    liftRef = np.array(
        [[ 2.5000,  0.5000,  -1.5000,  -1.5000,   2.5000,   0.5000],
         [ 0.5000,  2.5000,   2.5000,   0.5000,  -1.5000,  -1.5000],
         [-1.5000, -1.5000,   0.5000,   2.5000,   0.5000,   2.5000]]
    )
    assert np.allclose(lift, liftRef, rtol=1e-3)


def test_gradSimplexP():

    a, b = (
        np.array([-1., 0., 1., -1., 1., -1.]),
        np.array([-1., -1., -1., 0., 0., 1.])
    )

    dmodedr, dmodeds = dg.gradSimplexP(a, b, 1, 1)

    dmodedrRef = np.array(
        [-2.1213, -2.1213, -2.1213, 3.1820, 3.1820, 8.4853]
    )

    dmodedsRef = np.array(
        [-6.3640, -1.0607, 4.2426, -1.0607, 4.2426, 4.2426]
    )

    assert np.allclose(dmodedr, dmodedrRef, rtol=1e-3)
    assert np.allclose(dmodeds, dmodedsRef, rtol=1e-3)

def test_gradVandermonde_N1():

    r, s = (
        np.array([-1., 1., -1.]),
        np.array([-1., -1., 1.])
    )

    V2Dr, V2Ds = dg.gradVandermonde(1, r, s)

    V2DrExp = np.array([
        [0., 0., 1.7321],
        [0., 0., 1.7321],
        [0., 0., 1.7321]]
    )

    V2DsExp = np.array([
        [0., 1.5000, 0.8660],
        [0., 1.5000, 0.8660],
        [0., 1.5000, 0.8660]]
    )

    assert np.allclose(V2Dr, V2DrExp, rtol=1e-3)
    assert np.allclose(V2Ds, V2DsExp, rtol=1e-3)

def test_gradVandermonde_N2():

    r, s = (
        np.array([-1., 0., 1., -1., 0., -1.]),
        np.array([-1., -1., -1., 0., 0., 1.])
    )

    V2Dr, V2Ds = dg.gradVandermonde(2, r, s)

    V2DrExp = np.array([
        [0.0, 0.0, 0.0, 1.7321, -2.1213, -8.2158],
        [0.0, 0.0, 0.0, 1.7321, -2.1213,    0.0],
        [0.0, 0.0, 0.0, 1.7321, -2.1213, 8.2158],
        [0.0, 0.0, 0.0, 1.7321, 3.1820, -4.1079],
        [0.0, 0.0, 0.0, 1.7321, 3.1820, 4.1079],
        [0.0, 0.0, 0.0, 1.7321, 8.4853,    0.0]]
    )

    V2DsExp = np.array([
        [0.0, 1.5000, -4.8990, 0.8660, -6.3640, -2.7386],
        [0.0, 1.5000, -4.8990, 0.8660, -1.0607, 1.3693],
        [0.0, 1.5000, -4.8990, 0.8660, 4.2426, 5.4772],
        [0.0, 1.5000, 1.2247, 0.8660, -1.0607, -1.3693],
        [0.0, 1.5000, 1.2247, 0.8660, 4.2426, 2.7386],
        [0.0, 1.5000, 7.3485, 0.8660, 4.2426,    0.0]]
    )

    assert np.allclose(V2Dr, V2DrExp, rtol=1e-3)
    assert np.allclose(V2Ds, V2DsExp, rtol=1e-3)

def test_derivative_N1():

    r, s = (
        np.array([-1., 1., -1.]),
        np.array([-1., -1., 1.])
    )

    V = dg.vandermonde(1, r, s)

    [Dr, Ds] = dg.derivateMatrix(1, r, s, V)

    DrExp = np.array([
        [-5.0e-01, 5.0e-01,   0.],
        [-5.0e-01, 5.0e-01,   0.],
        [-5.0e-01, 5.0e-01,   0.]]
    )

    DsExp = np.array([ 
        [-0.5, 0.0, 0.5],
        [-0.5, 0.0, 0.5],
        [-0.5, 0.0, 0.5]]
    )

def test_derivative_N2():

    r, s = (
        np.array([-1., 0., 1., -1., 0., -1.]),
        np.array([-1., -1., -1., 0., 0., 1.])
    )

    V = dg.vandermonde(2, r, s)

    [Dr, Ds] = dg.derivateMatrix(2, r, s, V)

    DrExp = np.array([
        [-1.5000, 2.0000, -0.5000, -0.0000, 0.0000, 0.0000],
        [-0.5000, 0.0000,  0.5000, -0.0000, 0.0000, 0.0000],
        [ 0.5000, 2.0000,  1.5000, -0.0000, 0.0000, 0.0000],
        [-0.5000, 1.0000, -0.5000, -1.0000, 1.0000, 0.0000],
        [ 0.5000, 1.0000,  0.5000, -1.0000, 1.0000, 0.0000],
        [ 0.5000, 0.0000, -0.5000, -2.0000, 2.0000, 0.0000]]
    )

    DsExp = np.array([ 
        [-1.5000,  0.0000,  0.0000,  2.0000, -0.0000, -0.5000],
        [-0.5000, -1.0000,  0.0000,  1.0000,  1.0000, -0.5000],
        [ 0.5000, -2.0000, -0.0000, -0.0000,  2.0000, -0.5000],
        [-0.5000, -0.0000,  0.0000, -0.0000, -0.0000,  0.5000],
        [ 0.5000, -1.0000,  0.0000, -1.0000,  1.0000,  0.5000],
        [ 0.5000, -0.0000,  0.0000, -2.0000, -0.0000,  1.5000]]
    )


def test_geometric_factors():
    N = 1
    x = np.array([[-1.000], [-1.000], [-0.1640]])
    y = np.array([[ 0.000], [-1.000], [-0.1640]])
    r, s = dg.xy_to_rs(*dg.set_nodes(N))
    Dr, Ds = dg.derivateMatrix(N, r, s, dg.vandermonde(N, r, s))

    rx, sx, ry, sy, J = dg.geometricFactors(x, y, Dr, Ds)

    assert np.allclose(rx, np.array([[-0.3923], [-0.3923], [-0.3923]]), rtol=1e-3)
    assert np.allclose(sx, np.array([[ 2.3923], [ 2.3923], [ 2.3923]]), rtol=1e-3)
    assert np.allclose(ry, np.array([[-2.0000], [-2.0000], [-2.0000]]), rtol=1e-3)
    assert np.allclose(sy, np.array([[ 0.0000], [ 0.0000], [ 0.0000]]), rtol=1e-3)
    assert np.allclose( J, np.array([[ 0.2090], [ 0.2090], [ 0.2090]]), rtol=1e-3)
    

def test_normals_146_element():   
    # With K=146 and N=2, we have size=(9,146) normals' array, we will considere the first and the end column
    assert np.allclose(np.array([-1.0, -1.0, -1.0, 0.7088, 0.7088, 0.7088, 0.3921, 0.3921, 0.3921]),
                       dg.normals2D(146)[:,0]) 
    assert np.allclose(np.array([-0.9347, -0.9347, -0.9347, 0.4420, 0.4420, 0.4420, 0.7578, 0.7578, 0.7578]),
                       dg.normals2D(146)[:,145])
