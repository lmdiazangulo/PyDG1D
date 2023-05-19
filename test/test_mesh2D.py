import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import dgtd.mesh2d as ms
import dgtd.dg2d as dg2d

TEST_DATA_FOLDER = 'dgtd/testData/'

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
    # plt.triplot(msh.getTriangulation(), c='k', lw=1.0)
    # plt.gca().set_aspect('equal')
    # plt.show()
    assert True
    
def test_plotField2D():
    #Entries
    K = 8
    Nout = 1
    xin = np.array([[ -1.0, -0.1640,    0.0,    1.0,    1.0,      0.0,    1.0,   0.0],
                    [-1.0,     -1.0,   -1.0,    0.0,    1.0,     -1.0,    0.0,   1.0],
                    [-0.164,     0.,   -1.0,    1.0,    0.0,   -0.164, -0.164, -0.164]
                    ])
    yin = np.array([[ .0,    -0.1640,    1.0,    1.0,    -1.0,    1.0,    0.0,    -1.0],
                    [-1.0,      -1.0,    1.0,    1.0,    0.0,     0.0,    1.0,      0.0],
                    [-0.164,    -1.0,    0.0,    0.0,    -1.0,   -0.164, -0.164, -0.164]
                    ])
    uin = np.array([[-1.9385e-02,  -1.2718e-01,  -1.4559e-02,   1.4345e-03,   2.5271e-03,  -1.7483e-02,  -1.7840e-02,  -4.3292e-02],
                    [-1.6564e-03,  -1.6564e-03,   2.5271e-03,  -8.2897e-03,  -1.4559e-02,  -4.3292e-02,  -1.7840e-02,  -1.7483e-02],
                    [-1.2718e-01,  -1.9385e-02,  -2.2464e-02,  -8.2897e-03,  -2.2464e-02,  -1.1068e-01,  -9.7667e-02,  -1.1068e-01]
                    ])
    #build equally spaced grid on reference triangle
    Npout = int((Nout+1)*(Nout+2)/2)
    rout = np.zeros((Npout))
    sout = np.zeros((Npout))
    counter = np.zeros((Nout+1, Nout+1))
    sk = 0
    for n in range (Nout+1):
        for m in range (Nout+1-n):
            rout[sk] = -1 + 2*m/Nout
            sout[sk] = -1 + 2*n/Nout
            counter[n,m] = sk
            sk += 1
            
    # build matrix to interpolate field data to equally spaced nodes
    Vout = dg2d.vandermonde(Nout, rout, sout)
    interp = Vout*np.linalg.inv(Vout)

    #build triangulation of equally spaced nodes on reference triangle
    for n in range (Nout+1):
        for m in range (Nout-n):
            v1 = counter[n,m]
            v2 = counter[n,m+1]
            v3 = counter[n+1,m]
            v4 = counter[n+1,m+1]
        if v4:
            tri = np.vstack(([v1, v2, v3],[v2, v4, v3]))
        else:
            tri = np.vstack([[v1, v2, v3]])

    # build triangulation for all equally spaced nodes on all elements
    TRI = np.zeros((K*2,Npout))
    for k in range(K+1):
        TRI[k,:] = np.stack((tri+(k)*Npout))

    TRI_t = TRI[:,:]
    #interpolate node coordinates and field to equally spaced nodes
    xout = interp*xin 
    yout = interp*yin 
    uout = interp*uin

    #render and format solution field
    fig     = plt.figure()
    ax      = fig.add_subplot(111, projection='3d')
    surf    = ax.plot_trisurf(xout.flatten(), yout.flatten(), uout.flatten(), triangles=TRI, cmap='viridis')
    
    # Configurar el sombreado, material y iluminación
    surf.set_facecolor(cm.shiny)
    ax.set_facecolor('white')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Configurar la iluminación
    ax.light_sources = [(1, 1, 1)]
    ax.add_artist(ax.light[0])
    ax.light[0].position = (1, 1, 1)
    ax.light[0].ambient = 1
    ax.light[0].specular = 1
    ax.light[0].diffuse = 1
  
    plt.show()

    return