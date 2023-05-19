from pytest import approx
import numpy as np
import matplotlib.pyplot as plt

import dgtd.maxwell1d as dg1d
from dgtd.mesh1d import Mesh1D
import dgtd.maxwell2d as dg2d
from dgtd.mesh2d import Mesh2D, readFromGambitFile

from nodepy import runge_kutta_method as rk

TEST_DATA_FOLDER = 'dgtd/testData/'

N = 1000
x = np.linspace(-6, 0.5, N)
y = np.linspace(-5.0, 5.0, N)
X, Y = np.meshgrid(x, y)
z = X + 1j*Y

def test_centered_RK44():

    msh = Mesh1D(0, 20, 30, boundary_label='PEC')
    A_cen = dg1d.Maxwell1D(n_order=3, mesh=msh, fluxType='Centered').buildEvolutionOperator()
    A_cen_eig, _ = np.linalg.eig(A_cen)

    # plt.figure(dpi=200)
    # levels = np.linspace(0, 5, 100)
    # c1 = plt.contour(X,Y, np.abs(f_rk44), levels=[1.0], colors='green')
    # c1_max = np.abs(f_rk44).max()
    # h1,l1 = c1.legend_elements()
    # plt.legend(h1, 'RK44', loc='lower left')
    # plt.scatter(np.real(A_cen_eig), np.imag(A_cen_eig))
    # plt.axhline(y = c1_max, color = 'g', linestyle = '-')

    # print('Center y-Imag[',np.imag(A_cen_eig).min(), ' ', np.imag(A_cen_eig).max(),' ]')
    # plt.show()   
    
    assert   np.abs(np.imag(A_cen_eig).max())<1.95

def test_centered_2D_RK44():
    p,q = rk.loadRKM('RK44').stability_function(mode='float')
    f_rk44 = p(z)/q(z)
    
    msh = readFromGambitFile(TEST_DATA_FOLDER + 'Maxwell2Triang.neu')
    A_cen = dg2d.Maxwell2D(n_order=3, mesh=msh, fluxType='Centered').buildEvolutionOperator()
    A_cen_eig, _ = np.linalg.eig(A_cen)

    dt = 0.05
    A_cen_eig *= dt

    plt.figure(dpi=200)
    levels = np.linspace(0, 5, 100)
    c1 = plt.contour(X,Y, np.abs(f_rk44), levels=[1.0], colors='green')
    c1_max = np.abs(f_rk44).max()
    h1,l1 = c1.legend_elements()
    plt.legend(h1, 'RK44', loc='lower left')
    plt.scatter(np.real(A_cen_eig), np.imag(A_cen_eig))
    plt.axhline(y = c1_max, color = 'g', linestyle = '-')

    print('Center y-Imag[',np.imag(A_cen_eig).min(), ' ', np.imag(A_cen_eig).max(),' ]')
    plt.show()   
    
    assert   np.abs(np.imag(A_cen_eig).max())<1.95
