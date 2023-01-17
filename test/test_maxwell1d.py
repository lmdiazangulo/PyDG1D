import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dgtd.maxwell1d import *
from dgtd.meshUtils import *

import dgtd.meshUtils as ms
import dgtd.maxwell1d as mw


def test_spatial_discretization_lift():
    sp = SpatialDiscretization(1, Mesh1D(0.0, 1.0, 1))
    assert   np.allclose(surface_integral_dg(1, vandermonde_1d(1, jacobiGL(0.0,0.0,1))), 
                         np.array([[2.0,-1.0],[-1.0,2.0]]))


def test_empty_mesh():
    # Polynomial order for aproximation
    n_order = 3
    
    mesh = Mesh1D(-2.0, 2.0, 5)
    sp = SpatialDiscretization(n_order, mesh)
    
    # Set up material parameters
    epsilon = np.ones(mesh.number_of_elements())
    mu      = np.ones(mesh.number_of_elements())
    
    x       = set_nodes_1d(n_order, mesh.vx[mesh.EToV])
    
    # Set initial condition
    E_old = np.zeros([n_order+1, mesh.number_of_elements()]) #math.sin(np.pi*x) np.zeros((N+1, K))
    H_old = np.zeros([n_order+1, mesh.number_of_elements()]) #np.zeros((sp.number_of_nodes_per_element(), mesh.number_of_elements()))
    
    # Solve problem
    final_time = 10
    [E, H] = mw.maxwell1D(E_old, H_old, epsilon, mu, final_time, sp)
 #   assert np.allclose(E, )
    
# def test_maxwell1d_1():
#     assert np.allclose(maxwell1d(E, H, eps, mu, final_time, sp: SpatialDiscretization), )

def test_maxwell1d_mesh_graph_initial_condition():
    # Polynomial order for aproximation
    n_order = 6
    
    mesh = Mesh1D(-1.0, 1.0, 80)
    sp = SpatialDiscretization(n_order, mesh)
    
    epsilon = np.ones([n_order+1, mesh.number_of_elements()])
    mu      = np.ones([n_order+1, mesh.number_of_elements()])
    
    x       = set_nodes_1d(n_order, mesh.vx[mesh.EToV])
    
    # Set initial condition
    E_old = np.multiply(np.sin(np.pi*x),x<0)
    H_old = np.zeros([n_order+1, mesh.number_of_elements()]) 
    
    plt.figure()
    plt.title("Initial Magnetic Field")
    plt.plot(x, H_old)
    
    plt.figure()
    plt.title("Initial Electric Field")
    plt.plot(x, E_old, label='original')
    
    plt.show()
    
    # Solve problem
    final_time = 10
    [E, H] = mw.maxwell1D(E_old, H_old, epsilon, mu, final_time, sp)
    

def test_maxwell1d_mesh_graph_final_condition():
    # Polynomial order for aproximation
    fig, ax = plt.subplots()
    
    n_order = 6
    
    mesh = Mesh1D(-1.0, 1.0, 80)
    sp = SpatialDiscretization(n_order, mesh)
    
    epsilon = np.ones([n_order+1, mesh.number_of_elements()])
    mu      = np.ones([n_order+1, mesh.number_of_elements()])
    
    x       = set_nodes_1d(n_order, mesh.vx[mesh.EToV])
    
    
    # Set initial condition
    E_old = np.multiply(np.sin(np.pi*x),x<0)
    H_old = np.zeros([n_order+1, mesh.number_of_elements()]) 
    
    # Solve problem
    final_time = 10
    [E, H] = mw.maxwell1D(E_old, H_old, epsilon, mu, final_time, sp)
    
    x_list = np.zeros([x.size])
    H_list = np.zeros([E.size])
    x_reshaped = x_list.reshape((x_list.shape))
    H_reshaped = H_list.reshape((H_list.shape))
    line,   = ax.plot(x_reshaped, H_reshaped)
    
    plt.figure()
    plt.title("Final Magnetic Field")
  #  line, _ = ax.plot(x, H)
  #  line = plt.plot(x,H)
    
    def animate(i):
        line.set_ydata(H_reshaped + i / 50)  # update the data.
        return line,


    ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True, save_count=50)

    plt.show()
    
    