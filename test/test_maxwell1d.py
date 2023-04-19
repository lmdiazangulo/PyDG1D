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


def test_pec_box():
    mesh = Mesh1D(-1.0, 1.0, 10)
    
    n_order = 2
    final_time = 21.0
    sp = SpatialDiscretization(n_order, mesh)
    
    driver = MaxwellDriver(sp)
    x0 = 0.0
    s0 = 0.25
    driver.E = np.exp(-(sp.x-x0)**2/(2*s0**2))
    
    timeRange = np.arange(0.0, final_time, driver.dt)
    for t in timeRange:
        driver.step()

        # plt.plot(sp.x, driver.E, '.-b')
        # plt.plot(sp.x, driver.H, '.-r')
        # plt.grid()
        # plt.title('time=%f' %t)
        # plt.ylim(-1.1, 1.1)
        # plt.pause(0.0001)
        # plt.cla()

    # driver.step(driver.dt/2)
    plt.plot(sp.x, driver.E, '.-b')
    plt.plot(sp.x, driver.H, '.-r')
    plt.grid()
    
    plt.title('time=%f' %driver.time)
    plt.ylim(-1.1, 1.1)
    plt.show()
    dsad

    



    
def test_maxwell1d_mesh_graph_initial_condition():
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
    final_time = 1
    [E, H] = mw.maxwell1D(E_old, H_old, epsilon, mu, final_time, sp)
    

def test_maxwell1d_mesh_graph_final_condition():
    # Polynomial order for aproximation
    fig, ax = plt.subplots()
    
    n_order = 3
    
    mesh = Mesh1D(-1.0, 1.0,5)
    sp = SpatialDiscretization(n_order, mesh)
    
    epsilon = np.ones([n_order+1, mesh.number_of_elements()])
    mu      = np.ones([n_order+1, mesh.number_of_elements()])
    
    x       = set_nodes_1d(n_order, mesh.vx[mesh.EToV])
    
    
    # Set initial condition
    #E_old = np.sin(np.pi*x)*(x<0)
    E_old = (1.0/2.0*np.sqrt(2*np.pi))*np.exp(-40.0*np.power(2.0*sp.x-1.0,2)/4.0)
    H_old = np.zeros([n_order+1, mesh.number_of_elements()]) 
    
    
    
    # Solve problem
    final_time = 1
    [E, H] = mw.maxwell1D(E_old, H_old, epsilon, mu, final_time, sp)
    
    x_list = np.zeros([x.size])
    H_list = np.zeros([H_old.size])
    x_reshaped = x.reshape((x_list.shape))
    H_reshaped = H.reshape((H_list.shape))
    line,   = ax.plot(x_reshaped, H_reshaped)
    
    plt.figure()
    plt.title("Final Magnetic Field")

    def animate(i):
        line.set_ydata(H_reshaped + i / 50)  # update the data.
        return line,

  #  line, _ = ax.plot(x, H)
  #  line = plt.plot(x,H)
    
    ani = animation.FuncAnimation(
    fig, animate, interval=100, blit=True, save_count=10)

    plt.show()
    
    