import numpy as np
import matplotlib.pyplot as plt

from dgtd.maxwell1d import *
from dgtd.meshUtils import *

def test_empty_mesh():
    # Polynomial order for aproximation
    N = 3
    
    [n_v, vx, k_elem, etov] = mesh_generator(-2.0, 2.0, 0.8)
    
    # Set up material parameters
    eps1    = np.ones(int(k_elem/2))
    mu1     = np.ones(int(k_elem)) 
    epsilon = np.ones(n_p)*eps1
    mu      = np.ones(n_p)*mu1
    
    # Set initial condition
    E_old = np.multiply(math.sin(np.pi*x) , x)
    H_old = np.zeros((n_p, k_elem))
    
    # Solve problem
    final_time = 10
    [E, H] = maxwell1D(E_old, H_old, epsilon, mu, final_time)