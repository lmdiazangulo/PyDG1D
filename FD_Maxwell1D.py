import numpy as np
import matplotlib.pyplot as plt
from math import exp

# # Physics parameters
# epsilon0 = 8.854e-12  # electic permitivity in the vacum
# mu0 = 4*np.pi*1e-7    # magnetic permiability in the vacum
# c = 1/np.sqrt(epsilon0*mu0)  # light velocity in the vacum
# spread = 12

# # Numerical parameters
# L       = 1.0   # Domain longe
# N       = 20   # number of nodes
# dx      = L/N  # size of spatial step in the mesh
# dt      = dx/(2*c)   # time step
# t_max   = 10*dx/c  # max time in the simulation
# t_0    = 40
# steps   = 100

# # Crear la malla
# x = np.linspace(0, L, N+1)

# initial conditions
ke = 200
Ex = np.zeros(ke)
Hy = np.zeros(ke)

#Pulse parameters
kc = int(ke/2)
t_0 = 40
spread = 12
steps = 50

for time_step in range (1, steps + 1):
        
    #Calculate Ex field
    for k in range (1, ke):
        Ex[k] = Ex[k] + 0.5*(Hy[k-1]-Hy[k])      
    # Put a Gaussian pulse in middle
    pulse       = exp(-0.5 * ((t_0 - time_step) / spread) ** 2)
    Ex[kc-20]   = pulse
    Ex[kc+20]   = pulse
        
    #Calculate Hy Field
    for k in range (ke-1):
        Hy[k] = Hy[k] + 0.5*(Ex[k] - Ex[k+1])


## Plot the output:
plt.rcParams['font.size'] = 12
plt.figure(figsize=(8, 3.5))

plt.subplot(211)
plt.plot(Ex, color='k', linewidth=1)
plt.ylabel('E$_x$', fontsize='14')
plt.xticks(np.arange(0, 201, step=20))
plt.xlim(0, 200)
plt.yticks(np.arange(-1, 1.2, step=1))
plt.ylim(-1.2, 1.2)
plt.text(100, 0.5, 'T = {}'.format(time_step),
horizontalalignment='center')
plt.subplot(212)
plt.plot(Hy, color='k', linewidth=1)
plt.ylabel('H$_y$', fontsize='14')
plt.xlabel('FDTD cells')
plt.xticks(np.arange(0, 201, step=20))
plt.xlim(0, 200)
plt.yticks(np.arange(-1, 1.2, step=1))
plt.ylim(-1.2, 1.2)
plt.subplots_adjust(bottom=0.2, hspace=0.45)

plt.show()