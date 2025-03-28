import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytest
import time


from maxwell.driver import *
from maxwell.dg.mesh1d import *
from maxwell.dg.dg1d import *
from maxwell.fd.fd1d import *


from nodepy import runge_kutta_method as rk


def dft(x,freq,time):
    X=[]
    for f in range(len(freq)):
        summatory=0.
        for t in range(len(time)):
            summatory=summatory + x[t] * np.exp(-2j*np.pi*freq[f]*time[t])
        X.append(summatory)
    return X


def test_TFG_ep50_rho1_multislab_1cm():

    #Material distribution
    epsilon_1=1.0
    mu_0 = 4.0*np.pi*1e-7
    eps_0 = 8.854187817e-12
    Z_0 = np.sqrt(mu_0/eps_0)


    epsilon_r_material = 50
    rho_material = 1

    #Mesh
    L1 = 0.0
    L2 = 1.0
    elements=100
    epsilons = epsilon_1*np.ones(elements)
    epsilons[25] = epsilon_r_material
    epsilons[50] = epsilons[25]
    epsilons[75] = epsilons[25]
    sigmas = np.zeros(elements)
    sigmas[25] = Z_0/rho_material
    sigmas[50] = sigmas[25]
    sigmas[75] = sigmas[25]


    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(L1, L2, elements, boundary_label="SMA"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    #Type of wave
    s0 = 0.025
    x0 = 0.1
    final_time = 25.0
    steps = int(np.ceil(final_time/driver.dt))
    freq_vector = np.logspace(8, 9, 301)
    additional_freq_1 = np.linspace(612e6,730e6,40)
    freq_vector = np.union1d(freq_vector, additional_freq_1)
 

    initialFieldE = np.exp(-(sp.x-x0)**2.0/(2.0*s0**2.0))
    initialFieldH = initialFieldE

    #Driver operates
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    E_vector_R = []
    E_vector_T = []
    E_vector_0 = []


    # #DFT calculations
    time_vector_coeffs = np.linspace(0.0, final_time, steps)

    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]

    for _ in range(steps):
        driver.step()
        E_vector_R.append(driver['E'][3][5])
        E_vector_T.append(driver['E'][3][95])

    for t in time_vector_coeffs:
        E_vector_0.append(np.exp(-(t-x0)**2.0/(2.0*s0**2.0)))

    time_vector_coeffs_corrected = time_vector_coeffs / 299792458

    dft_E_R=dft(E_vector_R,freq_vector,time_vector_coeffs_corrected)
    dft_E_T=dft(E_vector_T,freq_vector,time_vector_coeffs_corrected)
    dft_0=dft(E_vector_0,freq_vector,time_vector_coeffs_corrected)

    #Transmission and Reflection coefficients
    T = np.abs(dft_E_T) / np.abs(dft_0)
    R = np.abs(dft_E_R) / np.abs(dft_0)
    dB_T=20*np.log10(T) 
    dB_R=20*np.log10(R) 

    # Save data to .dat file
    with open("transmission_reflection_data1.dat", "w") as f:
        f.write("# Frequency (MHz)    Transmission (dB)    Reflection (dB)\n")
        for freq, t, r in zip(freq_vector / 1e6, dB_T, dB_R):
            f.write(f"{freq:.6e}    {t:.6e}    {r:.6e}\n")

    # T and R graph 
    plt.figure(figsize=(10, 8))
    plt.plot(freq_vector/1e6, dB_T, label='Transmission Coefficient (T)', color='purple')
    plt.plot(freq_vector/1e6, dB_R, label='Reflection Coefficient (R)', color='orange')
    plt.title('Transmission and Reflection Coefficients')
    plt.xlabel('Frequency (MHz)')
    plt.xlim(1e2, 1e3)
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()



def test_TFG_ep112_rho7_multislab_1cm():

    #Material distribution
    epsilon_1=1.0
    mu_0 = 4.0*np.pi*1e-7
    eps_0 = 8.854187817e-12
    Z_0 = np.sqrt(mu_0/eps_0)


    epsilon_r_material = 11.2
    rho_material = 7.0

    #Mesh
    L1 = 0.0
    L2 = 1.0
    elements=100
    epsilons = epsilon_1*np.ones(elements)
    epsilons[25] = epsilon_r_material
    epsilons[50] = epsilons[25]
    epsilons[75] = epsilons[25]
    sigmas = np.zeros(elements)
    sigmas[25] = Z_0/rho_material
    sigmas[50] = sigmas[25]
    sigmas[75] = sigmas[25]


    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(L1, L2, elements, boundary_label="SMA"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    #Type of wave
    s0 = 0.025
    x0 = 0.1
    final_time = 25.0
    steps = int(np.ceil(final_time/driver.dt))
    freq_vector = np.logspace(8, 9, 301)
    additional_freq_1 = np.linspace(686e6,706e6,25)
    additional_freq_2 = np.linspace(817e6,843e6,25)
    freq_vector = np.union1d(freq_vector, additional_freq_1)
    freq_vector = np.union1d(freq_vector, additional_freq_2)
    

    initialFieldE = np.exp(-(sp.x-x0)**2.0/(2.0*s0**2.0))
    initialFieldH = initialFieldE

    #Driver operates
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    E_vector_R = []
    E_vector_T = []
    E_vector_0 = []


    # #DFT calculations
    time_vector_coeffs = np.linspace(0.0, final_time, steps)

    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    
    for _ in range(steps):
        driver.step()
        E_vector_R.append(driver['E'][3][5])
        E_vector_T.append(driver['E'][3][95])

    for t in time_vector_coeffs:
        E_vector_0.append(np.exp(-(t-x0)**2.0/(2.0*s0**2.0)))

    time_vector_coeffs_corrected = time_vector_coeffs / 299792458

    dft_E_R=dft(E_vector_R,freq_vector,time_vector_coeffs_corrected)
    dft_E_T=dft(E_vector_T,freq_vector,time_vector_coeffs_corrected)
    dft_0=dft(E_vector_0,freq_vector,time_vector_coeffs_corrected)

    #Transmission and Reflection coefficients
    T = np.abs(dft_E_T) / np.abs(dft_0)
    R = np.abs(dft_E_R) / np.abs(dft_0)
    dB_T=20*np.log10(T) 
    dB_R=20*np.log10(R) 

    # Save data to .dat file
    with open("transmission_reflection_data2.dat", "w") as f:
        f.write("# Frequency (MHz)    Transmission (dB)    Reflection (dB)\n")
        for freq, t, r in zip(freq_vector / 1e6, dB_T, dB_R):
            f.write(f"{freq:.6e}    {t:.6e}    {r:.6e}\n")

    # T and R graph 
    plt.figure(figsize=(10, 8))
    plt.plot(freq_vector/1e6, dB_T, label='Transmission Coefficient (T)', color='purple')
    plt.plot(freq_vector/1e6, dB_R, label='Reflection Coefficient (R)', color='orange')
    plt.title('Transmission and Reflection Coefficients')
    plt.xlabel('Frequency (MHz)')
    plt.xlim(1e2, 1e3)
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()



def test_TFG_ep20_rho5_multislab_1cm():

    #Material distribution
    epsilon_1=1.0
    mu_0 = 4.0*np.pi*1e-7
    eps_0 = 8.854187817e-12
    Z_0 = np.sqrt(mu_0/eps_0)


    epsilon_r_material = 20.0
    rho_material = 5.0

    #Mesh
    L1 = 0.0
    L2 = 1.0
    elements=100
    epsilons = epsilon_1*np.ones(elements)
    epsilons[25] = epsilon_r_material
    epsilons[50] = epsilons[25]
    epsilons[75] = epsilons[25]
    sigmas = np.zeros(elements)
    sigmas[25] = Z_0/rho_material
    sigmas[50] = sigmas[25]
    sigmas[75] = sigmas[25]


    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(L1, L2, elements, boundary_label="SMA"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    #Type of wave
    s0 = 0.025
    x0 = 0.1
    final_time = 25.0
    steps = int(np.ceil(final_time/driver.dt))
    freq_vector = np.logspace(8, 9, 301)
    additional_freq_1 = np.linspace(650e6,675e6,25)
    additional_freq_2 = np.linspace(755e6,778e6,25)
    freq_vector = np.union1d(freq_vector, additional_freq_1)
    freq_vector = np.union1d(freq_vector, additional_freq_2)
    

    initialFieldE = np.exp(-(sp.x-x0)**2.0/(2.0*s0**2.0))
    initialFieldH = initialFieldE

    #Driver operates
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    E_vector_R = []
    E_vector_T = []
    E_vector_0 = []


    # #DFT calculations
    time_vector_coeffs = np.linspace(0.0, final_time, steps)

    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    
    for _ in range(steps):
        driver.step()
        E_vector_R.append(driver['E'][3][5])
        E_vector_T.append(driver['E'][3][95])

    for t in time_vector_coeffs:
        E_vector_0.append(np.exp(-(t-x0)**2.0/(2.0*s0**2.0)))

    time_vector_coeffs_corrected = time_vector_coeffs / 299792458

    dft_E_R=dft(E_vector_R,freq_vector,time_vector_coeffs_corrected)
    dft_E_T=dft(E_vector_T,freq_vector,time_vector_coeffs_corrected)
    dft_0=dft(E_vector_0,freq_vector,time_vector_coeffs_corrected)

    #Transmission and Reflection coefficients
    T = np.abs(dft_E_T) / np.abs(dft_0)
    R = np.abs(dft_E_R) / np.abs(dft_0)
    dB_T=20*np.log10(T) 
    dB_R=20*np.log10(R) 

    # Save data to .dat file
    with open("transmission_reflection_data3.dat", "w") as f:
        f.write("# Frequency (MHz)    Transmission (dB)    Reflection (dB)\n")
        for freq, t, r in zip(freq_vector / 1e6, dB_T, dB_R):
            f.write(f"{freq:.6e}    {t:.6e}    {r:.6e}\n")

    # T and R graph 
    plt.figure(figsize=(10, 8))
    plt.plot(freq_vector/1e6, dB_T, label='Transmission Coefficient (T)', color='purple')
    plt.plot(freq_vector/1e6, dB_R, label='Reflection Coefficient (R)', color='orange')
    plt.title('Transmission and Reflection Coefficients')
    plt.xlabel('Frequency (MHz)')
    plt.xlim(1e2, 1e3)
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()



def test_TFG_ep5_rho15_multislab_1cm():

    #Material distribution
    epsilon_1=1.0
    mu_0 = 4.0*np.pi*1e-7
    eps_0 = 8.854187817e-12
    Z_0 = np.sqrt(mu_0/eps_0)


    epsilon_r_material = 5.0
    rho_material = 15.0

    #Mesh
    L1 = 0.0
    L2 = 1.0
    elements=100
    epsilons = epsilon_1*np.ones(elements)
    epsilons[25] = epsilon_r_material
    epsilons[50] = epsilons[25]
    epsilons[75] = epsilons[25]
    sigmas = np.zeros(elements)
    sigmas[25] = Z_0/rho_material
    sigmas[50] = sigmas[25]
    sigmas[75] = sigmas[25]


    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(L1, L2, elements, boundary_label="SMA"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    #Type of wave
    s0 = 0.025
    x0 = 0.1
    final_time = 25.0
    steps = int(np.ceil(final_time/driver.dt))
    freq_vector = np.logspace(8, 9, 301)
    additional_freq_1 = np.linspace(733e6,751e6,25)
    additional_freq_2 = np.linspace(908e6,927e6,25)
    freq_vector = np.union1d(freq_vector, additional_freq_1)
    freq_vector = np.union1d(freq_vector, additional_freq_2)
    

    initialFieldE = np.exp(-(sp.x-x0)**2.0/(2.0*s0**2.0))
    initialFieldH = initialFieldE

    #Driver operates
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    E_vector_R = []
    E_vector_T = []
    E_vector_0 = []


    # #DFT calculations
    time_vector_coeffs = np.linspace(0.0, final_time, steps)

    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    
    for _ in range(steps):
        driver.step()
        E_vector_R.append(driver['E'][3][5])
        E_vector_T.append(driver['E'][3][95])

    for t in time_vector_coeffs:
        E_vector_0.append(np.exp(-(t-x0)**2.0/(2.0*s0**2.0)))

    time_vector_coeffs_corrected = time_vector_coeffs / 299792458

    dft_E_R=dft(E_vector_R,freq_vector,time_vector_coeffs_corrected)
    dft_E_T=dft(E_vector_T,freq_vector,time_vector_coeffs_corrected)
    dft_0=dft(E_vector_0,freq_vector,time_vector_coeffs_corrected)

    #Transmission and Reflection coefficients
    T = np.abs(dft_E_T) / np.abs(dft_0)
    R = np.abs(dft_E_R) / np.abs(dft_0)
    dB_T=20*np.log10(T) 
    dB_R=20*np.log10(R) 

    # Save data to .dat file
    with open("transmission_reflection_data4.dat", "w") as f:
        f.write("# Frequency (MHz)    Transmission (dB)    Reflection (dB)\n")
        for freq, t, r in zip(freq_vector / 1e6, dB_T, dB_R):
            f.write(f"{freq:.6e}    {t:.6e}    {r:.6e}\n")


    # T and R graph 
    plt.figure(figsize=(10, 8))
    plt.plot(freq_vector/1e6, dB_T, label='Transmission Coefficient (T)', color='purple')
    plt.plot(freq_vector/1e6, dB_R, label='Reflection Coefficient (R)', color='orange')
    plt.title('Transmission and Reflection Coefficients')
    plt.xlabel('Frequency (MHz)')
    plt.xlim(1e2, 1e3)
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()