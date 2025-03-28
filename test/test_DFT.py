import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytest
import time

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for i in range(N):
        for n in range(N):
            X[i]=X[i]+ x[n] * np.exp(-2j*np.pi*i*n/N)
    return X

def test_DFT():
    # Función para realizar la DFT manualmente

    # Monto la gaussiana
    x = np.linspace(-5, 5, 100) 
    s0 = 0.50
    gaussiana = np.exp(-(x) ** 2 / (2 * s0 **2))  # Gaussiana

    # FFT y DFT
    fft_g = np.fft.fft(gaussiana)

    dft_g = dft(gaussiana)

    # Grafico
    plt.figure(figsize=(12, 6))
    # Ejes de frecuencia para FFT y DFT
    freq = np.fft.fftshift(np.fft.fftfreq(len(x), x[1] - x[0]))

    # Original
    plt.subplot(1, 3, 1)
    plt.plot(x, gaussiana, label='Gaussiana')
    plt.title('Gaussiana')
    plt.xlabel('x')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.legend()

    # DFT
    plt.subplot(1, 3, 2)
    plt.plot(freq, np.abs(fft_g), label='FFT')
    plt.title('Transformada FFT')
    plt.xlabel('Frecuencia')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.legend()

    # FFT
    plt.subplot(1, 3, 3)
    plt.plot(freq, np.abs(dft_g), label='DFT', linestyle='--')
    plt.title('Transformada DFT')
    plt.xlabel('Frecuencia')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def test_python_fft_practice():
        
    epsilon_1=1
    # epsilon_2=1
    # mu_1=1
    # mu_2=1
    # z_1=np.sqrt(mu_1/epsilon_1)
    # z_2=np.sqrt(mu_2/epsilon_2)
    # v_1=1/np.sqrt(epsilon_1*mu_1)
    # v_2=1/np.sqrt(epsilon_2*mu_2)
    L1=-5.0
    L2=5.0
    elements=100
    epsilons = epsilon_1*np.ones(elements)
    sigmas=np.zeros(elements) 
    sigmas[45:55]=2.
    # epsilons[int(elements/2):elements-1]=epsilon_2

    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(L1, L2, elements, boundary_label="PEC"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    s0 = 0.50
    x0=-2
    final_time = 1
    steps = 100
    time_vector = np.linspace(0, final_time, steps)
    #freq_vector = np.logspace(6, 9, 31) not necessary in fft

    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    initialFieldH = initialFieldE

    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    E_vector = []

    for t in range(len(time_vector)):
        driver.run_until(time_vector[t])
        E_vector.append(driver['E'][2])


    N = steps  # Number of time points
    dt = time_vector[1] - time_vector[0]  # Time step (assuming uniform spacing)
    
    # Compute FFT and corresponding frequencies
    E_freq = np.fft.fft(E_vector)  # Fourier Transform
    freqs = np.fft.fftfreq(N, d=dt)  # Compute frequency values
    
    # Shift zero frequency component to the center
    E_freq_shifted = np.fft.fftshift(E_freq)  
    freqs_shifted = np.fft.fftshift(freqs)  
    
    # Plot magnitude spectrum
    plt.figure(figsize=(8, 5))
    plt.plot(freqs_shifted, np.abs(E_freq_shifted), label="Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude |E(f)|")
    plt.title("Fourier Transform of Electric Field")
    plt.grid()
    plt.legend()
    plt.show()

