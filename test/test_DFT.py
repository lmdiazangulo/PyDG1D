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
    # Function to manually compute the DFT

    # Create the Gaussian function
    x = np.linspace(-5, 5, 100) 
    s0 = 0.50
    gaussian = np.exp(-(x) ** 2 / (2 * s0 **2))  

    # Compue FFT and DFT
    fft_g = np.fft.fft(gaussian)

    dft_g = dft(gaussian)

    # Assert to check the difference between both transforms
    assert np.allclose(dft_g, fft_g, atol=0.01), "DFT and FFT differ by more than 0.01"

    # # Graph
    # plt.figure(figsize=(12, 6))
    # # Ejes de frecuencia para FFT y DFT
    # freq = np.fft.fftshift(np.fft.fftfreq(len(x), x[1] - x[0]))

    # # Original
    # plt.subplot(1, 3, 1)
    # plt.plot(x, gaussian, label='Gaussian')
    # plt.title('Gaussian')
    # plt.xlabel('x')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # plt.legend()

    # # DFT
    # plt.subplot(1, 3, 2)
    # plt.plot(freq, np.abs(fft_g), label='FFT')
    # plt.title('FFT Transform')
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # plt.legend()

    # # FFT
    # plt.subplot(1, 3, 3)
    # plt.plot(freq, np.abs(dft_g), label='DFT', linestyle='--')
    # plt.title('DFT Transform')
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    # plt.grid()
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
