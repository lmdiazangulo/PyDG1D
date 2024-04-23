import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from maxwell.dg.mesh2d import *
from maxwell.dg.dg2d import *
from maxwell.driver import *
from maxwell.fd.fd2d import *


def gaussian(s):
    return lambda x : np.exp(-(x)**2/(2*s**2))

#······················································

def plot(sp, driver):
    for _ in range(1000):
        driver.step()
        plt.plot(sp.x, driver['E'],'b')
        plt.plot(sp.xH, driver['H'],'r')
        plt.ylim(-1.5, 1.5)
        plt.title(driver.timeIntegrator.time)
        plt.grid(which='both')
        plt.pause(0.01)
        plt.cla()
        
#······················································
