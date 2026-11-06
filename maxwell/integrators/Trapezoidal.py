from ..spatialDiscretization import *

# Second order trapezoidal (implicit) time integrator.
#
# Implements the (2, 2) trapezoidal FDTD method for Maxwell's equations in
# linear Lorentz media of Jiang et al. (2018), Eq. (4.2).  The spatial
# discretization is responsible for the actual update (the polarization
# variables are eliminated analytically); this class only drives it.

class Trapezoidal:

    def __init__(self, sp: SpatialDiscretization, fields):
        self.sp = sp
        self.time = 0.0

    def step(self, fields, dt):
        self.sp.computeTrapezoidalStep(fields, dt)
        self.time += dt
