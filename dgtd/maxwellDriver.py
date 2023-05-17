from .lserk4 import * 
from .spatialDiscretization import *

class MaxwellDriver:
    def __init__(self, sp: SpatialDiscretization, timeIntegratorType = 'LSERK4'):
        self.sp = sp

        self.E, self.H = sp.buildFields()

        # Compute time step size
        x_min = sp.get_minimum_node_distance()
        CFL = 1.0
        self.dt = CFL * x_min / 2
        
        # Init time integrator
        if timeIntegratorType == 'LSERK4':
            self.timeIntegrator = LSERK4(self.sp, (self.E, self.H))
        else:
            raise ValueError('Invalid time integrator')

    def step(self, dt = 0.0):
        if dt == 0.0:
            dt = self.dt
        self.timeIntegrator.step([self.E, self.H], dt)

    def run(self, final_time):
        for t_step in range(1, np.ceil(final_time/self.dt)):
            self.step()

    def run_until(self, final_time):
        timeRange = np.arange(0.0, final_time, self.dt)
        for t in timeRange:
            self.step()
