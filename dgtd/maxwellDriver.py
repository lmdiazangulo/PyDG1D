from .lserk4 import * 
from .leapfrog import *
from .spatialDiscretization import *

class MaxwellDriver:
    def __init__(self, sp: SpatialDiscretization, timeIntegratorType = 'Leapfrog'):
        self.sp = sp

        self.fields = sp.buildFields()

        
        # Compute time step size
        r_min = sp.get_minimum_node_distance()
        if (sp.get_mesh().dimension == 1):
            CFL = 1.0
            self.dt = CFL * r_min / 2.0
        elif (sp.get_mesh().dimension == 2):
            dtscale = sp.get_dt_scale()
            self.dt = min(dtscale)*r_min*2.0/3.0
            
        # Init time integrator
        if timeIntegratorType == 'LSERK4':
            self.timeIntegrator = LSERK4(self.sp, self.fields)
        elif timeIntegratorType == 'Leapfrog':
            self.timeIntegrator = Leapfrog(self.sp, self.fields)
        else:
            raise ValueError('Invalid time integrator')

    def step(self, dt = 0.0):
        if dt == 0.0:
            dt = self.dt
        self.timeIntegrator.step(self.fields, dt)

    def run(self, final_time):
        for t_step in range(1, np.ceil(final_time/self.dt)):
            self.step()

    def run_until(self, final_time):
        timeRange = np.arange(0.0, final_time, self.dt)
        for t in timeRange:
            self.step()

    def __getitem__(self, key):
        return self.fields[key]