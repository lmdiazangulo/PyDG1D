from maxwell.spatialDiscretization import *

from .integrators.LSERK4 import * 
from .integrators.DIRK2 import * 
from .integrators.IGLRK4 import *
from .integrators.IBE import * 
from .integrators.CN import * 
from .integrators.AM2 import * 
from .integrators.LSERK74 import * 
from .integrators.LSERK134 import * 
from .integrators.LF2 import *
from .integrators.LF2V import *
from .integrators.EULER import *


class MaxwellDriver:
    def __init__(self, 
                 sp: SpatialDiscretization, 
                 timeIntegratorType = 'LSERK4',
                 CFL = 1.0):
        self.sp = sp
        self.fields = sp.buildFields()
        
        # Compute time step size
        r_min = sp.get_minimum_node_distance()
        if (sp.isStaggered()):
            self.dt = CFL * r_min / np.sqrt(sp.dimension())
        else:
            if (sp.get_mesh().dimension == 1):
                self.dt = CFL * r_min * 2.0 / 3.0
            elif (sp.get_mesh().dimension == 2):
                dtscale = sp.get_dt_scale()
                self.dt = CFL * min(dtscale)*r_min*2.0/3.0

        self.sp.dt = self.dt       
            
        # Init time integrator
        if timeIntegratorType == 'EULER':
            self.timeIntegrator = EULER(self.sp, self.fields)
        elif timeIntegratorType == 'LSERK4':
            self.timeIntegrator = LSERK4(self.sp, self.fields)
        elif timeIntegratorType == 'LSERK74':
            self.timeIntegrator = LSERK74(self.sp, self.fields)
        elif timeIntegratorType == 'LSERK134':
            self.timeIntegrator = LSERK134(self.sp, self.fields)
        elif timeIntegratorType == 'LF2':
            self.timeIntegrator = LF2(self.sp, self.fields)
        elif timeIntegratorType == 'LF2V':
            self.timeIntegrator = LF2V(self.sp, self.fields)
        elif timeIntegratorType == 'IBE':
            self.timeIntegrator = IBE(self.sp, self.fields)
        elif timeIntegratorType == 'CN':
            self.timeIntegrator = CN(self.sp, self.fields)
        elif timeIntegratorType == 'DIRK2':
            self.timeIntegrator = DIRK2(self.sp, self.fields)
        elif timeIntegratorType == 'IGLRK4':
            self.timeIntegrator = IGLRK4(self.sp, self.fields)
        elif timeIntegratorType == 'AM2':
            self.timeIntegrator = AM2(self.sp, self.fields)
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
    
    def buildDrivedEvolutionOperator(self):
        if self.sp.mesh.dimension == 1:
            fieldE_components = 1
            fieldH_components = 1
        else:
            raise ValueError("Not implemented")
        fields_size = fieldE_components + fieldH_components
        Np = self.sp.number_of_nodes_per_element()
        K = self.sp.mesh.number_of_elements()
        N = fields_size * Np * K
        A = np.zeros((N,N))
        for i in range(N):
            self.fields = self.sp.buildFields()
            node = i % Np
            elem = int(np.floor(i / Np)) % K
            if i < N/2:
                self.fields['E'][node, elem] = 1.0
            else:
                self.fields['H'][node, elem] = 1.0
            self.step()
            q0 = np.vstack([
                self.fields['E'].reshape(Np*K,1, order='F'), 
                self.fields['H'].reshape(Np*K,1, order='F')
            ])
            A[:,i] = q0[:,0]
        return A