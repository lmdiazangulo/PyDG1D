from .spatialDiscretization import *

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

        self.fields = sp.buildFields()
            
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
        N = self.sp.number_of_unknowns()
        A = np.zeros((N,N))
        for i in range(N):
            self.fields = self.sp.buildFields()
            self.sp.setFieldWithIndex(self.fields, i, 1.0)
            self.step()
            q = self.sp.fieldsAsStateVector(self.fields) 
            A[:,i] = q[:]
        
        self.fields = self.sp.buildFields()
        
        return A
    
    def generateOutputFromAlternateBasisVectors(self):
        v_basis = self.sp.buildAlternateBasis()
        q_outputs = []

        if (self.sp.dimension() == 1):
            for i in range(len(v_basis)):
                self.fields = self.sp.buildFields()
                self.fields['E'][:] = v_basis[i][:len(self.sp.x)]
                self.fields['H'][:] = v_basis[i][len(self.sp.x):]
                self.step()
                qi = np.concatenate([self.fields['E'], self.fields['H']])
                q_outputs.append(qi)
            
            self.fields = self.sp.buildFields()

        elif (self.sp.dimension() == 2):
            for i in range(len(v_basis)):
                self.fields = self.sp.buildFields()
                if i < 2:
                    self.fields['E']['x'][:] = v_basis[i]
                elif i < 4:
                    self.fields['E']['y'][:] = v_basis[i]
                else:
                    self.fields['H'][:] = v_basis[i]
                self.step()

                qi = np.concatenate([self.fields['E']['x'].flatten(order='F'), self.fields['E']['y'].flatten(order='F'), self.fields['H'].flatten(order='F')])
                q_outputs.append(qi)
            
            self.fields = self.sp.buildFields()


        return q_outputs
    
    def buildDrivedEvolutionOperator_FromAlternateBasis(self):
        N = self.sp.number_of_unknowns()
        A = np.zeros((N,N))

        v_basis = self.sp.buildAlternateBasis()
        q_outputs = self.generateOutputFromAlternateBasisVectors()
        Q = np.column_stack(q_outputs)

        if (self.sp.dimension() == 2):
            vector_basis = []
            matrices_E = self.sp.buildElectricAlternateMatrixBasis()
            matrices_H = self.sp.buildMagneticAlternateMatrixBasis()

            for i, M in enumerate(matrices_E + matrices_H):
                if i < 2:
                    vector = np.concatenate([M.flatten(order='F'), 
                                            np.zeros(np.size(matrices_E[2])),
                                            np.zeros(np.size(matrices_H[0]))])
                elif i < 4:
                    vector = np.concatenate([np.zeros(np.size(matrices_E[0])),
                                            M.flatten(order='F'), 
                                            np.zeros(np.size(matrices_H[0]))])
                else:
                    vector = np.concatenate([np.zeros(np.size(matrices_E[0])),
                                            np.zeros(np.size(matrices_E[2])), 
                                            M.flatten(order='F')])
                vector_basis.append(vector)

            V = np.column_stack(vector_basis)

        else:
            V = np.column_stack(v_basis)

        # I need to add now the respective map to 2D
        column_nodes_map = self.sp.mesh.buildEvolutionOperator_Column_map()

        for col_idx in range(V.shape[1]):
            v_col = V[:, col_idx]
            nonzero_indices = np.nonzero(v_col)[0]
            if len(nonzero_indices) == 0:
                continue  

            for k in nonzero_indices:  
                affected_indices = column_nodes_map.get(k, [])

                for i in affected_indices:
                    A[i, k] = Q[i, col_idx]  

        return A