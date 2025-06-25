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
    
    def buildEvolutionOperatorMap(self):
        N = self.sp.number_of_unknowns()
        x_len = len(self.sp.x)

        # This part of index must be improved, in the examples of 1D FDTD this points are just the necessary ones and the covers
        # the frontiers columns of the time evolution operator (in both E and H) and two arbitrary points between each frontier
        # to obtain the general columns form from the operator.

        # Select representative indices from the evolution matrix
        indices = [
            0,                          # First E point
            (x_len - 1) // 2,           # Middle E point
            x_len - 1,                  # Last E point
            x_len,                      # First H point
            (N + x_len) // 2,           # Middle H point
            N - 1                       # Last H point
        ]

        # Compute the columns corresponding to the selected indices
        selected_columns = {}
        for idx in indices:
            self.fields = self.sp.buildFields()
            self.sp.setFieldWithIndex(self.fields, idx, 1.0)
            self.step()
            q = self.sp.fieldsAsStateVector(self.fields)
            selected_columns[idx] = q.copy()
        self.fields = self.sp.buildFields()  

        # Map each k to the nonzero indices of the relevant column
        k_to_nonzero_map = {}
        for k in range(N):
            if k in [indices[0], indices[2], indices[3], indices[5]]:
                col = selected_columns[k if k in selected_columns else indices[0]]
            elif k < x_len:
                # E: shift the middle E column as needed
                shift = k - indices[1]
                col = np.roll(selected_columns[indices[1]], shift)
            else:
                # H: shift the middle H column as needed
                shift = k - indices[4]
                col = np.roll(selected_columns[indices[4]], shift)
            k_to_nonzero_map[k] = np.flatnonzero(col).tolist()

        return k_to_nonzero_map
    
    def generateOutputFromAlternateBasisVectors(self):
        v_basis = self.sp.buildAlternateBasisVectors()
        q_outputs = []

        for i in range(len(v_basis)):
            self.fields = self.sp.buildFields()
            self.fields['E'][:] = v_basis[i][:len(self.sp.x)]
            self.fields['H'][:] = v_basis[i][len(self.sp.x):]
            self.step()
            qi = np.concatenate([self.fields['E'], self.fields['H']])
            q_outputs.append(qi)
        
        self.fields = self.sp.buildFields()

        return q_outputs
    
    def buildDrivedEvolutionOperator_FromAlternateBasis(self):
        N = self.sp.number_of_unknowns()
        A = np.zeros((N,N))

        v_basis = self.sp.buildAlternateBasisVectors()
        q_outputs = self.generateOutputFromAlternateBasisVectors()
        Q = np.column_stack(q_outputs)
        V = np.column_stack(v_basis)

        column_nodes_map = self.buildEvolutionOperatorMap()

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