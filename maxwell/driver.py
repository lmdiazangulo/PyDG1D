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

import copy
import scipy.sparse 


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
    
    def buildDrivedEvolutionOperator(self, reduceToEssentialDoF=True):
        N = self.sp.number_of_unknowns()
        A = np.zeros((N,N))
        
        oldFields = copy.deepcopy(self.fields)
        
        for i in range(N):
            self.fields = self.sp.buildFields()
            self.sp.setFieldWithIndex(self.fields, i, 1.0)
            self.step()
            q = self.sp.fieldsAsStateVector(self.fields) 
            A[:,i] = q[:]
        
        self.fields = oldFields
        
        if reduceToEssentialDoF and self.sp.isStaggered():
            A = self.sp.reduceToEssentialDoF(A)
        
        return A
    
    def buildPowerOperator(self):
        G = self.buildDrivedEvolutionOperator()
        Mg = self.sp.buildGlobalMassMatrix()
        return (1/self.dt)*(G.T.dot(Mg).dot(G) - Mg)
    

    def buildCausallyConnectedOperators(self, element=0, neighbors=-1):
        if neighbors == -1:
            neighs = self.timeIntegrator.N_STAGES
        else:
            neighs = neighbors
            
        local_indices, neigh_indices = self.sp.buildLocalAndNeighborIndices(element, neighs)

        G = self.sp.reorder_by_elements(self.buildDrivedEvolutionOperator())
        Mg =  self.sp.reorder_by_elements(self.sp.buildGlobalMassMatrix())
        A = G[local_indices][:,local_indices]
        B = G[local_indices][:,neigh_indices]
        C = G[neigh_indices][:,local_indices]
        D = G[neigh_indices][:,neigh_indices]
        Mk = Mg[local_indices][:,local_indices]
        Mn = Mg[neigh_indices][:,neigh_indices]
        
        return A, B, C, D, Mk, Mn
    
    def generateOutputFromAlternateBasisVectors(self):
        v_basis = self.sp.buildAlternateBasis()
        q_outputs = []
        
        oldFields = copy.deepcopy(self.fields)

        if (self.sp.dimension() == 1):
            for i in range(len(v_basis)):
                self.fields = self.sp.buildFields()
                self.fields['E'][:] = v_basis[i][:len(self.sp.x)]
                self.fields['H'][:] = v_basis[i][len(self.sp.x):]
                self.step()
                qi = self.sp.fieldsAsStateVector(self.fields)
                q_outputs.append(qi)
            
            self.fields = oldFields

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
        # A = np.zeros((N,N))
        A = scipy.sparse.lil_matrix((N, N))

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

        A = A.tocsr()            

        return A

    def buildSnapshots_ProperOrthogonalDecomposition(self, number_of_snapshots, time_step_skip=1):
        oldFields = copy.deepcopy(self.fields)
        qi = self.sp.fieldsAsStateVector(self.fields)

        if np.allclose(qi, 0.0):
            raise ValueError("Initial condition is zero, all the snapshots will be zero.")
        
        self.snapshots = np.zeros((len(qi), number_of_snapshots))
        self.number_of_snapshots = number_of_snapshots

        for n in range(number_of_snapshots):
            self.snapshots[:,n] = qi

            for t in range(time_step_skip):
                self.step()

            qi = self.sp.fieldsAsStateVector(self.fields)

        self.fields = oldFields

        return self.snapshots
    
    def buildSingularValueDecomposition(self):
        if not hasattr(self, 'snapshots'):
            raise ValueError("You need to build the snapshots first using buildSnapshots_ProperOrthogonalDecomposition method.")
        
        U, S, VT = np.linalg.svd(self.snapshots, full_matrices=False)
        return U, S, VT
    
    def quadraticEnergyCriterionForTruncation(self, percentage_threshold, S):
        total_energy = np.sum(S**2)
        cumulative_energy = np.cumsum(S**2) / total_energy
        r = np.searchsorted(cumulative_energy, percentage_threshold) + 1
        return r
    
    def energyCriterionForTruncation(self, percentage_threshold, S):
        total_energy = np.sum(S)
        cumulative_energy = np.cumsum(S) / total_energy
        r = np.searchsorted(cumulative_energy, percentage_threshold) + 1
        return r
    
    def buildReducedOrderModel(self, percentage_threshold=0.99, useAlternateBasis=False):
        
        if useAlternateBasis:
            A = self.buildDrivedEvolutionOperator_FromAlternateBasis()
        else:
            A = self.buildDrivedEvolutionOperator(reduceToEssentialDoF=False)
        
        U, S, VT = self.buildSingularValueDecomposition()
        r = self.quadraticEnergyCriterionForTruncation(percentage_threshold, S)

        Ur = U[:,:r]
        # Ar = Ur.T.dot(A).dot(Ur)
        Ar = Ur.T @ A @ Ur
        
        return Ur, Ar
    
    def buildReducedOrderModel_truncated_SVD(self, percentage_threshold=0.99, useAlternateBasis=False):

        C = self.snapshots.T.dot(self.snapshots)
        eigenValues = np.linalg.eigvalsh(C)
        eigenValues = eigenValues[::-1]
        r = self.energyCriterionForTruncation(percentage_threshold, eigenValues)

        Ur, Ar = self.buildReducedProjectionAndEvolutionOperators(number_of_important_eig_values=r, snapshot=self.snapshots, useAlternateBasis=useAlternateBasis)
        Ur1, Ar1 = self.buildReducedProjectionAndEvolutionOperators(number_of_important_eig_values=r+1, snapshot=self.snapshots, useAlternateBasis=useAlternateBasis)

        return Ur, Ar, Ur1, Ar1

    def buildReducedProjectionAndEvolutionOperators(self, number_of_important_eig_values, snapshot, useAlternateBasis=False):
        if useAlternateBasis:
            A = self.buildDrivedEvolutionOperator_FromAlternateBasis()
        else:
            A = self.buildDrivedEvolutionOperator(reduceToEssentialDoF=False)

        symmetric_snapshot_matrix = snapshot.T.dot(snapshot)

        reducedEigenValues, reducedEigenVectors = scipy.sparse.linalg.eigsh(symmetric_snapshot_matrix, k=number_of_important_eig_values, which="LM")
        Ur = np.zeros((np.size(snapshot.T[0]), number_of_important_eig_values))

        for k in range(number_of_important_eig_values):
            Ur[:, k] = snapshot.dot(reducedEigenVectors.T[k]) / np.sqrt(reducedEigenValues[k])

        Ar = Ur.T @ A @ Ur
        
        return Ur, Ar
    
    def evolveReducedOrderModel(self, Ar, Ur, Ar1, Ur1, initialField, time_steps, eps_adaptative=1e-4, useAlternateBasis=False):
        
        qi = self.sp.fieldsAsStateVector(initialField)

        qi_r = Ur.T.dot(copy.deepcopy(qi))
        qi_r1 = Ur1.T.dot(copy.deepcopy(qi))

        # qf_r = np.linalg.matrix_power(Ar, time_steps).dot(qi_r)

        qf_r = copy.deepcopy(qi_r)
        qf_r1 = copy.deepcopy(qi_r1)
        k = 0

        while k < time_steps:
            qf_r = Ar @ qf_r
            qf_r1 = Ar1 @ qf_r1

            qf_1 = Ur1 @ qf_r1
            qf = Ur @ qf_r

            if (np.linalg.norm(qf_1 - qf) / np.linalg.norm(qf_1) > eps_adaptative):
                print("Warning: The reduced order model might be inaccurate. Consider increasing the number of basis vectors.")

                # Ur, Ar, Ur1, Ar1 = self.updateReducedOrderModel(copy.deepcopy(qf_1), Ur=Ur, Ur1=Ur1, percentage_threshold=1-1e-12, useAlternateBasis=useAlternateBasis)

                self.updateSnapshots(qf_1)
                Ur, Ar, Ur1, Ar1 = self.buildReducedOrderModel_truncated_SVD(percentage_threshold=1-1e-12, useAlternateBasis=useAlternateBasis)

                qf_r = Ur.T.dot(copy.deepcopy(qi))
                qf_r1 = Ur1.T.dot(copy.deepcopy(qi))
                k = 0
                continue 

                # qf_r = Ur.T.dot(copy.deepcopy(qf_1))
                # qf_r1 = Ur1.T.dot(copy.deepcopy(qf_1))

            k += 1

        return Ur.dot(qf_r)

    def updateSnapshots(self, actualState):

        A = self.buildDrivedEvolutionOperator_FromAlternateBasis()

        for k in range(10):
            self.snapshots = np.column_stack((self.snapshots, actualState / np.linalg.norm(actualState)))
            actualState = A.dot(actualState)  

        return self.snapshots

    def updateReducedOrderModel(self, actualState, Ur, Ur1, percentage_threshold=1-1e-6, useAlternateBasis=False):

        if useAlternateBasis:
            A = self.buildDrivedEvolutionOperator_FromAlternateBasis()
        else:
            A = self.buildDrivedEvolutionOperator(reduceToEssentialDoF=False)

        auxiliarSnapshots = np.zeros((len(actualState), 10))
        
        for k in range(10):
            auxiliarSnapshots[:,k] = actualState
            actualState = A.dot(actualState)

        C = auxiliarSnapshots.T.dot(auxiliarSnapshots)
        eigenValues = np.linalg.eigvalsh(C)
        eigenValues = eigenValues[::-1]
        r = self.energyCriterionForTruncation(percentage_threshold, eigenValues)

        Ur_aux, _ = self.buildReducedProjectionAndEvolutionOperators(number_of_important_eig_values=r, snapshot=auxiliarSnapshots, useAlternateBasis=useAlternateBasis)

        newSnapshots = np.hstack((Ur, Ur_aux))

        newC = newSnapshots.T.dot(newSnapshots)
        newEigenValues = np.linalg.eigvalsh(newC)
        newEigenValues = newEigenValues[::-1]
        new_r = self.energyCriterionForTruncation(percentage_threshold, newEigenValues)

        Ur, Ar = self.buildReducedProjectionAndEvolutionOperators(number_of_important_eig_values=new_r-1, snapshot=newSnapshots, useAlternateBasis=useAlternateBasis)
        Ur1, Ar1 = self.buildReducedProjectionAndEvolutionOperators(number_of_important_eig_values=new_r, snapshot=newSnapshots, useAlternateBasis=useAlternateBasis)

        
        # totalSnapshots = np.column_stack((self.snapshots, auxiliarSnapshots))
        # self.snapshots = totalSnapshots

        return Ur, Ar, Ur1, Ar1
