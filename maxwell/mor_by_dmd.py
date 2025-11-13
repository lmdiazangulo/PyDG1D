from .spatialDiscretization import *
from .driver import *

import copy
import scipy.sparse 

class ModelOrderReduction_by_DynamicModeDecomposition:
    def __init__(self, sp : SpatialDiscretization, driver : MaxwellDriver, energyThreshold = 1 - 1e-12):
        self.sp = sp
        self.driver = driver
        self.energyThreshold = energyThreshold

    def addSnapshot(self, snapshot):
        if not hasattr(self, 'snapshots'):
            self.snapshots = snapshot
        else:
            self.snapshots = np.hstack((self.snapshots, snapshot))

    def buildSnapshots_fromInitialState(self, initialState, finalTime, time_step_skip=1):
        q = copy.deepcopy(initialState)

        number_of_time_steps = int(finalTime / self.sp.dt)
        number_of_snapshots = number_of_time_steps // time_step_skip

        self.snapshots = np.zeros((np.size(q), number_of_snapshots))

        for n in range(number_of_time_steps):
            if n % time_step_skip == 0:
                self.snapshots[:, n // time_step_skip] = q

            self.driver.step()
            q = self.sp.fieldsAsStateVector(self.driver.fields)

        self.driver.fields['E'] = self.sp.stateVectorAsFields(initialState)['E']
        self.driver.fields['H'] = self.sp.stateVectorAsFields(initialState)['H']

        return self.snapshots
    
    def buildEvolvedSnapshots(self, snapshots):
        number_of_snapshots = snapshots.shape[1]
        evolved_snapshots = np.zeros_like(snapshots)

        oldFields = copy.deepcopy(self.driver.fields)

        for i in range(number_of_snapshots):
            self.driver.fields['E'] = self.sp.stateVectorAsFields(snapshots[:, i])['E']
            self.driver.fields['H'] = self.sp.stateVectorAsFields(snapshots[:, i])['H']
            self.driver.step()
            evolved_snapshots[:, i] = self.sp.fieldsAsStateVector(self.driver.fields)

        self.driver.fields = oldFields

        return evolved_snapshots
    
    def energyCriterionForTruncation(self, singularValues, energyThreshold, order=1):
        total_energy = np.sum(singularValues**order)
        cumulative_energy = np.cumsum(singularValues**order) / total_energy
        r = np.searchsorted(cumulative_energy, energyThreshold) + 1
        return r
    
    def getReducedOrderDimension(self, eigenValues):
        dimensionROM = self.energyCriterionForTruncation(eigenValues, self.energyThreshold, order=2)
        dimensionROM_forControl = self.energyCriterionForTruncation(eigenValues, (99 + self.energyThreshold) / 100, order=2)

        if dimensionROM == dimensionROM_forControl:
            dimensionROM_forControl = min(dimensionROM + 1, len(eigenValues))
        
        return dimensionROM, dimensionROM_forControl
    
    def getreducedSVDdecomposition(self, snapshots):
        if not hasattr(self, 'snapshots'):
            raise ValueError("You need to build the snapshots first before calling the method.")
        
        U, S, Vh = np.linalg.svd(snapshots, full_matrices=False)
        r, r_control = self.getReducedOrderDimension(S)

        Ur = U[:, :r]
        Sr = S[:r]
        Vhr = Vh[:r, :]

        Ur_control = U[:, :r_control]
        Sr_control = S[:r_control]
        Vhr_control = Vh[:r_control, :]

        return Ur, Sr, Vhr, Ur_control, Sr_control, Vhr_control
    
    def generateFullDimensionEvolutionOperator(self, snapshots):

        Ur, Sr, Vhr, _, _, _ = self.getreducedSVDdecomposition(snapshots)
        snapshots_evolved = self.buildEvolvedSnapshots(snapshots)

        r = Sr.shape[0]
        S_inv = np.zeros((r, r))
        for i in range(r):
            S_inv[i, i] = 1 / Sr[i]

        A = np.matmul(np.matmul(snapshots_evolved, Vhr.T), np.matmul(S_inv, Ur.T))

        return A
    
    def buildReducedProjectionAndReducedEvolutionOperator(self, snapshots):
        
        Ur, Sr, Vhr, Ur_control, Sr_control, Vhr_control = self.getreducedSVDdecomposition(snapshots)
        snapshots_evolved = self.buildEvolvedSnapshots(snapshots)

        r = Sr.shape[0]
        S_inv = np.zeros((r, r))
        for i in range(r):
            S_inv[i, i] = 1 / Sr[i]

        r_control = Sr_control.shape[0]
        S_inv_control = np.zeros((r_control, r_control))
        for i in range(r_control):
            S_inv_control[i, i] = 1 / Sr_control[i]


        Ar = np.matmul(np.matmul(Ur.T, snapshots_evolved), np.matmul(Vhr.T, S_inv))
        Ar_control = np.matmul(np.matmul(Ur_control.T, snapshots_evolved), np.matmul(Vhr_control.T, S_inv_control))

        return Ur, Ar, Ur_control, Ar_control
    
    def buildInitialReducedOrderModel(self):
        return self.buildReducedProjectionAndReducedEvolutionOperator(self.snapshots)
    
    def updateReducedOrderModel(self, actualState, Ur, adaptativeSteps):
        auxiliarSnapshots = np.zeros((np.size(actualState), adaptativeSteps))
        oldFields = self.sp.stateVectorAsFields(actualState)

        for i in range(adaptativeSteps):
            auxiliarSnapshots[:, i] = actualState

            actual_field = self.sp.stateVectorAsFields(actualState)
            self.driver['E'][:] = actual_field['E']
            self.driver['H'][:] = actual_field['H']
            self.driver.step()
            actualState = self.sp.fieldsAsStateVector(self.driver.fields)

        self.driver.fields['E'] = oldFields['E']
        self.driver.fields['H'] = oldFields['H']


        Ur_aux, _, _, _ = self.buildReducedProjectionAndReducedEvolutionOperator(auxiliarSnapshots)
        newSnapshots = np.hstack((Ur, Ur_aux))
        Ur_new, Ar_new, Ur1_new, Ar1_new = self.buildReducedProjectionAndReducedEvolutionOperator(newSnapshots)

        return Ur_new, Ar_new, Ur1_new, Ar1_new
    
    def step_ROM(self, actualReducedState, actualReducedStateControl, Ur, Ar, Ur1, Ar1, errorCriterionForAdaptative=1e-3, adaptativeSteps=10):
        q_r = copy.deepcopy(actualReducedState)
        q_r = np.matmul(Ar, q_r)

        previousReducedStateControl = copy.deepcopy(actualReducedStateControl)
        dimensionDifference = np.size(actualReducedStateControl) - np.size(actualReducedState)

        q_r1 = copy.deepcopy(actualReducedStateControl)
        q_r1 = np.matmul(Ar1, q_r1)

        if (np.linalg.norm(q_r1[-dimensionDifference:])  >= errorCriterionForAdaptative * np.linalg.norm(q_r1)):
            actualState = np.matmul(Ur1, previousReducedStateControl)
            Ur_new, Ar_new, Ur1_new, Ar1_new = self.updateReducedOrderModel(copy.deepcopy(actualState), Ur, adaptativeSteps)

            q_r_new = np.matmul(Ur_new.T, copy.deepcopy(actualState))
            q_r_new = np.matmul(Ar_new, q_r_new)

            q_r1_new = np.matmul(Ur1_new.T, copy.deepcopy(actualState))
            q_r1_new = np.matmul(Ar1_new, q_r1_new)

            return q_r_new, q_r1_new, Ur_new, Ar_new, Ur1_new, Ar1_new

        return q_r, q_r1, Ur, Ar, Ur1, Ar1
    
    def run_until_ROM(self, initialState, Ur, Ar, Ur1, Ar1, finalTime, errorCriterionForAdaptative=1e-4, adaptativeSteps=10):
        q = copy.deepcopy(initialState)

        reducedState = np.matmul(Ur.T, q)
        reducedStateControl = np.matmul(Ur1.T, q)

        timeRange = np.arange(0.0, finalTime, self.sp.dt)

        for t in timeRange:
            reducedState, reducedStateControl, Ur, Ar, Ur1, Ar1 = self.step_ROM(reducedState, reducedStateControl, Ur, Ar, Ur1, Ar1, errorCriterionForAdaptative, adaptativeSteps)

        return np.matmul(Ur, reducedState)