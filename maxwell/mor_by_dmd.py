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

        for i in range(number_of_snapshots):
            self.driver.fields['E'] = self.sp.stateVectorAsFields(snapshots[:, i])['E']
            self.driver.fields['H'] = self.sp.stateVectorAsFields(snapshots[:, i])['H']
            self.driver.step()
            evolved_snapshots[:, i] = self.sp.fieldsAsStateVector(self.driver.fields)

        return evolved_snapshots
    
    def energyCriterionForTruncation(self, singularValues, energyThreshold, order=1):
        total_energy = np.sum(singularValues**order)
        cumulative_energy = np.cumsum(singularValues**order) / total_energy
        r = np.searchsorted(cumulative_energy, energyThreshold) + 1
        return r
    
    def getReducedOrderDimension(self, snapshots):
        symmetricSnapshotMatrix = snapshots.T @ snapshots
        eigenValues = np.linalg.eigvalsh(symmetricSnapshotMatrix)
        eigenValues = eigenValues[::-1]

        dimensionROM = self.energyCriterionForTruncation(eigenValues, self.energyThreshold, order=1)
        dimensionROM_forControl = self.energyCriterionForTruncation(eigenValues, (99 + self.energyThreshold) / 100, order=1)

        if dimensionROM == dimensionROM_forControl:
            dimensionROM_forControl = min(dimensionROM + 1, len(eigenValues))
        
        return dimensionROM, dimensionROM_forControl
    
    def getreducedSVDdecomposition(self, snapshots):
        if not hasattr(self, 'snapshots'):
            raise ValueError("You need to build the snapshots first before calling the method.")
        
        U, S, Vh = np.linalg.svd(snapshots)
        r = self.energyCriterionForTruncation(S, self.energyThreshold, order=2)

        Ur = U[:, :r]
        Sr = S[:r]
        Vhr = Vh[:r, :]

        return Ur, Sr, Vhr
    
    def generateFullDimensionEvolutionOperator(self, snapshots):
        if not hasattr(self, 'snapshots'):
            raise ValueError("You need to build the snapshots first before calling the method.")
        
        snapshots_evolved = self.buildEvolvedSnapshots(snapshots)

        Ur, Sr, Vhr = self.getreducedSVDdecomposition(snapshots)

        r = Sr.shape[0]
        S_inv = np.zeros((r, r))
        for i in range(r):
            S_inv[i, i] = 1 / Sr[i]

        A = snapshots_evolved @ Vhr.T @ S_inv @ Ur.T

        return A