from .spatialDiscretization import *

import copy
import scipy.sparse 


class ModelOrderReduction:
    def __init__(self, sp : SpatialDiscretization, evolutionOperator, energyThreshold = 1 - 1e-12):
        self.sp = sp
        self.evolutionOperator = evolutionOperator
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

            q = self.evolutionOperator @ q

        return self.snapshots
    
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
    
    def buildReducedProjectionAndReducedEvolutionOperator(self, snapshots, number_of_modes):
        symmetricSnapshotMatrix = snapshots.T @ snapshots
        vals, vecs = scipy.sparse.linalg.eigsh(symmetricSnapshotMatrix, k=number_of_modes, which='LM')
        
        idx = np.argsort(vals)[::-1]
        reducedEigenValues = vals[idx]
        reducedEigenVectors = vecs[:, idx]

        Ur = np.zeros((np.size(snapshots.T[0]), number_of_modes))

        for i in range(number_of_modes):
            Ur[:, i] = (snapshots @ reducedEigenVectors.T[i]) / np.sqrt(reducedEigenValues[i])

        Ur = Ur[:, ~np.isnan(Ur).all(axis=0)] # Remove NaN columns if any

        Ar = Ur.T @ self.evolutionOperator @ Ur

        return Ur, Ar
    
    def buildReducedOrderModel_truncated_SVD(self):
        if not hasattr(self, 'snapshots'):
            raise ValueError("You need to build the snapshots first using buildSnapshots_ProperOrthogonalDecomposition method.")
        
        r, r_forControl = self.getReducedOrderDimension(self.snapshots)
        Ur, Ar = self.buildReducedProjectionAndReducedEvolutionOperator(self.snapshots, r)
        Ur1, Ar1 = self.buildReducedProjectionAndReducedEvolutionOperator(self.snapshots, r_forControl)

        return Ur, Ar, Ur1, Ar1
    
    def updateReducedOrderModel(self, actualState, Ur, adaptativeSteps):
        auxiliarSnapshots = np.zeros((np.size(actualState), adaptativeSteps))

        for i in range(adaptativeSteps):
            auxiliarSnapshots[:, i] = actualState
            actualState = self.evolutionOperator @ actualState

        r, _ = self.getReducedOrderDimension(auxiliarSnapshots)
        Ur_aux, _ = self.buildReducedProjectionAndReducedEvolutionOperator(auxiliarSnapshots, r)

        newSnapshots = np.hstack((Ur, Ur_aux))

        r_new, r_new_forControl = self.getReducedOrderDimension(newSnapshots)

        Ur_new, Ar_new = self.buildReducedProjectionAndReducedEvolutionOperator(newSnapshots, r_new)
        Ur1_new, Ar1_new = self.buildReducedProjectionAndReducedEvolutionOperator(newSnapshots, r_new_forControl)

        return Ur_new, Ar_new, Ur1_new, Ar1_new
    
    def step_ROM(self, actualReducedState, actualReducedStateControl, Ur, Ar, Ur1, Ar1, errorCriterionForAdaptative=1e-4, adaptativeSteps=10):
        q_r = copy.deepcopy(actualReducedState)
        q_r = Ar @ q_r
        q = Ur @ q_r

        actualState = Ur1 @ copy.deepcopy(actualReducedStateControl)

        q_r1 = copy.deepcopy(actualReducedStateControl)
        q_r1 = Ar1 @ q_r1
        q1 = Ur1 @ q_r1

        if (np.linalg.norm(q - q1) / np.linalg.norm(q1) > errorCriterionForAdaptative):
            Ur_new, Ar_new, Ur1_new, Ar1_new = self.updateReducedOrderModel(copy.deepcopy(actualState), Ur, adaptativeSteps)

            q_r_new = Ur_new.T @ copy.deepcopy(actualState)
            q_r_new = Ar_new @ q_r_new

            q_r1_new = Ur1_new.T @ copy.deepcopy(actualState)
            q_r1_new = Ar1_new @ q_r1_new

            return q_r_new, q_r1_new, Ur_new, Ar_new, Ur1_new, Ar1_new

        return q_r, q_r1, Ur, Ar, Ur1, Ar1
    
    def run_until_ROM(self, initialState, Ur, Ar, Ur1, Ar1, finalTime, errorCriterionForAdaptative=1e-4, adaptativeSteps=10):
        q = copy.deepcopy(initialState)
        reducedState = Ur.T @ q
        reducedStateControl = Ur1.T @ q

        timeRange = np.arange(0.0, finalTime, self.sp.dt)

        for t in timeRange:
            reducedState, reducedStateControl, Ur, Ar, Ur1, Ar1 = self.step_ROM(reducedState, reducedStateControl, Ur, Ar, Ur1, Ar1, errorCriterionForAdaptative, adaptativeSteps)

        return Ur @ reducedState