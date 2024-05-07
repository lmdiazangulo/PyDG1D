import numpy as np


class SpatialDiscretization():
    def __init__(self, mesh):
        self.mesh = mesh

        return

    def get_mesh(self):
        return self.mesh

    def isStaggered(self):
        return False

    def dimension(self):
        return 1

    def fieldsAsStateVector(self, fields):
        q = np.array([])
        for f in fields.values():
            q = np.append(q, f.reshape(f.size, 1, order='F'))
        return q

    def buildStateVector(self):
        fields = self.buildFields()
        q0 = np.array([])
        for f in fields.values():
            q0 = np.append(q0, f)
        return q0

    def buildImpulseStateVector(self, i):
        q = self.buildStateVector()
        q[i] = 1.0
        return q

    def buildLocalAndNeighborIndices(self, element, neighs):
        Np = self.number_of_nodes_per_element()
        N = self.number_of_unknowns()
        k = element 
        local_indices = np.arange(k*2*Np, (k+1)*2*Np)
        right_neigh_indices = local_indices[-1] + 1 + np.arange(0, 2*Np*neighs)
        left_neigh_indices = np.sort(
            local_indices[0] - np.arange(0, 2*Np*neighs) - 1) % N
        neigh_indices = np.sort(np.concatenate(
            (left_neigh_indices, right_neigh_indices)))

        return local_indices, neigh_indices

    def number_of_unknowns(self):
        return len(self.buildStateVector())
