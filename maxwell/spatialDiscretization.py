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
            q = np.append(q, f.reshape(f.size,1, order='F')) 
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
    
    def number_of_unknowns(self):
        return len(self.buildStateVector())

