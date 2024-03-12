import numpy as np

class SpatialDiscretization():
    def __init__(self, mesh):
        self.mesh = mesh
        
        self.n_faces = 2
        self.n_fp = 1   

        return
    
    def get_mesh(self):
        return self.mesh
    
    def isStaggered(self):
        return False
