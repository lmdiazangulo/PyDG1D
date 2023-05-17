import numpy as np

class SpatialDiscretization():
    def __init__(self):
        return
    
    def buildFields(self):
        E = np.zeros([self.number_of_nodes_per_element(),
                          self.mesh.number_of_elements()])
        H = np.zeros(E.shape)

        return E, H