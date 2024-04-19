import numpy as np


DIR_X = 0
DIR_Y = 1
DIR_Z = 2

TAG_X = "x"
TAG_Y = "y"
TAG_Z = "z"


class General:
    def __init__(self):
        self.time_step = 0.0
        self.number_of_steps = 0

class Grid:
    def __init__(self):
        # self.number_of_cells = []
        self.dx = np.array([])
        self.dy = np.array([])
        self.dz = np.array([])

class element:
    def __init__(self, element_description : dict):
        
        if not "id" in element_description.keys():
            raise Exception("Key 'id' absent in element description")
        if not "type" in element_description.keys():
            raise Exception("Key 'type' absent in element description")
        if not "coordinateIds" in element_description.keys():
            raise Exception("Key 'coordinateIds' absent in element description")

        self.id = element_description["id"]
        self.type = element_description["type"]
        self.coordinate_ids = element_description["coordinateIds"]
        
class group:
    def __init__(self, elements : list, material_id):
        self.elements = []
        for element_description in filterById(elements, material_id):
            self.elements.append(element(element_description))
            

def filterById(elements, material_id):
    filtered_elements = []
    for element_description in elements:
        if not isinstance(element_description, dict):
            raise Exception("Element should be a dictionary")
        if element_description["id"] == material_id:
            filtered_elements.append(element_description)
    return filtered_elements