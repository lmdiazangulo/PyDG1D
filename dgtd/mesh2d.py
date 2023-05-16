import numpy as np


class Mesh2D:
    def __init__(self, vx, vy, EToV, boundary_label="PEC"):
        assert vx.shape == vy.shape
        assert np.max(np.max(EToV))+1  == vx.shape[0]

        self.vx = vx
        self.vy = vy
        self.EToV = EToV

        
        self.boundary_label = boundary_label

    def number_of_vertices(self):
        return self.vx.shape[0]

    def number_of_elements(self):
        return self.EToV.shape[0]
    

def readFromGambitFile(filename: str):
    DIMENSIONS_SECTION = 6
    COORDINATES_SECTION = 9

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find number of nodes and number of elements
    dims = lines[DIMENSIONS_SECTION].split()
    Nv = int(dims[0])
    Nk =  int(dims[1])
        
    # read node coordinates
    vx = np.zeros(Nv, dtype=float)
    vy = np.zeros(Nv, dtype=float)
    for i in range(Nv):
        splitted_line = lines[i + COORDINATES_SECTION].split()
        vx[i] = float(splitted_line[1])
        vy[i] = float(splitted_line[2])   
    
    # read element to node connectivity
    EToV = np.zeros((Nk, 3), dtype=int)
    for k in range(Nk):
        elements_section_begin = COORDINATES_SECTION + Nv + 2
        splitted_line = lines[k + elements_section_begin].split()
        EToV[k,0] = int(splitted_line[3]) - 1
        EToV[k,1] = int(splitted_line[4]) - 1
        EToV[k,2] = int(splitted_line[5]) - 1

    return Mesh2D(vx, vy, EToV)
