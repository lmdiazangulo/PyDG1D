import numpy as np

N_FACES = 3

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
    
    def connectivityMatrices(self):
        '''
            function [EToE,EToF]= connectivity(self)
            Purpose: triangle face connect algorithm due to Toby Isaac
        '''
        K = self.number_of_elements()
        Nnodes = self.number_of_vertices()
                
        # create list of all faces 0, then 1, & 2
        fnodes = np.concatenate(
            (self.EToV[:,[0,1]], 
             self.EToV[:,[1,2]], 
             self.EToV[:,[2,0]])
        )
        fnodes = np.sort(fnodes, 1)

        # set up default element to element and Element to faces connectivity
        EToE = np.outer(np.arange(0, K, 1, dtype=int), np.ones((1,N_FACES)))
        EToF = np.outer(np.ones((K,1)),                np.arange(0, N_FACES, 1, dtype=int))

        # uniquely number each set of three faces by their node numbers 
        id = fnodes[:,0] * Nnodes + fnodes[:,1] + 1
        spNodeToNode = np.concatenate((
            id.reshape(id.size, 1), 
            np.arange(0, N_FACES*K, 1, dtype=int).reshape(N_FACES*K, 1), 
            EToE.reshape(N_FACES*K, 1, order='F'), 
            EToF.reshape(N_FACES*K, 1, order='F')
        ), 1)
        spNodeToNode = np.int64(spNodeToNode)

        # Now we sort by global face number.
        sorted= spNodeToNode[np.argsort(spNodeToNode[:,0]), :]

        # find matches in the sorted face list
        indices = np.where(sorted[:-1,0]==sorted[1:, 0])[0]

        # make links reflexive 
        matchL = np.concatenate((sorted[indices,  :], sorted[indices+1,:]))
        matchR = np.concatenate((sorted[indices+1,:], sorted[indices,  :]))

        # insert matches
        EToE.transpose().flat[matchL[:,1]] = matchR[:,2]
        EToF.transpose().flat[matchL[:,1]] = matchR[:,3]

        return EToE, EToF

    

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
