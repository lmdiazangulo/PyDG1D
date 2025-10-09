import numpy as np

class Mesh1D:
    def __init__(self, xmin, xmax, k_elem, boundary_label = "PEC"):
        
        assert k_elem > 0
        _, vx, _, EToV = mesh_generator(xmin, xmax, k_elem)

        self.dimension = 1
        self.vx = vx
        self.EToV = EToV
        self.boundary_label = boundary_label

        if type(self.boundary_label) == str:
            self.boundary_label = dict()
            self.boundary_label["LEFT"] = boundary_label
            self.boundary_label["RIGHT"] = boundary_label
        else:
            self.boundary_labels = boundary_label

        # if self.boundary_label not in ["PEC", "PMC", "SMA", "Periodic", "PML", "Mur", self.mixed_boundaries]:
        #     raise ValueError("Invalid boundary label.")

    def number_of_vertices(self):
        return self.vx.shape[0]

    def number_of_elements(self):
        return self.vx.shape[0] - 1
    
    def frontierIndexNeighbors(self):
        special_indices = {}
        frontier_condition_on_only_extremes = ["PEC", "PMC", "Periodic", "Mur"]
        for bdr, label in self.boundary_label.items():
            
            if bdr == "LEFT":
                if label in frontier_condition_on_only_extremes:
                    # special_indices.append(0)
                    special_indices[bdr] = [0]
                    

            if bdr == "RIGHT":
                if label in frontier_condition_on_only_extremes:
                    # special_indices.append(self.number_of_vertices()) 
                    special_indices[bdr] = [self.number_of_vertices()-1]

        return special_indices
    
    def getRelatedEvolutionNodes_H_map(self):
        # This map is equivalent to obtain the non zero values of the lasts len(number_of_vertices()) rows in the drived operator evolution,
        # i.e, the rows associated to the evolution of the magnetic field.

        relatedENodes_map = {}
        relatedHNodes_map = {}

        for i in range(self.number_of_elements()):
            relatedENodes_map[i] = [i, i+1]
            relatedHNodes_map[i] = [i]

        return relatedENodes_map, relatedHNodes_map
    
    def getRelatedEvolutionNodes_E_map(self):
        # Similar to the previous one, this dictionary/map corresponds to the evolution of the electric field

        frontiers = self.frontierIndexNeighbors()
        specialNodes = set(node for nodes in frontiers.values() for node in nodes)
        relatedENodes_map = {}
        relatedHNodes_map = {}

        relatedENodes_fromH, _ = self.getRelatedEvolutionNodes_H_map()

        for i in range(self.number_of_vertices()):
            if i in specialNodes:
                continue    
            else:
                relatedENodes_map[i] = list(set(relatedENodes_fromH.get(i-1, [])) | set(relatedENodes_fromH.get(i, [])))
                relatedHNodes_map[i] = [i-1, i]


        for bdr, label in self.boundary_label.items():
            nodes = frontiers.get(bdr, [])
            
            # I need to verify if the boundary condition from MUR is well implemented, I think it is not correct
            for i in nodes:
                if bdr == "LEFT":
                    if label == "PEC" or label == "PMC":
                        relatedENodes_map[i] = list(set(relatedENodes_fromH.get(i, [])))
                        relatedHNodes_map[i] = [i]

                    if label == "Periodic":
                        relatedENodes_map[i] = list(set(relatedENodes_fromH.get(i, [])) | set(relatedENodes_fromH.get(self.number_of_elements(), [])))
                        relatedHNodes_map[i] = [i, self.number_of_elements()]

                    if label == "Mur":
                        relatedENodes_map[i] = list(set(relatedENodes_fromH.get(i, [])) | set(relatedENodes_fromH.get(i+1, [])))
                        relatedHNodes_map[i] = [i, i+1]

                if bdr == "RIGHT":
                    if label == "PEC" or label == "PMC":
                        relatedENodes_map[i] = list(set(relatedENodes_fromH.get(i-1, [])))
                        relatedHNodes_map[i] = [i-1]

                    if label == "Periodic":
                        relatedENodes_map[i] = list(set(relatedENodes_fromH.get(self.number_of_elements() - i, [])) | set(relatedENodes_fromH.get(i, [])))
                        relatedHNodes_map[i] = [self.number_of_elements() - i, i]

                    if label == "Mur":
                        relatedENodes_map[i] = list(set(relatedENodes_fromH.get(i-1, [])) | set(relatedENodes_fromH.get(i, [])))
                        relatedHNodes_map[i] = [i-1, i]

        relatedENodes_map = dict(sorted(relatedENodes_map.items()))
        relatedHNodes_map = dict(sorted(relatedHNodes_map.items()))

        return relatedENodes_map, relatedHNodes_map
    

    def buildEvolutionOperator_Row_map(self):
        # Using the previous two maps, we can construct another one containing all the information of the non zero values in each row for the
        # drived Evolution operation
        A_E_rows_map = self.getRelatedEvolutionNodes_E_map()
        A_H_rows_map = self.getRelatedEvolutionNodes_H_map()
        row_map_E = {}
        row_map_H = {}
        row_map = {}

        for i in A_E_rows_map[0]:
            E_E_nodes = A_E_rows_map[0].get(i, [])
            E_H_nodes = [e_h_nodes + self.number_of_vertices() for e_h_nodes in A_E_rows_map[1].get(i, [])]
            row_map_E[i] = E_E_nodes + E_H_nodes

        for i in A_H_rows_map[0]:
            H_E_nodes = A_H_rows_map[0].get(i, [])
            H_H_nodes = [h_h_nodes + self.number_of_vertices() for h_h_nodes in A_H_rows_map[1].get(i, [])]
            row_map_H[i] = H_E_nodes + H_H_nodes

        for i in row_map_E:
            row_map[i] = row_map_E[i]

        for i in row_map_H:
            row_map[i + self.number_of_vertices()] = row_map_H[i]

        return row_map
    
    def buildEvolutionOperator_Column_map(self):
        # Finally, we construct another dictionary that contains the same information but the key items is now the columns, and this is the one used
        # for the information extraction from the outputs obtained with alternate vector basis.
        row_map = self.buildEvolutionOperator_Row_map()
        col_map = {}

        for row, cols in row_map.items():
            for col in cols:
                if col not in col_map:
                    col_map[col] = []
                col_map[col].append(row)

        col_map = dict(sorted(col_map.items()))

        return col_map



def mesh_generator(xmin,xmax,k_elem):
    """
    Generate simple equidistant grid with K elements
    >>> [Nv, vx, K, etov] = mesh_generator(0,10,4)
    >>> Nv
    5
    >>> vx_test = ([0.00000000,2.50000000,5.00000000,7.50000000,10.00000000])
    >>> np.allclose(vx,vx_test)
    True
    >>> K
    4
    >>> etov_test = ([0,1],[[1, 2],[2, 3],[3, 4]])
    >>> np.allclose(etov,etov_test)
    True
    """

    n_v = k_elem+1
    vx = np.linspace(xmin, xmax, num=n_v)
    
    #np.zeros creates a float array. etov should be an integer array
    EToV = np.full((k_elem,2),0)
    #etov = np.zeros([K,2])
    for i in range(k_elem):
        EToV[i,0] = i
        EToV[i,1] = i+1

    return [n_v,vx,k_elem,EToV]

