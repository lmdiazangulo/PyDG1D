class Material:
    def __init__(self, epsilon_r, mu_r, sigma):
        self.epsilon = epsilon_r
        self.mu = mu_r
        self.sigma = sigma

class ElementInterval:
    def __init__(self, start = int, end = int):
        self.start = start
        self.end = end

class MaterialMap:
    def __init__(self):
        self.matmap = []

    def add_material(self, interval = ElementInterval, material = Material):
        self.matmap.append([interval, material])

    def get_interval(self, index = int):
        return self.matmap[index][0]
    
    def get_material_for_matmap_index(self, index = int):
        return self.matmap[index][1]