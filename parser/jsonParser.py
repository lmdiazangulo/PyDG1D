import json
import numpy as np
import logging
from utils import *

class Parser:
    def __init__(self, filename):
        self.filename = filename
        self.json = json.load(open(filename))
        self.general = General()
        self.grid = Grid()
        
    def readProblemDescription(self):
        self.readGeneral()
        self.readGrid()
        # self.boundaries = self._readBoundary()
        # self.coordinates = self._readCoordinates()

        # self.materials = self.readMaterials()

        # self.sources = self.readSources()
        # self.probes = self.readProbes()
        
    def readGeneral(self):
        
        try:
            self.general.timeStep = self.json["general"]["timeStep"]
            self.general.numberOfSteps = self.json["general"]["numberOfSteps"]
        except KeyError:
            logging.error('Problem reading section "General"  in json file')
                    
    def readGrid(self):
        try:
            grid = self.json["mesh"]["grid"]
            self.grid.dx = np.ones(grid["numberOfCells"][DIR_X]) * grid["steps"][TAG_X]
            self.grid.dy = np.ones(grid["numberOfCells"][DIR_Y]) * grid["steps"][TAG_Y]
            self.grid.dz = np.ones(grid["numberOfCells"][DIR_Z]) * grid["steps"][TAG_Z]
            
        except KeyError:
            logging.error('Problem reading section "Grid"  in json file')

    # def readBoundary(self):
    #     pass
    # def readCoordinates(self):
    #     pass
    # def readMaterials(self):
    #     pass
    # def readSources(self):
    #     pass
    # def readProbes(self):
    #     pass
    