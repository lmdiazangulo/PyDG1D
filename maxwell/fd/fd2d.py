import numpy as np

from ..spatialDiscretization import *

class FD2D(SpatialDiscretization):  # TE mode
    def __init__(self, x_min, x_max, kx_elem, y_min=0.0, y_max=0.0, ky_elem=0, boundary_labels="PEC"):
        
        if type(boundary_labels) == str:
            self.boundary_labels = dict()
            self.boundary_labels["XL"] = boundary_labels
            self.boundary_labels["XU"] = boundary_labels
            self.boundary_labels["YL"] = boundary_labels
            self.boundary_labels["YU"] = boundary_labels
        else:
            self.boundary_labels = boundary_labels

        if y_min == 0.0 and y_max == 0.0 and ky_elem == 0:
            y_min = x_min
            y_max = x_max
            ky_elem = kx_elem
        else:
            raise ValueError("Invalid values for y grid planes.")

        self.x = np.linspace(x_min, x_max, num=kx_elem+1)
        self.y = np.linspace(y_min, y_max, num=ky_elem+1)
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)

        self.xH = (self.x[:-1] + self.x[1:]) / 2.0
        self.yH = (self.y[:-1] + self.y[1:]) / 2.0

        self.dxH = self.xH[1:] - self.xH[:-1]

        self.cEy = 1.0 / self.dy[0]
        self.cEx = 1.0 / self.dx[0]

        self.c0 = 1.0
        self.tfsf = False
        self.source = None

    def TFSF_conditions(self, setup):

        self.tfsf =  True
        self.source = setup["source"]
        self.XL_TF_limit = (np.absolute(self.x - setup["XL"])).argmin() #por que se pone naranja?
        self.XU_TF_limit = (np.absolute(self.x - setup["XU"])).argmin()
        self.YL_TF_limit = (np.absolute(self.y - setup["YL"])).argmin()
        self.YU_TF_limit = (np.absolute(self.y - setup["YU"])).argmin()
        if not "source" in setup.keys() or not "XL" in setup.keys() or not "XU" in setup.keys()\
            or not "YL" in setup.keys() or not "YU" in setup.keys():
            raise ValueError('Missing TFSF setup variables')

    def buildFields(self):
        H = np.zeros((len(self.dy), len(self.dx)))
        Ex = np.zeros((len(self.y),  len(self.dx)))
        Ey = np.zeros((len(self.dy), len(self.x)))

        if (self.source != None and self.tfsf):
            self.buildIncidentFields()

        return {
            "E": {"x": Ex, "y": Ey},
            "H": H
        }
    
    def buildIncidentFields(self):
        
        self.xH_inc, self.yH_inc = np.meshgrid(self.xH, self.yH)
        self.x_inc, self.y_inc = np.meshgrid(self.x, self.y)

        self.Einc = np.ndarray(self.x_inc.shape)
        self.Einc[:,:] = self.source(self.x_inc[:,:])

        self.Eprev = np.zeros(self.x_inc.shape)
        
        self.Hinc = np.ndarray(self.xH_inc.shape)
        self.Hinc[:,:] = self.source(self.xH_inc[:,:] - 0.5*self.dt)
        

    def get_minimum_node_distance(self):
        return np.min(self.dx)
    

    def computeRHSE(self, fields):
        H = fields['H']
        Ex = fields['E']['x']
        Ey = fields['E']['y']

        rhsEx = np.zeros(Ex.shape)
        rhsEy = np.zeros(Ey.shape)

        rhsEx[1:-1, :] =   self.cEy * ( H[1:, :] - H[:-1, :])
        rhsEy[:, 1:-1] = - self.cEx * ( H[:, 1:] - H[:, :-1])

        if self.tfsf == True:

            self.updateIncidentFieldE()
            rhsEy[self.XL_TF_limit]  +=  (1.0/self.dxH[0]) * self.Hinc[self.XL_TF_limit-1]
            rhsEy[self.XU_TF_limit] -=  (1.0/self.dxH[0]) * self.Hinc[self.XU_TF_limit ]

            rhsEx[self.YL_TF_limit]  +=  (1.0/self.dxH[0]) * self.Hinc[self.YL_TF_limit-1]
            rhsEx[self.YU_TF_limit] -=  (1.0/self.dxH[0]) * self.Hinc[self.YU_TF_limit]



        for bdr, label in self.boundary_labels.items():
            if bdr == "XL":
                if label == "PEC":
                    rhsEy[:,  0] = 0.0
                elif label == "PMC":
                    rhsEy[:, 0] =  - self.cEx * (2*H[:,0])
                elif label == "Mur":  #Mur esta fallando

                    rhsEy[:, 0] = Ey[:, 1] + \
                    (self.c0 * self.dt - self.dy[0]) / \
                    (self.c0 * self.dt + self.dy[0]) * \
                    (rhsEy[:,1] - Ey[:,0])
                    
                    rhsEy[:, 0] -= Ey[:, 0]
                    rhsEy[:, 0] /= self.dt  

            elif bdr == "XU":
                if label == "PEC":
                    rhsEy[:, -1] = 0.0
                elif label == "PMC":
                    rhsEy[:, -1] =  - self.cEx * (-2*H[:,-1])
                elif label == "Mur":

                    rhsEy[:, -1] = Ey[:, -2] + \
                    (self.c0 * self.dt - self.dy[0]) / \
                    (self.c0 * self.dt + self.dy[0]) * \
                    (rhsEy[:,-2] - Ey[:,-1])
                    
                    rhsEy[:, -1] -= Ey[:, -1]
                    rhsEy[:, -1] /= self.dt  
                
            elif bdr == "YL":
                if label == "PEC":
                    rhsEx[0, :] = 0.0
                elif label == "PMC":
                    rhsEx[0, :] =  self.cEy * (2*H[0,:])
                elif label == "Mur":

                    rhsEx[0, :] = Ex[1, :] + \
                    (self.c0 * self.dt - self.dx[0]) / \
                    (self.c0 * self.dt + self.dx[0]) * \
                    (rhsEx[1, :] - Ex[0, :])
                    
                    rhsEx[0, :] -= Ex[0, :]
                    rhsEx[0, :] /= self.dt  

            elif bdr == "YU":
                if label == "PEC":
                    rhsEx[-1, :] = 0.0
                elif label == "PMC":
                    rhsEx[-1, :] =  self.cEy * (-2*H[-1,:])
                elif label == "Mur":
                    rhsEx[-1, :] = Ex[-2, :] + \
                    (self.c0 * self.dt - self.dx[0]) / \
                    (self.c0 * self.dt + self.dx[0]) * \
                    (rhsEx[-2, :] - Ex[-1, :])
                    
                    rhsEx[-1, :] -= Ex[-1, :]
                    rhsEx[-1, :] /= self.dt
            else:
                raise ValueError("Invalid boundary tag.")       

        return {'x': rhsEx, 'y': rhsEy}

    def computeRHSH(self, fields):
        Ex = fields['E']['x']
        Ey = fields['E']['y']

        rhsH = + self.cEy*(Ex[1:, :] - Ex[:-1, :]) \
               - self.cEx*(Ey[:, 1:] - Ey[:, :-1])
        
        if self.tfsf == True:  #esto como seria???
            self.updateIncidentFieldH()


            rhsH[self.left_TF_limit - 1] +=  (1.0/self.dx[0]) * self.Einc[self.left_TF_limit]
            rhsH[self.right_TF_limit]    -=  (1.0/self.dx[0]) * self.Einc[self.right_TF_limit]



            rhsH = + self.cEy*(self.Einc[self.XL_TF_limit -1, :] - self.Einc[self.XL_TF_limit, :]) \
               - self.cEx*(Ey[:, 1:] - Ey[:, :-1])
            
            rhsH = + self.cEy*(Ex[1:, :] - Ex[:-1, :]) \
               - self.cEx*(Ey[:, 1:] - Ey[:, :-1])
            
            rhsH = + self.cEy*(Ex[1:, :] - Ex[:-1, :]) \
               - self.cEx*(Ey[:, 1:] - Ey[:, :-1])
            
            rhsH = + self.cEy*(Ex[1:, :] - Ex[:-1, :]) \
               - self.cEx*(Ey[:, 1:] - Ey[:, :-1])

        return rhsH

    def computeRHS(self, fields):
        rhsE = self.computeRHSE(fields)
        rhsH = self.computeRHSH(fields)

        return {'E': rhsE, 'H': rhsH}
    
    def updateIncidentFieldE(self):
        self.Einc[1:-1,:] = self.Einc[1:-1,:] - self.dt*(1.0/self.dxH[0]) * (self.Hinc[1:,:] - self.Hinc[:-1,:])
            
        self.Einc[0,:] = \
            self.Eprev[1] - \
            (self.c0 * self.dt - self.dx[0]) / \
            (self.c0 * self.dt + self.dx[0]) * \
            (self.Einc[1,:] - self.Eprev[0,:])

        self.Einc[-1,:] = \
            self.Eprev[-2,:] - \
            (self.c0 * self.dt - self.dx[0]) / \
            (self.c0 * self.dt + self.dx[0]) * \
            (self.Einc[-2,:] - self.Eprev[-1,:])

        self.Eprev[:,:] = self.Einc[:,:]

    def updateIncidentFieldH(self):
        self.Hinc = self.Hinc - self.dt*(1.0/self.dx[0]) * (self.Einc[1:,:] - self.Einc[:-1,:])

#··································································································
       

    def isStaggered(self):
        return True

    def dimension(self):
        return 2
