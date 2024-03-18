import numpy as np

# import sys, os
# sys.path.insert(0, os.path.abspath('..'))


if __package__ == 'pydg1d.fdtd':
    from ..dgtd.spatialDiscretization import *
    from ..dgtd.mesh1d import Mesh1D
else:
    from dgtd.spatialDiscretization import *
    from dgtd.mesh1d import Mesh1D


class FDTD1D(SpatialDiscretization):
    def __init__(self, mesh: Mesh1D):
        