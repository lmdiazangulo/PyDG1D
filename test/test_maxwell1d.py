
from dgtd.maxwell1d import *
from dgtd.mesh1d import *

def test_get_energy_N1():
    m = Mesh1D(0, 1, 10)
    sp = Maxwell1D(1, m)
    fields = sp.buildFields()
    
    fields['E'].fill(0.0)
    fields['E'][0,0] = 1.0
    assert np.isclose(sp.getEnergy(fields['E']), 0.1*1.0/3.0, rtol=1e-9)

    fields['E'].fill(1.0)
    assert np.isclose(sp.getEnergy(fields['E']),         1.0, rtol=1e-9)
