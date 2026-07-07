import pathlib
import sys

import numpy as np

from maxwell.dg.dg2d import Maxwell2D
from maxwell.dg.mesh2d import readFromGambitFile
from maxwell.driver import MaxwellDriver

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from dissipative import build_continuous_time_KYP, build_discrete_time_KYP, ev


TEST_DATA_FOLDER = "testData/"


def build_periodic_centered_space(mesh_name, order=1):
    mesh = readFromGambitFile(TEST_DATA_FOLDER + mesh_name)
    mesh.boundary_label = "Periodic"
    return Maxwell2D(order, mesh, "Centered")


def test_periodic_maps_remove_boundary_faces_k8():
    sp = build_periodic_centered_space("Maxwell2D_K8.neu", order=1)

    assert sp.vmapB.size == 0
    assert sp.mapB.size == 0


def test_periodic_neighbor_indices_k146():
    sp = build_periodic_centered_space("Maxwell2D_K146.neu", order=1)

    local_indices, neigh_indices = sp.buildLocalAndNeighborIndices(0, 1)

    assert local_indices.size == 3 * sp.number_of_nodes_per_element()
    assert neigh_indices.size > 0
    assert np.intersect1d(local_indices, neigh_indices).size == 0
    assert np.unique(neigh_indices).size == neigh_indices.size
    assert sp.vmapB.size == 0


def test_continuous_global_and_local_kyp_k8():
    sp = build_periodic_centered_space("Maxwell2D_K8.neu", order=1)
    Ag = sp.reorder_by_elements(sp.buildEvolutionOperator())
    Mg = sp.reorder_by_elements(sp.buildGlobalMassMatrix())
    Pg = 0.5 * Mg

    ATPPA = Ag.T.dot(Pg) + Pg.dot(Ag)
    assert np.all(np.isfinite(Ag))
    assert np.allclose(ATPPA, 0.0, atol=1e-10)

    A, B, C, D, Mk, Mn = sp.buildConnectedOperators(0, 1)
    P = 0.5 * Mk
    Q = np.zeros_like(Mn)
    S = -0.5 * Mn
    R = np.zeros_like(Mn)

    KYP = build_continuous_time_KYP(A, B, C, D, P, Q, S, R)
    assert np.allclose(KYP, 0.0, atol=1e-10)
    assert np.isclose(np.max(np.real(ev(KYP))), 0.0, atol=1e-10)
    assert np.allclose(A.T.dot(P) + P.dot(A), 0.0, atol=1e-10)
    assert np.allclose(P.dot(B) - C.T.dot(S), 0.0, atol=1e-10)
    assert np.allclose(-D.T.dot(S) - S.dot(D), 0.0, atol=1e-10)


def test_discrete_global_and_local_kyp_k8():
    sp = build_periodic_centered_space("Maxwell2D_K8.neu", order=1)
    driver = MaxwellDriver(sp, CFL=0.5)

    G = sp.reorder_by_elements(driver.buildDrivedEvolutionOperator())
    M = sp.reorder_by_elements(sp.buildGlobalMassMatrix())
    P = 0.5 * M

    GTPGP = G.T.dot(P).dot(G) - P
    assert np.all(np.isfinite(G))
    assert np.max(np.abs(ev(G))) <= 1.0 + 1e-10
    assert np.max(np.real(ev(GTPGP))) <= 1e-10

    A, B, C, D, Mk, Mn = driver.buildCausallyConnectedOperators(0, -1)
    Pk = 0.5 * Mk
    Q = -0.5 * Mn
    S = np.zeros_like(Mn)
    R = 0.5 * Mn

    KYP = build_discrete_time_KYP(A, B, C, D, Pk, Q, S, R)
    assert np.max(np.real(ev(KYP))) <= 1e-10


def _central_element(mesh):
    x = mesh.vx[mesh.EToV]
    y = mesh.vy[mesh.EToV]
    centroids = np.column_stack([x.mean(axis=1), y.mean(axis=1)])
    centre = np.array([mesh.vx.mean(), mesh.vy.mean()])
    return int(np.argmin(np.sum((centroids - centre) ** 2, axis=1)))


def test_continuous_local_kyp_k146():
    """Centered-flux DGTD is energy conserving in 2D: the continuous-time KYP
    identity holds exactly on an unstructured periodic mesh."""
    sp = build_periodic_centered_space("Maxwell2D_K146.neu", order=1)
    element = _central_element(sp.mesh)

    Ag = sp.reorder_by_elements(sp.buildEvolutionOperator())
    Mg = sp.reorder_by_elements(sp.buildGlobalMassMatrix())
    Pg = 0.5 * Mg
    assert np.allclose(Ag.T.dot(Pg) + Pg.dot(Ag), 0.0, atol=1e-9)

    A, B, C, D, Mk, Mn = sp.buildConnectedOperators(element, 1)
    P = 0.5 * Mk
    Q = np.zeros_like(Mn)
    S = -0.5 * Mn
    R = np.zeros_like(Mn)

    KYP = build_continuous_time_KYP(A, B, C, D, P, Q, S, R)
    assert np.allclose(KYP, 0.0, atol=1e-9)
    assert np.isclose(np.max(np.real(ev(KYP))), 0.0, atol=1e-9)


def test_discrete_local_kyp_k146():
    """The discrete-time local KYP certificate holds for a central element and
    its N_STAGES-hop causal halo at a stable CFL on the K146 periodic mesh."""
    sp = build_periodic_centered_space("Maxwell2D_K146.neu", order=1)
    element = _central_element(sp.mesh)
    driver = MaxwellDriver(sp, timeIntegratorType="LSERK4", CFL=0.3)

    A, B, C, D, Mk, Mn = driver.buildCausallyConnectedOperators(element, -1)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(D))

    Pk = 0.5 * Mk
    Q = -0.5 * Mn
    S = np.zeros_like(Mn)
    R = 0.5 * Mn

    KYP = build_discrete_time_KYP(A, B, C, D, Pk, Q, S, R)
    assert np.max(np.real(ev(KYP))) <= 1e-9
