# Paper Supplement: Illustrative 2D Simulations

Draft text for the manuscript and response letter (Reviewers 2 and 3).
Adapt section numbers to match the final manuscript layout.

---

## Manuscript — new subsection (suggested title)

### Illustrative two-dimensional simulations

The local stability analysis in Sections 4–5 is restricted to the
one-dimensional transverse-curl Maxwell system with periodic boundaries.
To illustrate that the underlying solvers extend beyond this setting, we
include two supplementary time-domain examples computed with the PyDG1D
reference implementation [40].

**DGTD — PEC cavity.** A nodal DGTD scheme (`Maxwell2D`) on a triangular
mesh of a unit-square cavity with PEC walls is evolved with a low-storage
explicit Runge-Kutta integrator. The electric field \(E_z\) is initialized
with the dominant analytical resonant mode (Hesthaven & Warburton, p. 205).
The numerical solution remains correlated with the analytical mode over many
periods, confirming stable time integration on an unstructured 2D mesh.

**FDTD — TE Gaussian pulse.** A structured finite-difference TE formulation
(`FD2D`) on a square domain with PEC boundaries propagates a Gaussian pulse
in the magnetic field component \(H_z\). The pulse remains bounded and
physically consistent over the simulation window.

These examples are **illustrative only**: no two-dimensional local KYP
stability bounds are derived or validated here. Extension of the dissipative-
systems machinery to 2D unstructured elements—with multiple faces per element,
larger causal stencils under Runge–Kutta time integration, and non-trivial
local energy definitions—remains future work (see Conclusions).

**Suggested figure caption:** *Illustrative 2D simulations not included in
the 1D stability analysis. Left: DGTD \(E_z\) in a PEC cavity. Right:
FDTD TE \(H_z\) snapshots of a Gaussian pulse. Generated with PyDG1D
(`examples/dg2d_pec_cavity.py`, `examples/fd2d_te_pulse.py`).*

---

## Manuscript — outlook paragraph (Reviewer 3, comment 1)

In two spatial dimensions each mesh element has three faces and multiple
neighbors; the causal stencil of an \(s\)-stage explicit Runge–Kutta
integrator couples a element to a wider halo than in 1D. Defining a local
storage function that decomposes the global electromagnetic energy across
unstructured elements is substantially more involved than in the Yee FDTD
case (cf. Section 5.1.3) and remains open for DGTD with leapfrog
integration (cf. Section 5.2.2). A practical first step toward higher
dimensions is to verify operator assembly and time stepping on small 2D
meshes—as in the illustrative examples above—before attempting local
QSR-dissipativity tests with face-based supply rates.

---

## Response letter — Reviewer 2, comment 3

**Authors' Response:** We thank the reviewer for this suggestion. We have
added illustrative two-dimensional time-domain examples (DGTD PEC cavity and
FDTD TE pulse) computed with our open-source PyDG1D implementation. These
simulations demonstrate that the spatial discretizations and explicit time
integrators operate on 2D domains but **do not** extend the local KYP
stability analysis, which remains one-dimensional in this work. The new
subsection and figures are highlighted in blue; scope statements in the
Introduction, Section 3, and Conclusions are unchanged in substance.

**Changes in the manuscript:** New subsection on illustrative 2D simulations;
one new figure; reference to PyDG1D example scripts.

---

## Response letter — Reviewer 3, comment 1

**Authors' Response:** We agree that the present analysis is restricted to
1D periodic problems and that practical relevance in 2D requires additional
work. We have expanded the Conclusions and added an outlook paragraph
discussing how local regions would be defined on triangular meshes (multiple
faces, enlarged RK causal stencils, open local-energy questions). We also
include illustrative 2D simulations to show that the reference code supports
unstructured DGTD and structured FDTD time stepping, without claiming 2D
local stability certificates.

**Changes in the manuscript:** Outlook paragraph after the illustrative 2D
subsection; expanded Conclusions on higher-dimensional limitations.

---

## Reproducibility

From the repository root:

```bash
pip install -r requirements.txt
python examples/dg2d_pec_cavity.py --mesh Maxwell2D_K8.neu --output examples/output
python examples/fd2d_te_pulse.py --output examples/output
python -m pytest test/test_dgtd_2d.py -q
```

Tier-2 operator tooling on small 2D meshes:

```bash
python -m pytest test/dg/test_maxwell2d.py::test_build_drived_evolution_operator_k2 -q
python -m pytest test/dg/test_maxwell2d.py::test_get_energy_k2 -q
```
