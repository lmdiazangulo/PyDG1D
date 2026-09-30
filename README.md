# Python DGTD and FDTD experimental code.

[![Tests](https://github.com/lmdiazangulo/PyDG1D/actions/workflows/tests.yml/badge.svg)](https://github.com/lmdiazangulo/PyDG1D/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/111904725.svg)](https://doi.org/10.5281/zenodo.22115150)

Tests using `pytest` are available configuring in root directory.
The DGTD algorithms are python translations from the ones in

    Hesthaven, J. S., & Warburton, T. 
    Nodal discontinuous Galerkin methods: algorithms, analysis, and applications. 
    2007. Springer Science & Business Media.

## Lorentz (dispersive) media

The 1-D solvers support single pole Lorentz media

    dP/dt = J,  dJ/dt = -2 gamma J - omega_1^2 P + omega_p^2 E,
    D = epsilon_inf E + P

(Jiang et al., "Dispersion analysis of finite difference and discontinuous
Galerkin schemes for Maxwell's equations in linear Lorentz media", 2018)
through an optional `lorentz` argument with per element (DGTD) or per node
(FDTD) `omega_p`, `omega_1` and `gamma`; `epsilon` plays the role of
`epsilon_inf`.  The DGTD keeps the polarization equations as part of the RHS,
so the `LSERK` integrators can be used as usual.  For the FDTD the implicit
trapezoidal scheme of Eq. (4.2) of the paper is used
(`timeIntegratorType='TRAP'`); `P` and `J` are eliminated analytically and a
tridiagonal system is solved for the electric field.  A cold plasma is
`omega_1 = 0`, `gamma = nu_en / 2` and `epsilon_inf = 1` (with `c = 1` units,
see `test/test_lorentz_plasma_slab.py` for a conductor backed plasma slab
compared against the frequency domain reference).
