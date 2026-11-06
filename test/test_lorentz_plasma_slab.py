"""Reflection of a conductor backed Lorentz (plasma) slab in 1-D.

The time domain solvers are compared against the frequency domain reference
of a cold, collisional plasma slab covering a perfect conductor:

* C. X. Yuan, Z. X. Zhou and H. G. Sun, "Reflection Properties of
  Electromagnetic Wave in a Bounded Plasma Slab", IEEE Transactions on Plasma
  Science, 2010.  The relative permittivity is Eq. (10) of the paper,

      eps = 1 - omega_p^2 / (omega (omega - i nu_en)),

  and the reflection coefficient of a slab of thickness d over a conductor is
  Eq. (11), evaluated here with the same scikit-rf line model used by the
  ``plasma-slabs`` reference package.

* Y. Jiang, P. Sakkaplangkul, V. A. Bokil, Y. Cheng and F. Li, "Dispersion
  analysis of finite difference and discontinuous Galerkin schemes for
  Maxwell's equations in linear Lorentz media", 2018.  The single pole Lorentz
  model of the paper,

      dP/dt = J,  dJ/dt = -2 gamma J - omega_1^2 P + omega_p^2 E,
      D = eps_inf E + P,

  reduces to the cold plasma permittivity above with ``omega_1 = 0``,
  ``eps_inf = 1`` and ``gamma = nu_en / 2``.

PyDG1D works in units where the speed of light is one and lengths are metres,
so the angular frequencies are ``omega = 2 pi f / c``.
"""

import numpy as np

from maxwell.driver import *
from maxwell.dg.mesh1d import *
from maxwell.dg.dg1d import *
from maxwell.fd.fd1d import *

from skrf.frequency import Frequency
from skrf.media import Media


SPEED_OF_LIGHT = 299792458.0        # m/s
VACUUM_IMPEDANCE = 376.730313668    # ohm

PLASMA_FREQUENCY = 2.0e9            # Hz
COLLISION_FREQUENCY = 5.0e9         # s^-1
SLAB_THICKNESS = 0.10               # m

FREQUENCIES = np.linspace(2.0e9, 4.0e9, 201)

# Normalized (c = 1, lengths in metres) model parameters.
OMEGA_P = 2.0 * np.pi * PLASMA_FREQUENCY / SPEED_OF_LIGHT
GAMMA = 0.5 * COLLISION_FREQUENCY / SPEED_OF_LIGHT


class PlasmaMedia(Media):
    """Cold plasma as a scikit-rf media (Eq. (10) of Yuan2010)."""

    def __init__(self, frequency, plasma_frequency, collision_frequency,
                 **kwargs):
        super().__init__(frequency=frequency, **kwargs)
        self.plasma_frequency = plasma_frequency
        self.collision_frequency = collision_frequency

    @property
    def ep_r(self):
        omega = 2.0 * np.pi * self.frequency.f
        omega_p = 2.0 * np.pi * self.plasma_frequency
        return 1.0 - omega_p**2 / (omega * (omega - 1j * self.collision_frequency))

    @property
    def gamma(self):
        k0 = 2.0 * np.pi * self.frequency.f / SPEED_OF_LIGHT
        return 1j * k0 * np.sqrt(self.ep_r)

    @property
    def z0_characteristic(self):
        return VACUUM_IMPEDANCE / np.sqrt(self.ep_r)


def reference_reflection_db(frequencies=FREQUENCIES):
    """Reflected power [dB] of the conductor backed 10 cm plasma slab."""
    frequency = Frequency.from_f(np.asarray(frequencies, dtype=float),
                                 unit='hz')
    media = PlasmaMedia(frequency=frequency,
                        plasma_frequency=PLASMA_FREQUENCY,
                        collision_frequency=COLLISION_FREQUENCY)
    slab = media.line(d=SLAB_THICKNESS, unit='m') ** media.short()
    slab.renormalize(VACUUM_IMPEDANCE)
    return 10.0 * np.log10(np.abs(slab.s[:, 0, 0])**2)


def reflection_from_probe(time, signal, reference, dt, frequencies=FREQUENCIES):
    """Reflected power [dB] from the incident and total probe signals."""
    def dft(values, frequency):
        return np.sum(values * np.exp(
            -2j * np.pi * (frequency / SPEED_OF_LIGHT) * time) * dt)

    incident = np.array([dft(reference, f) for f in frequencies])
    reflected = np.array([dft(signal - reference, f) for f in frequencies])
    return 20.0 * np.log10(np.abs(reflected) / np.abs(incident))


def fdtd_probe(plasma, h=2.5e-4, tmax=1.6):
    """Run the trapezoidal FDTD scheme and return (time, E at the probe)."""
    xmin, xmax = -1.0, SLAB_THICKNESS if plasma else 1.0
    if plasma:
        boundary = {"LEFT": "Mur", "RIGHT": "PEC"}
    else:
        boundary = {"LEFT": "Mur", "RIGHT": "Mur"}

    mesh = Mesh1D(xmin, xmax, int(round((xmax - xmin) / h)),
                  boundary_label=boundary)

    lorentz = None
    if plasma:
        x = mesh.vx
        inside = (x >= 0.0) & (x <= SLAB_THICKNESS + 1e-12)
        lorentz = {"omega_p": np.where(inside, OMEGA_P, 0.0),
                   "gamma": np.where(inside, GAMMA, 0.0)}

    sp = FD1D(mesh, lorentz=lorentz)
    driver = MaxwellDriver(sp, timeIntegratorType='TRAP', CFL=0.9)

    source_x, probe_x = -0.2, -0.1
    i_source = int(round((source_x - xmin) / h))
    i_probe = int(round((probe_x - xmin) / h))

    source_width, source_delay = 0.02, 0.1
    steps = int(tmax / driver.dt)
    time = np.zeros(steps)
    signal = np.zeros(steps)
    for n in range(steps):
        driver.step()
        t = driver.timeIntegrator.time
        driver['E'][i_source] += np.exp(
            -((t - source_delay) / source_width)**2 / 2.0)
        time[n] = t
        signal[n] = driver['E'][i_probe]

    return time, signal, driver.dt


def dgtd_probe(plasma, n_order=2, tmax=2.5):
    """Run the DG scheme with the Lorentz RHS and return (time, E at probe)."""
    if plasma:
        xmax, elements = 1.0, 200
        boundary = {"LEFT": "ABC", "RIGHT": "PEC"}
    else:
        xmax, elements = 3.0, 600
        boundary = {"LEFT": "ABC", "RIGHT": "ABC"}

    lorentz = None
    if plasma:
        elements_in_slab = int(round(SLAB_THICKNESS / (xmax / elements)))
        omega_p = np.zeros(elements)
        gamma = np.zeros(elements)
        omega_p[-elements_in_slab:] = OMEGA_P
        gamma[-elements_in_slab:] = GAMMA
        lorentz = {"omega_p": omega_p, "gamma": gamma}

    sp = DG1D(n_order, Mesh1D(0.0, xmax, elements, boundary_label=boundary),
              lorentz=lorentz)
    driver = MaxwellDriver(sp, timeIntegratorType='LSERK4', CFL=1.0)

    source_x, source_width = 0.25, 0.02
    driver['E'][:] = np.exp(-(sp.x - source_x)**2 / (2 * source_width**2))
    driver['H'][:] = driver['E']

    # Probe at x = 0.6 (element 120 at the coarsest mesh), in vacuum.
    element, node = int(round(0.6 / (xmax / elements))), 0

    steps = int(tmax / driver.dt)
    time = np.zeros(steps)
    signal = np.zeros(steps)
    for n in range(steps):
        driver.step()
        time[n] = (n + 1) * driver.dt
        signal[n] = driver['E'][node, element]

    return time, signal, driver.dt


def test_lorentz_drude_mapping():
    """gamma = nu/2, omega_1 = 0 reproduces the cold plasma permittivity.

    With the ``exp(-i omega t)`` convention of the time domain solver the
    permittivity of the ODE is ``1 - omega_p^2 / (omega^2 + i nu omega)``,
    the complex conjugate of the ``exp(+i omega t)`` (scikit-rf) permittivity
    used for the frequency domain reference.
    """
    frequency = 3.0e9
    omega = 2.0 * np.pi * frequency / SPEED_OF_LIGHT
    nu = COLLISION_FREQUENCY / SPEED_OF_LIGHT

    eps_lorentz = 1.0 - OMEGA_P**2 / (omega**2 + 2j * GAMMA * omega)
    eps_drude = 1.0 - OMEGA_P**2 / (omega * (omega - 1j * nu))

    assert np.isclose(eps_lorentz, np.conj(eps_drude))
    assert np.isclose(np.conj(eps_lorentz).imag, -nu * OMEGA_P**2
                      / (omega * (omega**2 + nu**2)))


def _rk4_step(sp, fields, dt, substeps=50):
    """High accuracy integration of the semi-discrete RHS (reference)."""
    state = {label: f.copy() for label, f in fields.items()}
    h = dt / substeps
    for _ in range(substeps):
        k1 = sp.computeRHS(state)
        k2 = sp.computeRHS({l: state[l] + 0.5*h*k1[l] for l in state})
        k3 = sp.computeRHS({l: state[l] + 0.5*h*k2[l] for l in state})
        k4 = sp.computeRHS({l: state[l] + h*k3[l] for l in state})
        for label in state:
            state[label] = state[label] + \
                h/6.0*(k1[label] + 2*k2[label] + 2*k3[label] + k4[label])
    return state


def test_trapezoidal_step_matches_lorentz_rhs():
    """The implicit elimination of Eq. (4.2) must integrate the ADE RHS.

    This covers the resonant (omega_1 != 0) single pole Lorentz model; the
    slab test only excites the Drude limit (omega_1 = 0).
    """
    omega_p, omega_1, gamma, eps_inf = 3.0, 2.0, 0.4, 2.0

    sp = FD1D(Mesh1D(0.0, 1.0, 25, boundary_label="PEC"),
              epsilon=eps_inf,
              lorentz={"omega_p": omega_p, "omega_1": omega_1,
                       "gamma": gamma})

    fields = sp.buildFields()
    fields['E'][:] = np.sin(np.pi * sp.x)
    fields['H'][:] = 0.5 * np.cos(2.0 * np.pi * sp.xH)
    fields['P'][:] = 0.3 * np.sin(2.0 * np.pi * sp.x)
    fields['J'][:] = -0.2 * np.cos(np.pi * sp.x)

    dt = 1e-4
    expected = _rk4_step(sp, fields, dt)

    step = {label: f.copy() for label, f in fields.items()}
    sp.computeTrapezoidalStep(step, dt)

    for label in fields:
        assert np.allclose(expected[label], step[label], atol=1e-5), label


def test_fdtd_lorentz_plasma_slab_reflection():
    time, reference, dt = fdtd_probe(plasma=False)
    _, signal, _ = fdtd_probe(plasma=True)

    reflected_db = reflection_from_probe(time, signal, reference, dt)
    expected_db = reference_reflection_db()

    deviation = np.abs(reflected_db - expected_db)
    assert np.max(deviation) < 0.5

    # The Fabry-Perot dip of the 10 cm slab.
    assert abs(FREQUENCIES[np.argmin(reflected_db)] - 2.34e9) < 0.05e9
    assert abs(np.min(reflected_db) - np.min(expected_db)) < 1.0


def test_dgtd_lorentz_plasma_slab_reflection():
    time, reference, dt = dgtd_probe(plasma=False)
    _, signal, _ = dgtd_probe(plasma=True)

    reflected_db = reflection_from_probe(time, signal, reference, dt)
    expected_db = reference_reflection_db()

    deviation = np.abs(reflected_db - expected_db)
    assert np.max(deviation) < 0.3

    assert abs(FREQUENCIES[np.argmin(reflected_db)] - 2.34e9) < 0.05e9
    assert abs(np.min(reflected_db) - np.min(expected_db)) < 1.0
